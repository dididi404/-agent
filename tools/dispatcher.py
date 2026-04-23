"""Tool Dispatcher — 确定性路由，将 tool_intent 映射到具体工具并执行完整管道"""

import hashlib
import json
import time
import uuid

from schemas.common import RiskLevel, ToolStatus
from schemas.trial import ToolCallRecord, ToolExecutionRequest, ToolExecutionResult
from tools.base import BaseTool
from tools.guard import ToolGuard
from tools.registry import ToolRegistry

INTENT_TO_TOOL: dict[str, str] = {
    "inspect_repo": "inspect_repo",
    "read_file": "read_file",
    "config_read": "read_config",
    "config_update": "patch_config",
    "launch_train": "launch_train",
    "read_metrics": "read_metrics",
    "read_logs": "read_logs",
    "compare_runs": "compare_runs",
}


class ToolDispatcher:
    def __init__(self, registry: ToolRegistry, guard: ToolGuard):
        self.registry = registry
        self.guard = guard
        self.call_records: list[ToolCallRecord] = []

    def resolve_tool(self, intent: str) -> str | None:
        return INTENT_TO_TOOL.get(intent)

    def dispatch(
        self,
        intent: str,
        params: dict,
        caller_agent: str = "",
        intent_description: str = "",
    ) -> ToolExecutionResult:
        request_id = f"req_{uuid.uuid4().hex[:8]}"

        tool_name = self.resolve_tool(intent)
        if tool_name is None:
            return ToolExecutionResult(
                request_id=request_id,
                tool_name=intent,
                status=ToolStatus.FAILED,
                error_message=f"Unknown tool_intent: '{intent}'. Valid: {list(INTENT_TO_TOOL.keys())}",
            )

        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolExecutionResult(
                request_id=request_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error_message=f"Tool '{tool_name}' not registered",
            )

        allowed, reason = self.guard.check(tool, params)
        if not allowed:
            return ToolExecutionResult(
                request_id=request_id,
                tool_name=tool_name,
                status=ToolStatus.REJECTED,
                error_message=f"Guard rejected: {reason}",
            )

        input_schema = tool.meta.input_schema
        try:
            validated_input = input_schema(**params)
        except Exception as e:
            return ToolExecutionResult(
                request_id=request_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error_message=f"Input validation failed: {e}",
            )

        pre_ok, pre_msg = tool.check_preconditions(validated_input)
        if not pre_ok:
            return ToolExecutionResult(
                request_id=request_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error_message=f"Precondition failed: {pre_msg}",
            )

        start_time = time.time()
        try:
            result_obj = tool.execute(validated_input)
        except TimeoutError:
            elapsed = time.time() - start_time
            return ToolExecutionResult(
                request_id=request_id,
                tool_name=tool_name,
                status=ToolStatus.TIMEOUT,
                runtime_sec=elapsed,
                error_message="Tool execution timed out",
            )
        except Exception as e:
            elapsed = time.time() - start_time
            return ToolExecutionResult(
                request_id=request_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                runtime_sec=elapsed,
                error_message=str(e),
            )
        elapsed = time.time() - start_time

        post_ok, post_msg = tool.post_validate(result_obj)
        status = ToolStatus.SUCCESS if post_ok else ToolStatus.SUSPICIOUS

        result_dict = result_obj.model_dump()
        artifacts = {}
        for key in ("log_path", "metrics_path", "backup_path"):
            if key in result_dict and result_dict[key]:
                artifacts[key] = str(result_dict[key])

        exec_result = ToolExecutionResult(
            request_id=request_id,
            tool_name=tool_name,
            status=status,
            return_code=result_dict.get("return_code"),
            artifacts=artifacts,
            stdout_summary=result_dict.get("stdout_tail", "")[:500],
            stderr_summary=result_dict.get("stderr_tail", "")[:500],
            runtime_sec=elapsed,
            post_validation_passed=post_ok,
            error_message=post_msg if not post_ok else None,
        )

        input_hash = hashlib.md5(
            json.dumps(params, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

        self.call_records.append(
            ToolCallRecord(
                request_id=request_id,
                tool_name=tool_name,
                caller_agent=caller_agent,
                intent=intent_description or intent,
                input_hash=input_hash,
                output_status=exec_result.status,
                latency_ms=int(elapsed * 1000),
                post_validation_passed=post_ok,
            )
        )

        return exec_result