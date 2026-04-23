"""metric_tools — read_metrics / read_logs / compare_runs 工具实现"""

import json
import re
from pathlib import Path
from typing import Any

from schemas.common import RiskLevel
from tools.base import BaseTool, ToolMeta
from tools.schemas import (
    CompareRunsInput,
    CompareRunsOutput,
    ReadLogsInput,
    ReadLogsOutput,
    ReadMetricsInput,
    ReadMetricsOutput,
    RunSummary,
)


class ReadMetricsTool(BaseTool):
    meta = ToolMeta(
        name="read_metrics",
        description="读取 JSON 格式的指标文件并提取关键指标",
        input_schema=ReadMetricsInput,
        output_schema=ReadMetricsOutput,
        risk_level=RiskLevel.LOW,
        timeout_sec=10,
        is_idempotent=True,
        preconditions=["metrics_path must exist"],
    )

    def check_preconditions(self, params: ReadMetricsInput) -> tuple[bool, str]:
        if not Path(params.metrics_path).is_file():
            return False, f"Metrics file not found: {params.metrics_path}"
        return True, ""

    def execute(self, params: ReadMetricsInput) -> ReadMetricsOutput:
        content = Path(params.metrics_path).read_text(encoding="utf-8")
        data = json.loads(content)

        summary = data.get("summary", {})
        if not summary and "epochs" in data:
            epochs = data["epochs"]
            if epochs:
                last = epochs[-1]
                summary = {k: v for k, v in last.items() if isinstance(v, (int, float))}

        if params.metric_keys:
            filtered = {}
            for key in params.metric_keys:
                if key in summary:
                    filtered[key] = summary[key]
                for epoch in data.get("epochs", []):
                    if key in epoch:
                        filtered.setdefault(f"{key}_history", []).append(epoch[key])
            return ReadMetricsOutput(metrics=data, summary=filtered)

        return ReadMetricsOutput(metrics=data, summary=summary)

    def post_validate(self, result: ReadMetricsOutput) -> tuple[bool, str]:
        if not result.metrics:
            return False, "Metrics file was empty or unparseable"
        return True, ""


class ReadLogsTool(BaseTool):
    meta = ToolMeta(
        name="read_logs",
        description="读取训练日志文件尾部，支持 grep 过滤",
        input_schema=ReadLogsInput,
        output_schema=ReadLogsOutput,
        risk_level=RiskLevel.LOW,
        timeout_sec=10,
        is_idempotent=True,
        preconditions=["log_path must exist"],
    )

    ERROR_PATTERNS = [
        r"(?i)out of memory",
        r"(?i)cuda.*error",
        r"(?i)runtime\s*error",
        r"(?i)nan",
        r"(?i)traceback",
        r"(?i)killed",
    ]

    def check_preconditions(self, params: ReadLogsInput) -> tuple[bool, str]:
        if not Path(params.log_path).is_file():
            return False, f"Log file not found: {params.log_path}"
        return True, ""

    def execute(self, params: ReadLogsInput) -> ReadLogsOutput:
        all_lines = Path(params.log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(all_lines)
        tail = all_lines[-params.tail_lines:] if total > params.tail_lines else all_lines

        matched = []
        if params.grep_pattern:
            pattern = re.compile(params.grep_pattern, re.IGNORECASE)
            matched = [line for line in all_lines if pattern.search(line)]

        detected_errors = []
        for line in all_lines:
            for pat in self.ERROR_PATTERNS:
                if re.search(pat, line):
                    detected_errors.append(line.strip())
                    break

        return ReadLogsOutput(
            lines=tail,
            total_lines=total,
            matched_lines=matched[:50],
            detected_errors=detected_errors[:20],
        )


class CompareRunsTool(BaseTool):
    meta = ToolMeta(
        name="compare_runs",
        description="对比多次实验的指标结果",
        input_schema=CompareRunsInput,
        output_schema=CompareRunsOutput,
        risk_level=RiskLevel.LOW,
        timeout_sec=10,
        is_idempotent=True,
    )

    def __init__(self, trial_store: dict[str, dict] | None = None):
        self._trial_store = trial_store or {}

    def set_trial_store(self, store: dict[str, dict]):
        self._trial_store = store

    def execute(self, params: CompareRunsInput) -> CompareRunsOutput:
        runs = []
        for tid in params.trial_ids:
            info = self._trial_store.get(tid, {})
            metric_val = info.get(params.metric_name)
            runs.append(
                RunSummary(
                    trial_id=tid,
                    metric_value=metric_val,
                    config_patch=info.get("config_patch", {}),
                    status=info.get("status", "unknown"),
                )
            )

        valid_runs = [r for r in runs if r.metric_value is not None]
        if valid_runs:
            best = max(valid_runs, key=lambda r: r.metric_value)
            ranking = sorted(valid_runs, key=lambda r: r.metric_value, reverse=True)
            ranking_ids = [r.trial_id for r in ranking]
            summary = f"Best: {best.trial_id} ({params.metric_name}={best.metric_value})"
        else:
            best = None
            ranking_ids = []
            summary = "No valid metric values to compare"

        return CompareRunsOutput(
            runs=runs,
            best_trial_id=best.trial_id if best else None,
            ranking=ranking_ids,
            summary=summary,
        )