"""exec_tools — launch_train 工具实现"""

import os
import subprocess
import time
from pathlib import Path

from schemas.common import RiskLevel
from tools.base import BaseTool, PostValidation, ToolMeta
from tools.schemas import LaunchTrainInput, LaunchTrainOutput


class LaunchTrainTool(BaseTool):
    meta = ToolMeta(
        name="launch_train",
        description="启动训练进程并等待完成，收集输出",
        input_schema=LaunchTrainInput,
        output_schema=LaunchTrainOutput,
        risk_level=RiskLevel.HIGH,
        timeout_sec=7200,
        is_idempotent=False,
        preconditions=["working_dir must exist", "cmd must not be empty"],
        rollback_hint="kill process, restore config from .bak",
        failure_types=["timeout", "oom", "nan_loss", "non_zero_exit", "missing_output"],
        side_effects=["starts subprocess", "writes output files", "uses GPU"],
        post_validations=[
            PostValidation(
                check="return_code == 0",
                on_fail="mark_failed",
            ),
            PostValidation(
                check="runtime_sec > 10",
                on_fail="mark_suspicious",
            ),
        ],
        agent_visibility=["executor_agent"],
    )

    def check_preconditions(self, params: LaunchTrainInput) -> tuple[bool, str]:
        if not Path(params.working_dir).is_dir():
            return False, f"Working directory not found: {params.working_dir}"
        if not params.cmd.strip():
            return False, "Command is empty"
        return True, ""

    def execute(self, params: LaunchTrainInput) -> LaunchTrainOutput:
        env = os.environ.copy()
        env.update(params.env_vars)

        timeout_sec = params.timeout_minutes * 60
        start = time.time()

        try:
            proc = subprocess.run(
                params.cmd,
                shell=True,
                cwd=params.working_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return LaunchTrainOutput(
                return_code=-1,
                stderr_tail="TIMEOUT: training exceeded time limit",
                runtime_sec=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            return LaunchTrainOutput(
                return_code=-1,
                stderr_tail=str(e)[:2000],
                runtime_sec=elapsed,
            )

        elapsed = time.time() - start

        log_path = None
        metrics_path = None
        working = Path(params.working_dir)
        for candidate in ["outputs/train.log", "train.log", "logs/train.log"]:
            if (working / candidate).exists():
                log_path = str(working / candidate)
                break
        for candidate in ["outputs/metrics.json", "metrics.json", "outputs/results.json"]:
            if (working / candidate).exists():
                metrics_path = str(working / candidate)
                break

        stdout_lines = proc.stdout.splitlines() if proc.stdout else []
        stderr_lines = proc.stderr.splitlines() if proc.stderr else []

        return LaunchTrainOutput(
            pid=None,
            return_code=proc.returncode,
            log_path=log_path,
            metrics_path=metrics_path,
            stdout_tail="\n".join(stdout_lines[-50:]),
            stderr_tail="\n".join(stderr_lines[-50:]),
            runtime_sec=elapsed,
        )

    def post_validate(self, result: LaunchTrainOutput) -> tuple[bool, str]:
        if result.return_code != 0:
            stderr = result.stderr_tail.lower()
            if "out of memory" in stderr or "oom" in stderr:
                return False, "OOM detected"
            if "nan" in stderr:
                return False, "NaN detected in training"
            return False, f"Non-zero exit code: {result.return_code}"
        if result.runtime_sec < 10:
            return False, "Suspiciously short runtime (<10s), may not have trained"
        return True, ""