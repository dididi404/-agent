"""observability/report.py — 生成可读的 Markdown 报告"""

from datetime import datetime
from typing import Any

from schemas.state import RunState
from schemas.task import TaskSpec


class ReportGenerator:
    def generate(
        self,
        task: TaskSpec,
        run_state: RunState,
        messages: list[dict[str, Any]],
        tool_records: list,
        trace_path: str | None = None,
    ) -> str:
        lines = [
            f"# ML Optimization Report",
            f"",
            f"**Task ID:** {task.task_id}",
            f"**Goal:** {task.goal}",
            f"**Repo:** {task.repo_path}",
            f"**Status:** {run_state.phase.value}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"---",
            f"",
            f"## Results Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Trials completed | {run_state.trials_completed} |",
            f"| Trials succeeded | {run_state.trials_succeeded} |",
            f"| Budget used | {run_state.trials_completed}/{task.budget.max_trials} |",
        ]

        if run_state.current_best:
            lines.append(f"| Best {run_state.current_best.metric_name} | **{run_state.current_best.metric_value}** |")
            lines.append(f"| Best trial | {run_state.current_best.trial_id} |")

        if run_state.total_tokens_used > 0:
            lines.append(f"| LLM tokens used | {run_state.total_tokens_used:,} |")

        lines.extend(["", "---", "", "## Trial History", ""])

        if run_state.attempted_trials:
            lines.append("| Trial | Status | Metric | Config |")
            lines.append("|-------|--------|--------|--------|")
            for trial in run_state.attempted_trials:
                metric_str = f"{trial.metric_value}" if trial.metric_value is not None else "N/A"
                patch_str = ", ".join(f"{k}={v}" for k, v in trial.config_patch.items()) if trial.config_patch else "-"
                lines.append(f"| {trial.trial_id} | {trial.status.value} | {metric_str} | {patch_str} |")
        else:
            lines.append("No trials recorded.")

        lines.extend(["", "---", "", "## Decision Trace", ""])
        for msg in messages:
            role = msg.get("role", "system")
            content = msg.get("content", "")
            lines.append(f"**[{role}]** {content}")
            lines.append("")

        lines.extend(["", "---", "", "## Tool Call Audit", ""])
        if tool_records:
            lines.append("| Tool | Status | Agent | Latency | Intent |")
            lines.append("|------|--------|-------|---------|--------|")
            for rec in tool_records:
                lines.append(
                    f"| {rec.tool_name} | {rec.output_status.value} | {rec.caller_agent} | "
                    f"{rec.latency_ms}ms | {rec.intent} |"
                )

            total_calls = len(tool_records)
            success_calls = sum(1 for r in tool_records if r.output_status.value == "success")
            lines.extend([
                "",
                f"**Total calls:** {total_calls}, **Success rate:** {success_calls}/{total_calls} "
                f"({100*success_calls/total_calls:.0f}%)" if total_calls > 0 else "",
            ])

        if trace_path:
            lines.extend(["", "---", "", f"Full trace: `{trace_path}`"])

        return "\n".join(lines)
