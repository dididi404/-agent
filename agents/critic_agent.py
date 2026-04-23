"""agents/critic_agent.py — Critic / Evaluator Agent

评估实验结果有效性，控制记忆写入质量。
支持 LLM 模式和硬编码降级模式。
"""

from typing import Any

from schemas.state import RunState
from schemas.task import TaskSpec
from schemas.trial import TrialAssessment, MemoryCandidate
from schemas.common import MemoryType
from agents.base import BaseAgent
from llm.adapter import LLMAdapter
from llm.token_budget import TokenBudget


class CriticAgent(BaseAgent):
    name = "critic_agent"
    output_schema = TrialAssessment

    def __init__(
        self,
        llm: LLMAdapter | None = None,
        token_budget: TokenBudget | None = None,
        use_llm: bool = False,
    ):
        if llm:
            super().__init__(llm, token_budget)
        else:
            self.llm = None
            self.token_budget = token_budget or TokenBudget()
        self.use_llm = use_llm and llm is not None

    def assess(
        self,
        task: TaskSpec,
        run_state: RunState,
        metrics_data: dict[str, Any],
        config_patch: dict[str, Any] | None = None,
        memory_context: str = "",
    ) -> tuple[TrialAssessment, dict]:
        if self.use_llm:
            return self._assess_llm(task, run_state, metrics_data, config_patch, memory_context)
        return self._assess_hardcoded(task, run_state, metrics_data, config_patch)

    def _assess_hardcoded(
        self,
        task: TaskSpec,
        run_state: RunState,
        metrics_data: dict,
        config_patch: dict | None,
    ) -> tuple[TrialAssessment, dict]:
        summary = metrics_data.get("summary", {})
        primary = task.success_criteria.primary_metric
        metric_value = summary.get(f"best_{primary}", summary.get(primary))

        if metric_value is None:
            for key, val in summary.items():
                if primary.replace("_", "") in key.replace("_", "") and isinstance(val, (int, float)):
                    metric_value = val
                    break

        trial_id = run_state.active_plan_id or "unknown"
        is_valid = metric_value is not None

        is_better = False
        if is_valid and run_state.current_best is not None:
            is_better = task.is_better(metric_value, run_state.current_best.metric_value)
        elif is_valid and run_state.current_best is None:
            is_better = True

        confidence = 0.0
        if is_valid:
            confidence = 0.8
            if is_better and run_state.current_best:
                delta = abs(metric_value - run_state.current_best.metric_value)
                if delta > 0.05:
                    confidence = 0.9
                elif delta < 0.005:
                    confidence = 0.5

        diagnosis = f"{primary}={metric_value}"
        if is_better:
            diagnosis += " (new best)"
        elif not is_valid:
            diagnosis = f"Could not extract {primary} from metrics"

        memory_candidate = None
        if is_valid and confidence >= 0.6 and config_patch:
            patch_desc = ", ".join(f"{k}={v}" for k, v in config_patch.items())
            if is_better:
                memory_candidate = MemoryCandidate(
                    type=MemoryType.SKILL,
                    summary=f"Setting {patch_desc} improved {primary} to {metric_value}",
                    context={"metric": primary, "config_patch": config_patch},
                    confidence=confidence,
                )
            else:
                memory_candidate = MemoryCandidate(
                    type=MemoryType.EPISODIC,
                    summary=f"Setting {patch_desc} yielded {primary}={metric_value} (no improvement)",
                    context={"metric": primary, "config_patch": config_patch},
                    confidence=confidence * 0.7,
                )

        assessment = TrialAssessment(
            trial_id=trial_id,
            is_valid=is_valid,
            metric_value=metric_value,
            metric_name=primary,
            better_than_best=is_better,
            confidence=confidence,
            diagnosis=diagnosis,
            memory_candidate=memory_candidate,
        )
        return assessment, {"mode": "hardcoded"}

    def _assess_llm(
        self,
        task: TaskSpec,
        run_state: RunState,
        metrics_data: dict,
        config_patch: dict | None,
        memory_context: str,
    ) -> tuple[TrialAssessment, dict]:
        try:
            system_prompt = self.load_prompt("critic.md")
            user_prompt = self.format_context(
                task_goal=task.goal,
                primary_metric=task.success_criteria.primary_metric,
                current_best=run_state.current_best.model_dump() if run_state.current_best else None,
                config_patch=config_patch,
                metrics_summary=metrics_data.get("summary", {}),
                epoch_history=metrics_data.get("epochs", [])[-5:],
                memory=memory_context,
            )
            assessment, call_result = self.call_llm_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=TrialAssessment,
            )
            return assessment, {"mode": "llm", "tokens_used": call_result.total_tokens}
        except Exception as e:
            result, meta = self._assess_hardcoded(task, run_state, metrics_data, config_patch)
            meta["llm_error"] = str(e)
            meta["mode"] = "hardcoded_fallback"
            return result, meta
