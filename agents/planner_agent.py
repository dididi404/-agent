"""agents/planner_agent.py — 实验规划 Agent

基于当前状态和记忆，生成下一轮实验计划。
"""

import json

from schemas.plan import ExperimentPlan
from schemas.repo import RepoProfile
from schemas.state import RunState
from schemas.task import TaskSpec
from agents.base import BaseAgent
from llm.adapter import LLMAdapter
from llm.token_budget import TokenBudget
from tools.chains import ALL_CHAINS


class PlannerAgent(BaseAgent):
    name = "planner_agent"
    output_schema = ExperimentPlan

    def __init__(
        self,
        llm: LLMAdapter,
        token_budget: TokenBudget | None = None,
    ):
        super().__init__(llm, token_budget)

    def plan(
        self,
        task: TaskSpec,
        profile: RepoProfile,
        run_state: RunState,
        memory_context: list[dict] | None = None,
    ) -> tuple[ExperimentPlan, dict]:

        chains_info = {
            name: {"description": c.description, "steps": [s.tool_intent for s in c.steps]}
            for name, c in ALL_CHAINS.items()
        }

        trials_summary = []
        for t in run_state.attempted_trials:
            trials_summary.append({
                "trial_id": t.trial_id,
                "status": t.status.value,
                "metric_value": t.metric_value,
                "config_patch": t.config_patch,
                "error": t.error_message,
            })

        system_prompt = self.load_prompt("planner.md")
        user_prompt = self.format_context(
            task_spec=task,
            repo_profile=profile,
            run_state_summary={
                "phase": run_state.phase.value,
                "current_best": run_state.current_best.model_dump() if run_state.current_best else None,
                "trials_completed": run_state.trials_completed,
                "budget_remaining": task.budget.max_trials - run_state.trials_completed,
                "last_error": run_state.last_error,
            },
            attempted_trials=trials_summary,
            available_tool_chains=chains_info,
            memory_context=memory_context or [],
        )

        plan, call_result = self.call_llm_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=ExperimentPlan,
        )

        return plan, {
            "tokens_used": call_result.total_tokens,
            "attempts": call_result.attempts,
        }
