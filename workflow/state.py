"""workflow/state.py — LangGraph 图状态定义

这是 LangGraph StateGraph 使用的 TypedDict 状态，
与 schemas/state.py 中的 Pydantic RunState 不同：
  - RunState 是持久化业务模型
  - GraphState 是图流转的运行时载体，包含所有中间产物
"""

from typing import Any, TypedDict

from schemas.common import TaskPhase
from schemas.plan import ExperimentPlan
from schemas.repo import RepoProfile
from schemas.state import RunState
from schemas.task import TaskSpec
from schemas.trial import TrialAssessment, ToolExecutionResult


class GraphState(TypedDict, total=False):
    task_spec: TaskSpec
    run_state: RunState
    repo_profile: RepoProfile | None
    current_plan: ExperimentPlan | None
    current_step_index: int
    current_tool_result: ToolExecutionResult | None
    current_assessment: TrialAssessment | None
    error: str | None
    error_category: str | None
    decision: str  # "continue" | "next_trial" | "finish" | "recover" | "pause"
    messages: list[dict[str, Any]]
