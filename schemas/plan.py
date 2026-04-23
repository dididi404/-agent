"""ExperimentPlan: Planner Agent 的输出"""

from typing import Any

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: str
    action: str
    tool_intent: str
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    optional: bool = False


class ExperimentPlan(BaseModel):
    plan_id: str
    hypothesis: str
    steps: list[PlanStep] = Field(min_length=1)
    estimated_duration_min: float | None = None
    rollback_plan: str | None = None
    source_chain: str | None = None