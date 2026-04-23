"""RunState: Supervisor 持有的全局状态 + MemoryRecord"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .common import MemoryType, TaskPhase, TrialStatus


class TrialRecord(BaseModel):
    trial_id: str
    plan_id: str
    status: TrialStatus = TrialStatus.PENDING
    config_patch: dict[str, Any] = Field(default_factory=dict)
    metric_value: float | None = None
    metric_name: str = ""
    error_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ActiveRunHandle(BaseModel):
    pid: int | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    expected_duration_min: float = 60.0
    last_checked_at: datetime | None = None
    intermediate_metrics: list[dict[str, Any]] = Field(default_factory=list)


class BestResult(BaseModel):
    trial_id: str
    metric_value: float
    metric_name: str = ""


class RunState(BaseModel):
    task_id: str
    phase: TaskPhase = TaskPhase.INIT
    current_best: BestResult | None = None
    attempted_trials: list[TrialRecord] = Field(default_factory=list)
    active_plan_id: str | None = None
    pending_steps: list[str] = Field(default_factory=list)
    active_run_handle: ActiveRunHandle | None = None
    last_error: str | None = None
    last_error_category: str | None = None
    retry_count: int = 0
    human_approval_required: bool = False
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0

    @property
    def trials_completed(self) -> int:
        return sum(
            1
            for t in self.attempted_trials
            if t.status in (TrialStatus.SUCCESS, TrialStatus.FAILED)
        )

    @property
    def trials_succeeded(self) -> int:
        return sum(
            1 for t in self.attempted_trials if t.status == TrialStatus.SUCCESS
        )


class MemoryRecord(BaseModel):
    id: str
    type: MemoryType
    content: str
    context: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_validated: datetime = Field(default_factory=datetime.now)
    validation_count: int = 0
    source_trial_id: str | None = None

    def compute_relevance(self, current_time: datetime, context_match: float) -> float:
        age_days = (current_time - self.last_validated).days
        decay = 0.95**age_days
        return self.confidence * decay * context_match