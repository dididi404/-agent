"""TaskSpec: 用户输入的任务定义"""

from pydantic import BaseModel, Field


class BudgetSpec(BaseModel):
    max_trials: int = Field(default=10, ge=1, le=100)
    max_hours: float = Field(default=6.0, gt=0)
    max_concurrent_runs: int = Field(default=1, ge=1)


class ConstraintSpec(BaseModel):
    allow_code_edit: bool = False
    allow_config_edit: bool = True
    gpu_ids: list[int] = Field(default_factory=lambda: [0])
    allowed_param_keys: list[str] | None = None
    blocked_commands: list[str] = Field(default_factory=list)


class SuccessCriteria(BaseModel):
    primary_metric: str = "val_accuracy"
    direction: str = Field(default="maximize", pattern="^(maximize|minimize)$")
    improvement_threshold: float = Field(default=0.01, ge=0)


class TaskSpec(BaseModel):
    task_id: str
    repo_path: str
    goal: str
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    constraints: ConstraintSpec = Field(default_factory=ConstraintSpec)
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)

    def is_better(self, new_value: float, old_value: float) -> bool:
        if self.success_criteria.direction == "maximize":
            return new_value > old_value + self.success_criteria.improvement_threshold
        return new_value < old_value - self.success_criteria.improvement_threshold