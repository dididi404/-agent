"""Tool I/O Schemas — 每个工具的具体输入输出定义"""

from typing import Any

from pydantic import BaseModel, Field


# ---- inspect_repo ----
class InspectRepoInput(BaseModel):
    repo_path: str


class InspectRepoOutput(BaseModel):
    files: list[str]
    directories: list[str]
    has_requirements: bool = False
    has_setup_py: bool = False
    has_pyproject: bool = False
    config_candidates: list[str] = Field(default_factory=list)
    script_candidates: list[str] = Field(default_factory=list)


# ---- read_file ----
class ReadFileInput(BaseModel):
    file_path: str
    max_lines: int = 500


class ReadFileOutput(BaseModel):
    content: str
    line_count: int
    truncated: bool = False


# ---- read_config ----
class ReadConfigInput(BaseModel):
    config_path: str


class ReadConfigOutput(BaseModel):
    data: dict[str, Any]
    format: str = "yaml"


# ---- patch_config ----
class PatchConfigInput(BaseModel):
    config_path: str
    patch: dict[str, Any]
    backup: bool = True


class PatchConfigOutput(BaseModel):
    original_values: dict[str, Any]
    new_values: dict[str, Any]
    backup_path: str | None = None


# ---- launch_train ----
class LaunchTrainInput(BaseModel):
    cmd: str
    working_dir: str
    timeout_minutes: int = 120
    env_vars: dict[str, str] = Field(default_factory=dict)


class LaunchTrainOutput(BaseModel):
    pid: int | None = None
    return_code: int | None = None
    log_path: str | None = None
    metrics_path: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    runtime_sec: float = 0.0


# ---- read_metrics ----
class ReadMetricsInput(BaseModel):
    metrics_path: str
    metric_keys: list[str] = Field(default_factory=list)


class ReadMetricsOutput(BaseModel):
    metrics: dict[str, Any]
    summary: dict[str, Any] = Field(default_factory=dict)


# ---- read_logs ----
class ReadLogsInput(BaseModel):
    log_path: str
    tail_lines: int = 100
    grep_pattern: str | None = None


class ReadLogsOutput(BaseModel):
    lines: list[str]
    total_lines: int
    matched_lines: list[str] = Field(default_factory=list)
    detected_errors: list[str] = Field(default_factory=list)


# ---- compare_runs ----
class CompareRunsInput(BaseModel):
    trial_ids: list[str] = Field(min_length=2)
    metric_name: str


class RunSummary(BaseModel):
    trial_id: str
    metric_value: float | None = None
    config_patch: dict[str, Any] = Field(default_factory=dict)
    status: str = ""


class CompareRunsOutput(BaseModel):
    runs: list[RunSummary]
    best_trial_id: str | None = None
    ranking: list[str] = Field(default_factory=list)
    summary: str = ""