"""RepoProfile: 项目理解 Agent 的输出"""

from pydantic import BaseModel, Field

from .common import FrameworkType, ParamType


class SearchSpaceCandidate(BaseModel):
    name: str
    type: ParamType
    current_value: float | int | str | bool | None = None
    config_path: str = ""
    config_key: str = ""


class RepoProfile(BaseModel):
    train_entry: str
    config_files: list[str]
    metric_sources: list[str]
    framework: FrameworkType
    search_space_candidates: list[SearchSpaceCandidate] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    python_version: str | None = None
    dependencies: list[str] = Field(default_factory=list)