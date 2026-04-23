"""Tool 基类 — 定义工具元信息、前置校验、后置校验的统一接口"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from schemas.common import RiskLevel


@dataclass
class PostValidation:
    check: str
    on_fail: str = "mark_suspicious"


@dataclass
class ToolMeta:
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    risk_level: RiskLevel = RiskLevel.LOW
    timeout_sec: int = 300
    is_idempotent: bool = True
    preconditions: list[str] = field(default_factory=list)
    rollback_hint: str = ""
    failure_types: list[str] = field(default_factory=list)
    approval_required: bool = False
    side_effects: list[str] = field(default_factory=list)
    post_validations: list[PostValidation] = field(default_factory=list)
    agent_visibility: list[str] = field(default_factory=lambda: ["*"])


class BaseTool(ABC):
    meta: ToolMeta

    @abstractmethod
    def execute(self, params: BaseModel) -> BaseModel:
        ...

    def check_preconditions(self, params: BaseModel) -> tuple[bool, str]:
        return True, ""

    def post_validate(self, result: BaseModel) -> tuple[bool, str]:
        return True, ""