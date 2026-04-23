"""TrialAssessment + ToolExecution 相关 schema"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .common import MemoryType, RiskLevel, ToolStatus


class ToolExecutionRequest(BaseModel):
    request_id: str
    tool_name: str
    caller_agent: str = ""
    intent: str = ""
    validated: bool = False
    preconditions_passed: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    timeout_sec: int = 300
    dry_run: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    request_id: str
    tool_name: str
    status: ToolStatus
    return_code: int | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    stdout_summary: str = ""
    stderr_summary: str = ""
    runtime_sec: float = 0.0
    post_validation_passed: bool = True
    error_message: str | None = None


class ToolCallRecord(BaseModel):
    request_id: str
    tool_name: str
    caller_agent: str
    intent: str
    input_hash: str = ""
    output_status: ToolStatus = ToolStatus.SUCCESS
    latency_ms: int = 0
    was_retried: bool = False
    post_validation_passed: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)


class MemoryCandidate(BaseModel):
    type: MemoryType
    summary: str
    context: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class TrialAssessment(BaseModel):
    trial_id: str
    is_valid: bool
    metric_value: float | None = None
    metric_name: str = ""
    better_than_best: bool = False
    confidence: float = 0.0
    diagnosis: str = ""
    memory_candidate: MemoryCandidate | None = None