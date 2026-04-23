"""公共类型定义"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskPhase(str, Enum):
    INIT = "init"
    ANALYZING_REPO = "analyzing_repo"
    PLANNING = "planning"
    EXECUTING_TRIAL = "executing_trial"
    WAITING_FOR_RESULT = "waiting_for_result"
    ASSESSING_RESULT = "assessing_result"
    DECIDING_NEXT = "deciding_next"
    FINISHED = "finished"
    FAILED = "failed"
    PAUSED = "paused"


class TrialStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    OOM = "oom"
    CANCELLED = "cancelled"


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SKILL = "skill"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SUSPICIOUS = "suspicious"
    REJECTED = "rejected"


class FrameworkType(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    OTHER = "other"


class ParamType(str, Enum):
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STR = "str"
    CHOICE = "choice"


class Timestamp(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)