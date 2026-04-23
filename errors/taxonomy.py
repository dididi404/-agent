"""errors/taxonomy.py — 错误分类体系"""

from enum import Enum


class ErrorCategory(str, Enum):
    TOOL_ERROR = "tool_error"
    RESOURCE_ERROR = "resource_error"
    TIMEOUT_ERROR = "timeout_error"
    LLM_ERROR = "llm_error"
    EXPERIMENT_ERROR = "experiment_error"
    POLICY_ERROR = "policy_error"


def classify_error(error_message: str) -> ErrorCategory:
    msg = error_message.lower()

    if "out of memory" in msg or "oom" in msg or "cuda out" in msg:
        return ErrorCategory.RESOURCE_ERROR
    if "disk" in msg and ("full" in msg or "space" in msg):
        return ErrorCategory.RESOURCE_ERROR
    if "gpu" in msg and ("unavailable" in msg or "not found" in msg):
        return ErrorCategory.RESOURCE_ERROR

    if "timeout" in msg or "timed out" in msg or "time limit" in msg:
        return ErrorCategory.TIMEOUT_ERROR

    if "nan" in msg or "inf" in msg or "diverge" in msg or "loss explod" in msg:
        return ErrorCategory.EXPERIMENT_ERROR

    if "parse" in msg or "json" in msg or "schema" in msg or "validation" in msg:
        if "llm" in msg or "model" in msg or "output" in msg:
            return ErrorCategory.LLM_ERROR

    if "api" in msg and ("rate" in msg or "limit" in msg or "quota" in msg):
        return ErrorCategory.LLM_ERROR

    if "budget" in msg or "approval" in msg or "blocked" in msg or "rejected" in msg:
        return ErrorCategory.POLICY_ERROR

    return ErrorCategory.TOOL_ERROR
