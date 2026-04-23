"""errors/recovery.py — 按错误类型的恢复策略映射"""

from dataclasses import dataclass
from enum import Enum

from errors.taxonomy import ErrorCategory


class RecoveryAction(str, Enum):
    RETRY_SAME = "retry_same"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    ENABLE_GRAD_ACCUM = "enable_grad_accum"
    INCREASE_TIMEOUT = "increase_timeout"
    ROLLBACK_CONFIG = "rollback_config"
    SKIP_TRIAL = "skip_trial"
    PAUSE_FOR_HUMAN = "pause_for_human"
    RETRY_WITH_LOWER_TEMP = "retry_with_lower_temp"
    ABORT = "abort"


@dataclass
class RecoveryStrategy:
    actions: list[RecoveryAction]
    max_retries: int
    description: str


RECOVERY_TABLE: dict[ErrorCategory, RecoveryStrategy] = {
    ErrorCategory.TOOL_ERROR: RecoveryStrategy(
        actions=[RecoveryAction.RETRY_SAME, RecoveryAction.ROLLBACK_CONFIG, RecoveryAction.SKIP_TRIAL],
        max_retries=2,
        description="Tool execution failed. Retry, then rollback config, then skip.",
    ),
    ErrorCategory.RESOURCE_ERROR: RecoveryStrategy(
        actions=[RecoveryAction.REDUCE_BATCH_SIZE, RecoveryAction.ENABLE_GRAD_ACCUM, RecoveryAction.SKIP_TRIAL],
        max_retries=2,
        description="OOM or resource issue. Reduce batch_size or enable grad accumulation.",
    ),
    ErrorCategory.TIMEOUT_ERROR: RecoveryStrategy(
        actions=[RecoveryAction.INCREASE_TIMEOUT, RecoveryAction.REDUCE_BATCH_SIZE, RecoveryAction.SKIP_TRIAL],
        max_retries=1,
        description="Training timed out. Increase timeout or simplify config.",
    ),
    ErrorCategory.LLM_ERROR: RecoveryStrategy(
        actions=[RecoveryAction.RETRY_WITH_LOWER_TEMP, RecoveryAction.RETRY_SAME, RecoveryAction.ABORT],
        max_retries=3,
        description="LLM output parsing failed. Retry with lower temperature.",
    ),
    ErrorCategory.EXPERIMENT_ERROR: RecoveryStrategy(
        actions=[RecoveryAction.ROLLBACK_CONFIG, RecoveryAction.SKIP_TRIAL],
        max_retries=1,
        description="Training NaN/divergence. Rollback config and try different params.",
    ),
    ErrorCategory.POLICY_ERROR: RecoveryStrategy(
        actions=[RecoveryAction.PAUSE_FOR_HUMAN, RecoveryAction.ABORT],
        max_retries=0,
        description="Policy violation. Requires human intervention.",
    ),
}


def get_recovery_strategy(category: ErrorCategory) -> RecoveryStrategy:
    return RECOVERY_TABLE.get(
        category,
        RecoveryStrategy(
            actions=[RecoveryAction.RETRY_SAME, RecoveryAction.ABORT],
            max_retries=1,
            description="Unknown error. Retry once then abort.",
        ),
    )


def get_next_action(category: ErrorCategory, retry_count: int) -> RecoveryAction:
    strategy = get_recovery_strategy(category)
    if retry_count >= strategy.max_retries:
        return strategy.actions[-1]
    idx = min(retry_count, len(strategy.actions) - 1)
    return strategy.actions[idx]
