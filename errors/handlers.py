"""errors/handlers.py — 统一错误处理入口，将恢复动作转成具体 config patch"""

from typing import Any

from errors.taxonomy import ErrorCategory, classify_error
from errors.recovery import RecoveryAction, get_next_action, get_recovery_strategy


class ErrorHandler:
    def __init__(self):
        self._retry_counts: dict[str, int] = {}

    def handle(
        self,
        task_id: str,
        error_message: str,
        current_config: dict[str, Any] | None = None,
    ) -> dict:
        category = classify_error(error_message)
        key = f"{task_id}:{category.value}"
        retry_count = self._retry_counts.get(key, 0)

        action = get_next_action(category, retry_count)
        strategy = get_recovery_strategy(category)

        config_patch = self._action_to_patch(action, current_config or {})
        should_continue = action not in (RecoveryAction.ABORT, RecoveryAction.PAUSE_FOR_HUMAN)

        self._retry_counts[key] = retry_count + 1

        return {
            "category": category.value,
            "action": action.value,
            "description": strategy.description,
            "config_patch": config_patch,
            "should_continue": should_continue,
            "retry_count": retry_count + 1,
            "max_retries": strategy.max_retries,
        }

    def reset(self, task_id: str) -> None:
        keys_to_remove = [k for k in self._retry_counts if k.startswith(f"{task_id}:")]
        for k in keys_to_remove:
            del self._retry_counts[k]

    def _action_to_patch(self, action: RecoveryAction, config: dict) -> dict:
        if action == RecoveryAction.REDUCE_BATCH_SIZE:
            current_bs = config.get("training.batch_size", config.get("batch_size", 64))
            new_bs = max(8, int(current_bs) // 2)
            return {"training.batch_size": new_bs}

        if action == RecoveryAction.ENABLE_GRAD_ACCUM:
            return {"training.gradient_accumulation_steps": 4}

        if action == RecoveryAction.INCREASE_TIMEOUT:
            return {"_timeout_minutes": 240}

        if action == RecoveryAction.ROLLBACK_CONFIG:
            return {"_rollback": True}

        return {}
