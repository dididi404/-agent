"""llm/token_budget.py — 单任务 token 预算管理"""


class TokenBudget:
    def __init__(self, max_tokens: int = 200_000):
        self.max_tokens = max_tokens
        self.used_tokens = 0

    def consume(self, tokens: int) -> None:
        self.used_tokens += tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def exhausted(self) -> bool:
        return self.used_tokens >= self.max_tokens

    def check(self, estimated_tokens: int = 5000) -> tuple[bool, str]:
        if self.remaining < estimated_tokens:
            return False, f"Token budget nearly exhausted: {self.used_tokens}/{self.max_tokens}"
        return True, ""
