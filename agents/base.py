"""agents/base.py — Agent 基类，统一 LLM 调用 + 结构化输出校验"""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from llm.adapter import LLMAdapter
from llm.retry_policy import LLMCallResult, call_llm_with_retry
from llm.token_budget import TokenBudget

T = TypeVar("T", bound=BaseModel)

PROMPT_DIR = Path(__file__).parent / "prompts"


class BaseAgent:
    name: str = "base_agent"
    output_schema: type[BaseModel]

    def __init__(
        self,
        llm: LLMAdapter,
        token_budget: TokenBudget | None = None,
    ):
        self.llm = llm
        self.token_budget = token_budget or TokenBudget()

    def load_prompt(self, filename: str) -> str:
        path = PROMPT_DIR / filename
        return path.read_text(encoding="utf-8")

    def call_llm_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        max_retries: int = 3,
    ) -> tuple[T, LLMCallResult]:
        ok, msg = self.token_budget.check()
        if not ok:
            raise RuntimeError(f"[{self.name}] {msg}")

        parsed, result = call_llm_with_retry(
            adapter=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            max_retries=max_retries,
        )
        self.token_budget.consume(result.total_tokens)
        return parsed, result

    def format_context(self, **kwargs: Any) -> str:
        parts = []
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, BaseModel):
                parts.append(f"<{key}>\n{value.model_dump_json(indent=2)}\n</{key}>")
            elif isinstance(value, (dict, list)):
                parts.append(f"<{key}>\n{json.dumps(value, indent=2, ensure_ascii=False)}\n</{key}>")
            else:
                parts.append(f"<{key}>\n{value}\n</{key}>")
        return "\n\n".join(parts)
