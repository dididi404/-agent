"""llm/retry_policy.py — LLM 输出解析失败时的重试策略"""

from typing import TypeVar

from pydantic import BaseModel

from llm.adapter import LLMAdapter, LLMResponse
from llm.output_parser import ParseError, build_repair_prompt, parse_llm_output

T = TypeVar("T", bound=BaseModel)


class LLMCallResult(BaseModel):
    parsed: dict | None = None
    raw: str = ""
    total_tokens: int = 0
    attempts: int = 0
    success: bool = False
    error: str = ""


def call_llm_with_retry(
    adapter: LLMAdapter,
    system_prompt: str,
    user_prompt: str,
    schema: type[T],
    max_retries: int = 3,
    use_json_mode: bool = True,
) -> tuple[T, LLMCallResult]:
    total_tokens = 0
    last_error = ""
    last_raw = ""

    for attempt in range(1, max_retries + 1):
        if use_json_mode:
            resp = adapter.chat_json(system_prompt, user_prompt)
        else:
            resp = adapter.chat(system_prompt, user_prompt)

        total_tokens += resp.tokens_used
        last_raw = resp.content

        try:
            parsed = parse_llm_output(resp.content, schema)
            result = LLMCallResult(
                parsed=parsed.model_dump(),
                raw=resp.content,
                total_tokens=total_tokens,
                attempts=attempt,
                success=True,
            )
            return parsed, result
        except Exception as e:
            last_error = str(e)
            repair = build_repair_prompt(resp.content, last_error, schema)
            user_prompt = repair

    raise ParseError(
        f"Failed to parse LLM output after {max_retries} attempts: {last_error}",
        raw_content=last_raw,
        attempts=max_retries,
    )
