"""llm/adapter.py — Provider-agnostic LLM 接口

上层 Agent 只调 LLMAdapter 的方法，不直接接触 SDK。
"""

import json
import os
from typing import Any

from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    model: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0


class LLMAdapter:
    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
    ):
        self.provider = provider
        self.temperature = temperature
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI
                kwargs: dict[str, Any] = {"api_key": self._api_key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = OpenAI(**kwargs)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=temperature if temperature is not None else self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        tokens = usage.total_tokens if usage else 0
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            tokens_used=tokens,
        )

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=temperature if temperature is not None else self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0]
        usage = response.usage
        tokens = usage.total_tokens if usage else 0
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            tokens_used=tokens,
        )
