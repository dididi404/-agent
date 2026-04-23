"""llm/output_parser.py — 将 LLM 原始输出解析为 Pydantic 结构化对象"""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ParseError(Exception):
    def __init__(self, message: str, raw_content: str, attempts: int):
        self.raw_content = raw_content
        self.attempts = attempts
        super().__init__(message)


def extract_json(text: str) -> str:
    patterns = [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
        r"(\{.*\})",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text.strip()


def parse_llm_output(raw: str, schema: type[T]) -> T:
    json_str = extract_json(raw)
    data = json.loads(json_str)
    return schema.model_validate(data)


def build_repair_prompt(raw: str, error: str, schema: type[T]) -> str:
    schema_json = json.dumps(schema.model_json_schema(), indent=2, ensure_ascii=False)
    return (
        f"Your previous output could not be parsed. Error:\n{error}\n\n"
        f"Your raw output was:\n{raw[:1000]}\n\n"
        f"Please fix it. Return ONLY valid JSON matching this schema:\n{schema_json}"
    )
