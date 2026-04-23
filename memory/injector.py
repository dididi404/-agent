"""memory/injector.py — 按 agent/phase 将记忆注入 prompt"""

import json

from schemas.state import MemoryRecord
from memory.retriever import MemoryRetriever


class MemoryInjector:
    def __init__(self, retriever: MemoryRetriever):
        self.retriever = retriever

    def inject_for_agent(
        self,
        agent_name: str,
        current_tags: list[str] | None = None,
        context_match: dict | None = None,
        max_tokens: int = 2000,
        top_k: int = 5,
    ) -> str:
        memories = self.retriever.retrieve_for_agent(
            agent_name=agent_name,
            current_tags=current_tags,
            context_match=context_match,
            top_k=top_k,
        )

        if not memories:
            return ""

        parts = ["<memory_context>"]
        total_chars = 0
        char_limit = max_tokens * 4

        for mem in memories:
            entry = self._format_memory(mem)
            if total_chars + len(entry) > char_limit:
                break
            parts.append(entry)
            total_chars += len(entry)

        parts.append("</memory_context>")
        return "\n".join(parts)

    def _format_memory(self, mem: MemoryRecord) -> str:
        ctx_str = ""
        if mem.context:
            ctx_parts = [f"{k}={v}" for k, v in mem.context.items()]
            ctx_str = f" [{', '.join(ctx_parts)}]"

        confidence_label = "high" if mem.confidence > 0.7 else "medium" if mem.confidence > 0.4 else "low"

        return (
            f"- [{mem.type.value}] (confidence={confidence_label}, validated={mem.validation_count}x){ctx_str}\n"
            f"  {mem.content}"
        )
