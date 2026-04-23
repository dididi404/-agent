"""memory/retriever.py — 记忆检索 + 排序 + 截断"""

from datetime import datetime

from schemas.common import MemoryType
from schemas.state import MemoryRecord
from memory.store import MemoryStore


class MemoryRetriever:
    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.1,
        top_k: int = 10,
        context_match: dict | None = None,
    ) -> list[MemoryRecord]:
        candidates = self.store.query(
            memory_type=memory_type,
            tags=tags,
            min_confidence=min_confidence,
            limit=top_k * 3,
        )

        now = datetime.now()
        scored = []
        for mem in candidates:
            ctx_score = self._compute_context_match(mem, context_match) if context_match else 1.0
            relevance = mem.compute_relevance(now, ctx_score)
            scored.append((relevance, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def retrieve_for_agent(
        self,
        agent_name: str,
        current_tags: list[str] | None = None,
        context_match: dict | None = None,
        top_k: int = 5,
    ) -> list[MemoryRecord]:
        agent_memory_config = {
            "planner_agent": {
                "types": [MemoryType.EPISODIC, MemoryType.SKILL],
                "min_confidence": 0.3,
            },
            "executor_agent": {
                "types": [MemoryType.SKILL],
                "min_confidence": 0.5,
            },
            "critic_agent": {
                "types": [MemoryType.EPISODIC],
                "min_confidence": 0.2,
            },
        }

        config = agent_memory_config.get(agent_name, {
            "types": [MemoryType.EPISODIC, MemoryType.SKILL],
            "min_confidence": 0.3,
        })

        all_results = []
        for mem_type in config["types"]:
            results = self.retrieve(
                memory_type=mem_type,
                tags=current_tags,
                min_confidence=config["min_confidence"],
                top_k=top_k,
                context_match=context_match,
            )
            all_results.extend(results)

        now = datetime.now()
        all_results.sort(
            key=lambda m: m.compute_relevance(now, 1.0),
            reverse=True,
        )
        return all_results[:top_k]

    def _compute_context_match(self, mem: MemoryRecord, query_context: dict) -> float:
        if not query_context or not mem.context:
            return 0.5

        matches = 0
        total = 0
        for key, val in query_context.items():
            if key in mem.context:
                total += 1
                if mem.context[key] == val:
                    matches += 1
                elif isinstance(val, str) and isinstance(mem.context[key], str):
                    if val.lower() in mem.context[key].lower() or mem.context[key].lower() in val.lower():
                        matches += 0.5

        if total == 0:
            return 0.5
        return matches / total
