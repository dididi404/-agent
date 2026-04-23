"""memory/service.py — Memory Service 统一入口"""

from schemas.common import MemoryType
from schemas.state import MemoryRecord, TrialRecord
from schemas.trial import TrialAssessment
from memory.store import MemoryStore
from memory.retriever import MemoryRetriever
from memory.injector import MemoryInjector
from memory.writer import MemoryWriter


class MemoryService:
    def __init__(self, db_path: str = "memory.db"):
        self.store = MemoryStore(db_path)
        self.retriever = MemoryRetriever(self.store)
        self.injector = MemoryInjector(self.retriever)
        self.writer = MemoryWriter(self.store)

    def get_context_for_agent(
        self,
        agent_name: str,
        tags: list[str] | None = None,
        context_match: dict | None = None,
        top_k: int = 5,
    ) -> str:
        return self.injector.inject_for_agent(
            agent_name=agent_name,
            current_tags=tags,
            context_match=context_match,
            top_k=top_k,
        )

    def record_trial(
        self,
        trial: TrialRecord,
        assessment: TrialAssessment,
        config_patch: dict | None = None,
        repo_context: dict | None = None,
    ) -> dict:
        episodic = self.writer.write_episodic(trial, assessment, config_patch, repo_context)
        skill = self.writer.write_skill(assessment, repo_context)
        return {
            "episodic_written": episodic is not None,
            "skill_written": skill is not None,
            "episodic_id": episodic.id if episodic else None,
            "skill_id": skill.id if skill else None,
        }

    def get_stats(self) -> dict:
        return {
            "total": self.store.count(),
            "episodic": self.store.count(MemoryType.EPISODIC),
            "skill": self.store.count(MemoryType.SKILL),
        }

    def close(self):
        self.store.close()
