"""memory/writer.py — 记忆写入 + Critic 审核逻辑"""

import uuid
from datetime import datetime

from schemas.common import MemoryType
from schemas.state import MemoryRecord, TrialRecord
from schemas.trial import TrialAssessment
from memory.store import MemoryStore


class MemoryWriter:
    def __init__(self, store: MemoryStore):
        self.store = store

    def write_episodic(
        self,
        trial: TrialRecord,
        assessment: TrialAssessment,
        config_patch: dict | None = None,
        repo_context: dict | None = None,
    ) -> MemoryRecord | None:
        if not assessment.is_valid:
            return None
        if assessment.confidence < 0.3:
            return None

        context = repo_context.copy() if repo_context else {}
        context["trial_id"] = trial.trial_id
        context["metric_name"] = trial.metric_name
        if config_patch:
            context["config_patch"] = config_patch

        content = (
            f"Trial {trial.trial_id}: {trial.metric_name}={trial.metric_value} "
            f"(status={trial.status.value}). "
            f"{assessment.diagnosis}"
        )

        record = MemoryRecord(
            id=f"ep_{uuid.uuid4().hex[:8]}",
            type=MemoryType.EPISODIC,
            content=content,
            context=context,
            confidence=assessment.confidence,
            tags=self._extract_tags(trial, assessment),
            source_trial_id=trial.trial_id,
        )
        self.store.save(record)
        return record

    def write_skill(
        self,
        assessment: TrialAssessment,
        repo_context: dict | None = None,
    ) -> MemoryRecord | None:
        if not self._should_write_skill(assessment):
            return None
        if assessment.memory_candidate is None:
            return None

        context = repo_context.copy() if repo_context else {}
        record = MemoryRecord(
            id=f"sk_{uuid.uuid4().hex[:8]}",
            type=MemoryType.SKILL,
            content=assessment.memory_candidate.summary,
            context={**context, **assessment.memory_candidate.context},
            confidence=assessment.memory_candidate.confidence or assessment.confidence,
            tags=["skill"],
            source_trial_id=assessment.trial_id,
        )
        self.store.save(record)
        return record

    def _should_write_skill(self, assessment: TrialAssessment) -> bool:
        if not assessment.is_valid:
            return False
        if assessment.confidence < 0.6:
            return False
        if assessment.memory_candidate is None:
            return False

        existing = self.store.query(
            memory_type=MemoryType.SKILL,
            min_confidence=0.3,
            limit=100,
        )
        for mem in existing:
            if self._is_duplicate(mem.content, assessment.memory_candidate.summary):
                self.store.update_validation(mem.id)
                return False
        return True

    def _is_duplicate(self, existing: str, new: str) -> bool:
        existing_words = set(existing.lower().split())
        new_words = set(new.lower().split())
        if not existing_words or not new_words:
            return False
        overlap = len(existing_words & new_words) / max(len(existing_words), len(new_words))
        return overlap > 0.7

    def _extract_tags(self, trial: TrialRecord, assessment: TrialAssessment) -> list[str]:
        tags = [trial.status.value]
        if trial.metric_name:
            tags.append(trial.metric_name)
        if assessment.better_than_best:
            tags.append("improvement")
        if trial.error_message:
            error_lower = trial.error_message.lower()
            if "oom" in error_lower or "out of memory" in error_lower:
                tags.append("oom")
            elif "timeout" in error_lower:
                tags.append("timeout")
            elif "nan" in error_lower:
                tags.append("nan")
        return tags
