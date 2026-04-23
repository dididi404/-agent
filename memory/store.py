"""memory/store.py — SQLite 记忆存储层"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from schemas.common import MemoryType
from schemas.state import MemoryRecord


class MemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_table(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT DEFAULT '{}',
                confidence REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                last_validated TEXT NOT NULL,
                validation_count INTEGER DEFAULT 0,
                source_trial_id TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_trial_id)
        """)
        conn.commit()

    def save(self, record: MemoryRecord) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, type, content, context, confidence, tags,
                created_at, last_validated, validation_count, source_trial_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.type.value,
                record.content,
                json.dumps(record.context, default=str),
                record.confidence,
                json.dumps(record.tags),
                record.created_at.isoformat(),
                record.last_validated.isoformat(),
                record.validation_count,
                record.source_trial_id,
            ),
        )
        conn.commit()

    def get(self, memory_id: str) -> MemoryRecord | None:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def query(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        conn = self._get_conn()
        conditions = ["confidence >= ?"]
        params: list = [min_confidence]

        if memory_type is not None:
            conditions.append("type = ?")
            params.append(memory_type.value)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY last_validated DESC LIMIT ?",
            params + [limit],
        ).fetchall()

        results = [self._row_to_record(r) for r in rows]

        if tags:
            tag_set = set(tags)
            results = [r for r in results if tag_set & set(r.tags)]

        return results

    def update_validation(self, memory_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            """UPDATE memories
               SET last_validated = ?, validation_count = validation_count + 1
               WHERE id = ?""",
            (datetime.now().isoformat(), memory_id),
        )
        conn.commit()

    def delete(self, memory_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()

    def count(self, memory_type: MemoryType | None = None) -> int:
        conn = self._get_conn()
        if memory_type:
            row = conn.execute("SELECT COUNT(*) FROM memories WHERE type = ?", (memory_type.value,)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            context=json.loads(row["context"]),
            confidence=row["confidence"],
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_validated=datetime.fromisoformat(row["last_validated"]),
            validation_count=row["validation_count"],
            source_trial_id=row["source_trial_id"],
        )
