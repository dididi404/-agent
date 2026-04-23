"""observability/trace_logger.py — 结构化 JSON trace 记录"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class TraceEvent:
    def __init__(self, node: str, agent: str = "", action: str = "", **kwargs):
        self.timestamp = datetime.now().isoformat()
        self.node = node
        self.agent = agent
        self.action = action
        self.data = kwargs
        self.duration_ms: int | None = None

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "node": self.node,
            "agent": self.agent,
            "action": self.action,
            "duration_ms": self.duration_ms,
        }
        d.update(self.data)
        return d


class TraceLogger:
    def __init__(self, task_id: str, output_dir: str = "traces"):
        self.task_id = task_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[TraceEvent] = []
        self._start_time = time.time()

    def log(self, node: str, agent: str = "", action: str = "", **kwargs) -> TraceEvent:
        event = TraceEvent(node=node, agent=agent, action=action, **kwargs)
        self.events.append(event)
        return event

    def log_from_messages(self, messages: list[dict[str, Any]]) -> None:
        for msg in messages:
            self.log(
                node="message",
                agent=msg.get("role", ""),
                action="message",
                content=msg.get("content", ""),
            )

    def log_tool_calls(self, records: list) -> None:
        for rec in records:
            self.log(
                node="tool_call",
                agent=rec.caller_agent,
                action=rec.tool_name,
                status=rec.output_status.value,
                latency_ms=rec.latency_ms,
                intent=rec.intent,
            )

    def save(self) -> str:
        total_time = time.time() - self._start_time
        trace = {
            "task_id": self.task_id,
            "total_duration_sec": round(total_time, 2),
            "total_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }
        path = self.output_dir / f"trace_{self.task_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
        return str(path)
