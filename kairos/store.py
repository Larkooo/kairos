"""Storage interfaces for Kairos memory records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .types import MemoryRecord, MemoryStatus, MemoryType, TaskState


@dataclass(slots=True)
class MemoryQuery:
    """Filter configuration for memory candidate lookup."""

    task_id: str | None = None
    episode_id: str | None = None
    types: set[MemoryType] = field(default_factory=set)
    statuses: set[MemoryStatus] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    limit: int = 100


class MemoryStore(Protocol):
    """Protocol for Kairos memory storage backends."""

    def write(self, records: list[MemoryRecord]) -> None: ...

    def update(self, records: list[MemoryRecord]) -> None: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def query(self, query: MemoryQuery) -> list[MemoryRecord]: ...

    def candidates_for_task(self, task_state: TaskState, top_k: int = 100) -> list[MemoryRecord]: ...


class InMemoryMemoryStore:
    """Simple in-process memory store for experimentation and tests."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def write(self, records: list[MemoryRecord]) -> None:
        for record in records:
            self._records[record.memory_id] = record

    def update(self, records: list[MemoryRecord]) -> None:
        for record in records:
            if record.memory_id not in self._records:
                raise KeyError(f"unknown memory_id: {record.memory_id}")
            self._records[record.memory_id] = record

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def query(self, query: MemoryQuery) -> list[MemoryRecord]:
        results = []
        for record in self._records.values():
            if query.task_id is not None and record.task_id != query.task_id:
                continue
            if query.episode_id is not None and record.episode_id != query.episode_id:
                continue
            if query.types and record.type not in query.types:
                continue
            if query.statuses and record.status not in query.statuses:
                continue
            if query.tags and not query.tags.intersection(record.tags):
                continue
            results.append(record)

        results.sort(
            key=lambda record: (
                record.importance,
                record.last_accessed_at,
                record.created_at,
            ),
            reverse=True,
        )
        return results[: query.limit]

    def candidates_for_task(self, task_state: TaskState, top_k: int = 100) -> list[MemoryRecord]:
        """Return a broad candidate set before task-aware reranking."""
        query = MemoryQuery(
            task_id=task_state.task_id,
            episode_id=task_state.episode_id,
            limit=top_k,
        )
        primary = self.query(query)
        if len(primary) >= top_k:
            return primary[:top_k]

        seen = {record.memory_id for record in primary}
        fallback = self.query(MemoryQuery(limit=top_k * 2))
        for record in fallback:
            if record.memory_id in seen:
                continue
            primary.append(record)
            seen.add(record.memory_id)
            if len(primary) >= top_k:
                break
        return primary
