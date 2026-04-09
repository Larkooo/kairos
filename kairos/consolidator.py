"""Heuristic trace consolidation into durable Kairos memory records."""

from __future__ import annotations

import hashlib
from typing import Protocol

from .types import (
    AgentAction,
    ConsolidatorInput,
    ConsolidatorOutput,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
)


class Consolidator(Protocol):
    """Protocol for modules that compress traces into memory records."""

    def consolidate(self, inputs: ConsolidatorInput) -> ConsolidatorOutput: ...


class HeuristicConsolidator:
    """Turn raw observations and actions into a small useful memory set.

    This baseline intentionally favors explicit task state, user signals,
    and executed actions over storing every token seen by the agent.
    """

    def consolidate(self, inputs: ConsolidatorInput) -> ConsolidatorOutput:
        writes: list[MemoryRecord] = []

        writes.append(
            self._record(
                inputs=inputs,
                record_type=MemoryType.GOAL,
                status=MemoryStatus.ACTIVE,
                content=inputs.task_state.current_goal,
                summary="Current agent goal",
                importance=0.95,
                tags=["goal"],
            )
        )

        for constraint in inputs.task_state.constraints:
            writes.append(
                self._record(
                    inputs=inputs,
                    record_type=MemoryType.CONSTRAINT,
                    status=MemoryStatus.ACTIVE,
                    content=constraint,
                    summary="Task constraint",
                    importance=0.85,
                    tags=["constraint"],
                )
            )

        for open_loop in inputs.task_state.open_loops:
            writes.append(
                self._record(
                    inputs=inputs,
                    record_type=MemoryType.OPEN_LOOP,
                    status=MemoryStatus.ACTIVE,
                    content=open_loop,
                    summary="Unresolved work item",
                    importance=0.9,
                    tags=["open_loop"],
                )
            )

        for observation in inputs.observations:
            if observation.user_message:
                writes.append(
                    self._record(
                        inputs=inputs,
                        record_type=MemoryType.OBSERVATION,
                        status=MemoryStatus.ACTIVE,
                        content=observation.user_message,
                        summary="User-provided observation",
                        importance=0.65,
                        timestamp=observation.timestamp,
                        tags=["user"],
                    )
                )
            if observation.tool_output:
                writes.append(
                    self._record(
                        inputs=inputs,
                        record_type=MemoryType.RESULT,
                        status=MemoryStatus.ACTIVE,
                        content=observation.tool_output,
                        summary=f"Tool result from {observation.tool_name or 'tool'}",
                        importance=0.7,
                        timestamp=observation.timestamp,
                        tags=["tool_output", observation.tool_name or "tool"],
                    )
                )

        for action in inputs.actions:
            writes.extend(self._records_for_action(inputs, action))

        deduped: dict[str, MemoryRecord] = {}
        for record in writes:
            deduped[record.memory_id] = record

        return ConsolidatorOutput(
            writes=list(deduped.values()),
            updates=[],
            resolves=[],
        )

    def _records_for_action(
        self,
        inputs: ConsolidatorInput,
        action: AgentAction,
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        if action.kind == "plan_update" and action.text:
            records.append(
                self._record(
                    inputs=inputs,
                    record_type=MemoryType.PLAN_STEP,
                    status=MemoryStatus.ACTIVE,
                    content=action.text,
                    summary="Planned next step",
                    importance=0.75,
                    tags=["plan"],
                )
            )
        elif action.kind == "tool_call":
            details = action.tool_name or "tool"
            if action.tool_args:
                details = f"{details}: {action.tool_args}"
            records.append(
                self._record(
                    inputs=inputs,
                    record_type=MemoryType.DECISION,
                    status=MemoryStatus.ACTIVE,
                    content=details,
                    summary="Tool call chosen by the agent",
                    importance=0.6,
                    tags=["tool_call", action.tool_name or "tool"],
                )
            )
        elif action.kind == "respond" and action.text:
            records.append(
                self._record(
                    inputs=inputs,
                    record_type=MemoryType.RESULT,
                    status=MemoryStatus.ACTIVE,
                    content=action.text,
                    summary="Agent response",
                    importance=0.55,
                    tags=["response"],
                )
            )
        elif action.kind == "memory_write" and action.text:
            records.append(
                self._record(
                    inputs=inputs,
                    record_type=MemoryType.FACT,
                    status=MemoryStatus.ACTIVE,
                    content=action.text,
                    summary="Explicit agent memory write",
                    importance=0.8,
                    tags=["memory_write"],
                )
            )
        return records

    def _record(
        self,
        *,
        inputs: ConsolidatorInput,
        record_type: MemoryType,
        status: MemoryStatus,
        content: str,
        summary: str,
        importance: float,
        tags: list[str],
        timestamp: float | None = None,
    ) -> MemoryRecord:
        created_at = inputs.task_state.current_timestamp if timestamp is None else timestamp
        digest = hashlib.sha1(
            f"{inputs.task_state.task_id}|{record_type.value}|{content}|{created_at}".encode("utf-8")
        ).hexdigest()[:16]
        return MemoryRecord(
            memory_id=f"mem_{digest}",
            episode_id=inputs.task_state.episode_id,
            task_id=inputs.task_state.task_id,
            type=record_type,
            status=status,
            content=content,
            summary=summary,
            created_at=created_at,
            last_accessed_at=created_at,
            last_updated_at=created_at,
            importance=importance,
            source="heuristic_consolidator",
            tags=tags,
        )
