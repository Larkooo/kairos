"""Core interfaces for Kairos task-aware memory and training data.

These types model the agent-facing pieces that sit around the language
model:

  * typed memory records with temporal metadata
  * explicit task state for the current objective
  * observations and actions collected during an agent episode
  * ranking and consolidation payloads for training and inference

The goal is to make memory selection a first-class concern instead of
encoding all long-range state purely as token history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class MemoryType(str, Enum):
    """Typed memory categories used by Kairos."""

    FACT = "fact"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    PLAN_STEP = "plan_step"
    OBSERVATION = "observation"
    RESULT = "result"
    FAILURE = "failure"
    PREFERENCE = "preference"
    OPEN_LOOP = "open_loop"


class MemoryStatus(str, Enum):
    """Lifecycle state for a memory record."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    STALE = "stale"
    ARCHIVED = "archived"


@dataclass(slots=True)
class MemoryRecord:
    """A single long-lived memory unit for the agent."""

    memory_id: str
    episode_id: str
    task_id: str | None
    type: MemoryType
    status: MemoryStatus
    content: str
    summary: str | None = None
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    last_updated_at: float = 0.0
    importance: float = 0.0
    confidence: float = 1.0
    source: str | None = None
    tags: list[str] = field(default_factory=list)
    dependency_ids: list[str] = field(default_factory=list)
    related_memory_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def age(self, now: float) -> float:
        """Return the record age in the same units as the timestamps."""
        return max(0.0, now - self.created_at)

    def time_since_access(self, now: float) -> float:
        """Return elapsed time since the record was last accessed."""
        return max(0.0, now - self.last_accessed_at)


@dataclass(slots=True)
class TaskState:
    """Explicit working state for the task the agent is handling now."""

    task_id: str
    episode_id: str
    user_query: str
    current_goal: str
    subgoals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    open_loops: list[str] = field(default_factory=list)
    active_files: list[str] = field(default_factory=list)
    active_entities: list[str] = field(default_factory=list)
    current_timestamp: float = 0.0
    context_window_budget: int = 8192
    metadata: dict[str, Any] = field(default_factory=dict)

    def task_text(self) -> str:
        """Flatten the task state into a single retrieval-friendly string."""
        parts = [
            self.user_query,
            self.current_goal,
            *self.subgoals,
            *self.constraints,
            *self.open_loops,
            *self.active_files,
            *self.active_entities,
        ]
        return " ".join(part for part in parts if part)


@dataclass(slots=True)
class Observation:
    """A single observed step in an agent episode."""

    timestamp: float
    user_message: str | None = None
    tool_name: str | None = None
    tool_input: str | None = None
    tool_output: str | None = None
    notes: str | None = None

    def text(self) -> str:
        """Flatten the observation into a single retrieval-friendly string."""
        parts = [
            self.user_message,
            self.tool_name,
            self.tool_input,
            self.tool_output,
            self.notes,
        ]
        return " ".join(part for part in parts if part)


@dataclass(slots=True)
class AgentAction:
    """A model output or external action taken during an episode."""

    kind: Literal["respond", "tool_call", "plan_update", "memory_write"]
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


@dataclass(slots=True)
class RankerInput:
    """Inputs for scoring memory candidates against the current task."""

    task_state: TaskState
    observation: Observation
    candidates: list[MemoryRecord]
    top_k: int = 8


@dataclass(slots=True)
class MemoryScore:
    """Decomposed score for a single memory candidate."""

    memory_id: str
    total: float
    semantic_relevance: float = 0.0
    task_match: float = 0.0
    temporal_fit: float = 0.0
    importance_score: float = 0.0
    dependency_bonus: float = 0.0
    stale_penalty: float = 0.0


@dataclass(slots=True)
class RankerOutput:
    """Ranked candidate scores plus the chosen memory IDs."""

    scored: list[MemoryScore]
    selected_ids: list[str]


@dataclass(slots=True)
class ConsolidatorInput:
    """Raw episode data to compress into durable memory records."""

    task_state: TaskState
    observations: list[Observation]
    actions: list[AgentAction]


@dataclass(slots=True)
class ConsolidatorOutput:
    """Writes and updates produced by the consolidator."""

    writes: list[MemoryRecord]
    updates: list[MemoryRecord]
    resolves: list[str]


@dataclass(slots=True)
class TrainingBatch:
    """A single supervised training item for Kairos."""

    batch_id: str
    task_state: TaskState
    observation: Observation
    candidate_memories: list[MemoryRecord]
    positive_memory_ids: list[str]
    selected_memories: list[MemoryRecord] = field(default_factory=list)
    target_response: str | None = None
    target_action: AgentAction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
