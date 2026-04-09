"""Task-aware heuristic ranking for Kairos memory retrieval."""

from __future__ import annotations

import math
import re
from typing import Protocol

from .types import (
    MemoryRecord,
    MemoryScore,
    MemoryStatus,
    MemoryType,
    RankerInput,
    RankerOutput,
)


class Ranker(Protocol):
    """Protocol for memory reranking models."""

    def score(self, inputs: RankerInput) -> RankerOutput: ...


class HeuristicMemoryRanker:
    """Cheap relevance-first ranker with temporal priors.

    This is not meant to be the final learned model. It provides a usable
    baseline and a clear target interface for future training.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")

    def score(self, inputs: RankerInput) -> RankerOutput:
        now = inputs.task_state.current_timestamp or inputs.observation.timestamp
        task_tokens = self._tokens(inputs.task_state.task_text())
        obs_tokens = self._tokens(inputs.observation.text())
        active_tokens = task_tokens | obs_tokens

        scored: list[MemoryScore] = []
        for record in inputs.candidates:
            semantic_relevance = self._semantic_relevance(record, active_tokens)
            task_match = self._task_match(record, inputs.task_state, active_tokens)
            temporal_fit = self._temporal_fit(record, now)
            importance_score = max(0.0, min(1.0, record.importance))
            dependency_bonus = self._dependency_bonus(record)
            stale_penalty = self._stale_penalty(record, now)
            total = (
                semantic_relevance
                + task_match
                + temporal_fit
                + importance_score
                + dependency_bonus
                - stale_penalty
            )
            scored.append(
                MemoryScore(
                    memory_id=record.memory_id,
                    total=total,
                    semantic_relevance=semantic_relevance,
                    task_match=task_match,
                    temporal_fit=temporal_fit,
                    importance_score=importance_score,
                    dependency_bonus=dependency_bonus,
                    stale_penalty=stale_penalty,
                )
            )

        scored.sort(key=lambda item: item.total, reverse=True)
        selected_ids = [item.memory_id for item in scored[: inputs.top_k]]
        return RankerOutput(scored=scored, selected_ids=selected_ids)

    def _semantic_relevance(self, record: MemoryRecord, active_tokens: set[str]) -> float:
        if not active_tokens:
            return 0.0
        record_tokens = self._tokens(" ".join([record.content, record.summary or "", *record.tags]))
        if not record_tokens:
            return 0.0
        overlap = len(active_tokens & record_tokens)
        return overlap / math.sqrt(len(record_tokens))

    def _task_match(self, record: MemoryRecord, task_state, active_tokens: set[str]) -> float:
        score = 0.0
        if record.task_id == task_state.task_id:
            score += 0.75
        if record.episode_id == task_state.episode_id:
            score += 0.35
        tag_overlap = active_tokens.intersection(record.tags)
        score += 0.15 * len(tag_overlap)
        return score

    def _temporal_fit(self, record: MemoryRecord, now: float) -> float:
        age = record.age(now)
        age_hours = age / 3600.0 if age > 0 else 0.0

        # Stable facts and preferences should decay slowly; ephemeral
        # observations and plan steps should decay faster.
        if record.type in {MemoryType.FACT, MemoryType.CONSTRAINT, MemoryType.PREFERENCE}:
            base = 0.5
            decay = 0.02
        elif record.type in {MemoryType.GOAL, MemoryType.OPEN_LOOP}:
            base = 0.8
            decay = 0.01
        else:
            base = 0.7
            decay = 0.12
        return max(0.0, base - decay * age_hours)

    def _dependency_bonus(self, record: MemoryRecord) -> float:
        score = 0.0
        if record.status == MemoryStatus.ACTIVE:
            score += 0.15
        if record.type in {MemoryType.GOAL, MemoryType.OPEN_LOOP, MemoryType.PLAN_STEP}:
            score += 0.2
        if record.dependency_ids:
            score += min(0.25, 0.05 * len(record.dependency_ids))
        return score

    def _stale_penalty(self, record: MemoryRecord, now: float) -> float:
        penalty = 0.0
        age_hours = record.age(now) / 3600.0 if now else 0.0
        if record.status == MemoryStatus.STALE:
            penalty += 0.6
        elif record.status == MemoryStatus.ARCHIVED:
            penalty += 1.0
        elif record.status == MemoryStatus.RESOLVED and record.type != MemoryType.FACT:
            penalty += 0.35
        if record.type in {MemoryType.OBSERVATION, MemoryType.RESULT}:
            penalty += min(0.5, 0.05 * age_hours)
        return penalty

    def _tokens(self, text: str) -> set[str]:
        return set(self._TOKEN_RE.findall(text.lower()))
