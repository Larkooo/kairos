"""Kairos package for time-aware Gemma models.

Layers a continuous-time positional encoding, per-head temporal attention
decay, and multi-timescale memory on top of the stock Gemma 3 decoder so
that "when" becomes a first-class input to the model alongside tokens.
"""

from .config import KairosGemmaConfig, TemporalGemmaConfig
from .consolidator import Consolidator, HeuristicConsolidator
from .ranker import HeuristicMemoryRanker, Ranker
from .store import InMemoryMemoryStore, MemoryQuery, MemoryStore
from .temporal_embeddings import ContinuousTimeEmbedding
from .model import (
    KairosGemmaForCausalLM,
    KairosGemmaTextModel,
    TemporalGemmaForCausalLM,
    TemporalGemmaTextModel,
)
from .types import (
    AgentAction,
    ConsolidatorInput,
    ConsolidatorOutput,
    MemoryRecord,
    MemoryScore,
    MemoryStatus,
    MemoryType,
    Observation,
    RankerInput,
    RankerOutput,
    TaskState,
    TrainingBatch,
)

__all__ = [
    "AgentAction",
    "Consolidator",
    "ConsolidatorInput",
    "ConsolidatorOutput",
    "KairosGemmaConfig",
    "HeuristicConsolidator",
    "HeuristicMemoryRanker",
    "InMemoryMemoryStore",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryScore",
    "MemoryStatus",
    "MemoryStore",
    "MemoryType",
    "Observation",
    "Ranker",
    "RankerInput",
    "RankerOutput",
    "TaskState",
    "TemporalGemmaConfig",
    "TrainingBatch",
    "ContinuousTimeEmbedding",
    "KairosGemmaForCausalLM",
    "KairosGemmaTextModel",
    "TemporalGemmaForCausalLM",
    "TemporalGemmaTextModel",
]
