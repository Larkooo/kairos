"""Kairos package for time-aware Gemma models.

Layers a continuous-time positional encoding, per-head temporal attention
decay, and multi-timescale memory on top of the stock Gemma 3 decoder so
that "when" becomes a first-class input to the model alongside tokens.
"""

from .config import KairosGemmaConfig, TemporalGemmaConfig
from .temporal_embeddings import ContinuousTimeEmbedding
from .model import (
    KairosGemmaForCausalLM,
    KairosGemmaTextModel,
    TemporalGemmaForCausalLM,
    TemporalGemmaTextModel,
)

__all__ = [
    "KairosGemmaConfig",
    "TemporalGemmaConfig",
    "ContinuousTimeEmbedding",
    "KairosGemmaForCausalLM",
    "KairosGemmaTextModel",
    "TemporalGemmaForCausalLM",
    "TemporalGemmaTextModel",
]
