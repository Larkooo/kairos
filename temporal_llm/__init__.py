"""Temporal transformer built on top of Gemma 3.

Layers a continuous-time positional encoding, per-head temporal attention
decay, and multi-timescale memory on top of the stock Gemma 3 decoder so
that "when" becomes a first-class input to the model alongside tokens.
"""

from .config import TemporalGemmaConfig
from .temporal_embeddings import ContinuousTimeEmbedding
from .model import TemporalGemmaForCausalLM, TemporalGemmaTextModel

__all__ = [
    "TemporalGemmaConfig",
    "ContinuousTimeEmbedding",
    "TemporalGemmaForCausalLM",
    "TemporalGemmaTextModel",
]
