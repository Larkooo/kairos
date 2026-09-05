"""Continuous-time positional embedding.

Stock transformer positional embeddings encode *integer token index*.
A temporal LLM should encode *real elapsed time*. We implement that with
a log-scale Fourier feature bank (similar in spirit to NeRF's positional
encoding on continuous coordinates) followed by a small MLP projection
into the model's hidden dimension.

Given a float timestamp t (seconds), we:
  1. Scale by config.temporal_time_scale -> tau
  2. For k = 0..K-1 with periods log-spaced in [min_period, max_period],
     compute (sin(2*pi*tau/period_k), cos(2*pi*tau/period_k))
  3. Concatenate and project to hidden_size

The projection's output weight is initialised near zero so that the
temporal contribution is ~0 at init and a pretrained Gemma model loaded
on top of this module behaves like the stock model until fine-tuned.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ContinuousTimeEmbedding(nn.Module):
    """Maps a (batch, seq) float timestamp tensor to (batch, seq, hidden)."""

    def __init__(
        self,
        hidden_size: int,
        num_frequencies: int = 32,
        min_period: float = 1.0,
        max_period: float = 10_000_000.0,
        time_scale: float = 1.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be >= 1")
        if min_period <= 0 or max_period <= min_period:
            raise ValueError("require 0 < min_period < max_period")

        self.hidden_size = hidden_size
        self.num_frequencies = num_frequencies
        self.time_scale = float(time_scale)

        # Log-spaced periods (not frequencies) so we cover many orders of
        # magnitude of time cleanly: periods = geomspace(min_period, max_period, K)
        log_min = math.log(min_period)
        log_max = math.log(max_period)
        periods = torch.exp(
            torch.linspace(log_min, log_max, num_frequencies, dtype=torch.float32)
        )
        # Angular frequency w_k = 2*pi / period_k
        inv_periods = (2.0 * math.pi) / periods
        self.register_buffer("inv_periods", inv_periods, persistent=False)

        feature_dim = 2 * num_frequencies  # sin + cos per frequency
        # Two-layer MLP is overkill for a smoke test; a single projection is
        # enough to route Fourier features into the model's hidden dim.
        self.proj = nn.Linear(feature_dim, hidden_size, bias=True)

        if zero_init:
            # Zero-init output weights so the added temporal signal is 0
            # at initialisation and the wrapped Gemma behaves identically
            # to the stock pretrained model.
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps.

        Args:
            timestamps: (batch, seq) float tensor of timestamps in seconds
                (or whatever units the model was configured with).

        Returns:
            (batch, seq, hidden_size) embedding additive to the token
            embeddings.
        """
        if timestamps.dim() != 2:
            raise ValueError(
                f"timestamps must be (batch, seq); got shape {tuple(timestamps.shape)}"
            )
        # Scale into the unit system the inv_periods were computed for.
        scaled = timestamps.float() / self.time_scale  # (b, s)
        # Outer product (b, s, K)
        phases = scaled.unsqueeze(-1) * self.inv_periods
        sin_feats = torch.sin(phases)
        cos_feats = torch.cos(phases)
        feats = torch.cat([sin_feats, cos_feats], dim=-1)  # (b, s, 2K)
        out = self.proj(feats.to(self.proj.weight.dtype))  # (b, s, hidden)
        return out.to(dtype=self._out_dtype())

    def _out_dtype(self) -> torch.dtype:
        return self.proj.weight.dtype
