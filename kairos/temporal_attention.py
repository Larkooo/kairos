"""Per-head temporal attention decay.

The stock transformer attention has no notion of *how long ago* a token
was: only token distance matters (via causal masking / RoPE). For a
temporal LLM we want attention scores to decay as a function of real
elapsed time, with different heads specialising in different timescales
(some heads look at recent context, others at long-range).

We implement this as an additive bias on the pre-softmax attention
logits:

    logits[b, h, i, j] += -softplus(raw_lambda[h]) * |t_i - t_j|

Key properties:
  - softplus keeps the decay rate non-negative, so this can only shrink
    attention, never amplify it.
  - Initialised so softplus(raw) is very small (near 0); at init the
    bias is ~0 and the model behaves like stock Gemma, so pretrained
    weights can be loaded without distortion.
  - Per-head decay lets different heads specialise.

We return a (batch, heads, q_len, kv_len) tensor that should be added
into the attention mask (which is itself an additive bias in log-space).

This module is agnostic to the attention backend: it only builds a bias
tensor. The caller is responsible for folding it into whatever mask the
model's attention path consumes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDecayBias(nn.Module):
    """Builds an additive, per-head temporal-decay bias from timestamps."""

    def __init__(
        self,
        num_heads: int,
        time_scale: float = 1.0,
        init_decay: float = 0.0,
        per_head: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.time_scale = float(time_scale)
        self.per_head = per_head

        # We parameterise the decay rate as raw -> softplus(raw), so that
        # the effective rate is always non-negative. We want the initial
        # effective rate to equal `init_decay`; solve raw from the
        # softplus inverse when possible.
        if init_decay < 0:
            raise ValueError("init_decay must be >= 0")
        if init_decay == 0.0:
            raw_init = -6.0  # softplus(-6) ~= 0.00247
        else:
            raw_init = math.log(math.expm1(init_decay))

        shape = (num_heads,) if per_head else (1,)
        self.raw_decay = nn.Parameter(torch.full(shape, raw_init, dtype=torch.float32))

        # Gate: multiplicative scalar on the bias, initialised to zero so
        # the temporal contribution is *exactly* 0 at init. This matters
        # because even a small per-layer bias compounds across 18
        # decoder layers and visibly drifts the logits. Gradients still
        # flow through `gate` (the non-zero decay bias is multiplied by
        # it) so training can lift the gate away from zero.
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def effective_decay(self) -> torch.Tensor:
        """Return the non-negative effective decay rate per head."""
        return F.softplus(self.raw_decay)

    def forward(
        self,
        timestamps: torch.Tensor,
        q_len: int | None = None,
    ) -> torch.Tensor:
        """Build the decay bias for a batch of timestamps.

        Args:
            timestamps: (batch, seq) float tensor of timestamps. `seq` is
                the *total* kv length (past + current). When decoding a
                cached prefix, pass the concatenated timestamps.
            q_len: length of the query slice. Defaults to the full seq,
                which is the standard prefill case.

        Returns:
            Bias tensor of shape (batch, heads, q_len, kv_len) ready to
            be added into an attention mask. Values are <= 0.
        """
        if timestamps.dim() != 2:
            raise ValueError(
                f"timestamps must be (batch, seq); got {tuple(timestamps.shape)}"
            )
        batch, kv_len = timestamps.shape
        q_len = kv_len if q_len is None else q_len

        t = timestamps.float() / self.time_scale  # (b, kv_len)
        # Queries are the last q_len tokens (prefill: all of them; decode:
        # the newly arrived tokens appended to the cached prefix).
        q_times = t[:, -q_len:]  # (b, q_len)
        kv_times = t  # (b, kv_len)

        # |dt_ij| = |q_time_i - kv_time_j|, shape (b, q_len, kv_len)
        dt = (q_times.unsqueeze(-1) - kv_times.unsqueeze(1)).abs()

        decay = self.effective_decay()  # (heads,) or (1,)
        # (b, 1, q, kv) * (1, h, 1, 1) -> (b, h, q, kv)
        bias = -dt.unsqueeze(1) * decay.view(1, -1, 1, 1)

        if not self.per_head:
            # Broadcast scalar decay across heads
            bias = bias.expand(batch, self.num_heads, q_len, kv_len)

        return bias * self.gate
