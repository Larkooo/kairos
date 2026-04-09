"""Multi-timescale episodic memory bank.

Loosely inspired by the hippocampal / neocortical split: fast episodic
memory (small capacity, high detail, short-lived) + slower consolidated
memory (larger capacity, coarser resolution, long-lived). We model this
as a stack of fixed-capacity FIFO ring buffers ("tiers") with:

  * different capacities per tier (fast tier is small, slow is large)
  * different write strides per tier, so slow tiers only ingest every
    Nth write — a crude substitute for consolidation that nonetheless
    gives coarser-grained temporal resolution at longer scales
  * a learned cross-attention path that lets hidden states *query* the
    stored entries, weighted by time decay so old memories contribute
    less than recent ones

The retrieval path has its own Q/K/V/output projections (independent
of the transformer's self-attention) and a zero-initialised output gate,
so at construction the memory bank contributes exactly zero to the
hidden states. Training can open the gate and shape the projections.

Storage state (buffers) is kept as non-persistent module buffers so the
module moves correctly with `.to(device)` and `.cuda()`, but the stored
contents are *not* saved in state_dict — memory is conversational state,
not model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryStats:
    """Light-weight snapshot of memory occupancy for logging/debugging."""

    tier_fill: list[int]
    tier_capacity: list[int]

    def as_dict(self) -> dict:
        return {
            "tier_fill": list(self.tier_fill),
            "tier_capacity": list(self.tier_capacity),
        }


class MemoryTier(nn.Module):
    """A single FIFO ring buffer of (key, value, timestamp) triples.

    Storage is kept in fp32 buffers so that accumulated temporal decay
    calculations don't lose precision at long timescales, regardless of
    the model's compute dtype.
    """

    def __init__(
        self,
        capacity: int,
        key_dim: int,
        value_dim: int,
        write_stride: int = 1,
    ) -> None:
        super().__init__()
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if write_stride < 1:
            raise ValueError("write_stride must be >= 1")

        self.capacity = capacity
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.write_stride = write_stride

        self.register_buffer(
            "keys", torch.zeros(capacity, key_dim, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "values",
            torch.zeros(capacity, value_dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "timestamps", torch.zeros(capacity, dtype=torch.float32), persistent=False
        )
        # Bitmask of which slots are populated.
        self.register_buffer(
            "occupied",
            torch.zeros(capacity, dtype=torch.bool),
            persistent=False,
        )
        # Next write position in the ring buffer.
        self.register_buffer(
            "write_head", torch.zeros((), dtype=torch.long), persistent=False
        )
        # Total number of writes observed (used by stride logic).
        self.register_buffer(
            "writes_seen", torch.zeros((), dtype=torch.long), persistent=False
        )

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.timestamps.zero_()
        self.occupied.zero_()
        self.write_head.zero_()
        self.writes_seen.zero_()

    @torch.no_grad()
    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> int:
        """Append entries to the ring buffer, respecting the stride.

        Args:
            keys:       (N, key_dim)
            values:     (N, value_dim)
            timestamps: (N,) float

        Returns:
            Number of entries actually written (after stride filtering).
        """
        if keys.shape[0] != values.shape[0] or keys.shape[0] != timestamps.shape[0]:
            raise ValueError("keys/values/timestamps must share first dim")

        n = keys.shape[0]
        accepted: list[int] = []
        for i in range(n):
            write_idx = int(self.writes_seen.item())
            self.writes_seen += 1
            # Stride: only every Nth write lands in this tier.
            if write_idx % self.write_stride != 0:
                continue
            slot = int(self.write_head.item())
            self.keys[slot] = keys[i].float()
            self.values[slot] = values[i].float()
            self.timestamps[slot] = timestamps[i].float()
            self.occupied[slot] = True
            self.write_head[...] = (slot + 1) % self.capacity
            accepted.append(slot)
        return len(accepted)

    @property
    def fill(self) -> int:
        return int(self.occupied.sum().item())

    def forward(
        self,
        queries: torch.Tensor,
        current_time: torch.Tensor,
        decay_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve values for the given queries with time-decay weighting.

        Args:
            queries:       (batch, q_len, key_dim)
            current_time:  (batch, q_len) float timestamps of the queries
            decay_rate:    scalar (nonneg) — larger = faster forgetting

        Returns:
            (batch, q_len, value_dim) — zero if the tier is empty.
        """
        batch, q_len, _ = queries.shape
        if self.fill == 0:
            return torch.zeros(
                batch,
                q_len,
                self.value_dim,
                device=queries.device,
                dtype=queries.dtype,
            )

        occ = self.occupied  # (C,)
        # Use the occupied slots only.
        keys = self.keys[occ]  # (M, key_dim)
        values = self.values[occ]  # (M, value_dim)
        ts = self.timestamps[occ]  # (M,)

        # (batch, q, M) scores = <q, k> / sqrt(d)
        scale = 1.0 / (self.key_dim ** 0.5)
        scores = torch.einsum(
            "bqd,md->bqm", queries.to(torch.float32), keys
        ) * scale

        # Time decay: entries older than the current query time get a
        # negative bias in log-space proportional to elapsed time.
        dt = (current_time.to(torch.float32).unsqueeze(-1) - ts.view(1, 1, -1))
        # Only "past" entries should count; entries with dt < 0 came
        # "from the future" (shouldn't happen normally, but guard).
        dt = dt.clamp(min=0.0)
        scores = scores - decay_rate * dt

        weights = F.softmax(scores, dim=-1)  # (batch, q, M)
        retrieved = torch.einsum("bqm,md->bqd", weights, values)  # (b, q, vdim)
        return retrieved.to(queries.dtype)


class MultiTimescaleMemory(nn.Module):
    """Stack of memory tiers with learned Q/K/V retrieval projections.

    The module is designed to be plugged into a transformer layer as a
    side path: given the current hidden states and their timestamps, it
    retrieves a context vector from the memory and produces a zero-gated
    additive update to the hidden states.

    Writes are done externally (e.g. by the caller after a forward pass
    finishes, or in a dedicated "consolidation" step). Keeping writes
    external simplifies gradient semantics: we don't backprop through
    the stored memory contents.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        tier_capacities: tuple[int, ...] = (64, 256, 1024),
        tier_strides: tuple[int, ...] = (1, 4, 16),
        tier_decays: tuple[float, ...] = (1e-1, 1e-3, 1e-5),
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if not (len(tier_capacities) == len(tier_strides) == len(tier_decays)):
            raise ValueError("tier_* tuples must all have the same length")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        # Learned projections for query / write-time key / write-time value.
        self.q_proj = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.o_proj = nn.Linear(self.inner_dim * len(tier_capacities), hidden_size, bias=False)

        # Per-tier learnable decay rates (softplus to stay non-negative).
        # Initialised so softplus(raw) ~= the provided tier_decays.
        import math as _m

        raw_inits = []
        for d in tier_decays:
            if d <= 0:
                raw_inits.append(-6.0)  # ~0.00247
            else:
                raw_inits.append(_m.log(_m.expm1(d)))
        self.raw_decay = nn.Parameter(torch.tensor(raw_inits, dtype=torch.float32))

        # Zero-initialised output gate so the memory contribution is 0
        # at construction (same trick as the temporal decay bias gate).
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

        if zero_init_output:
            nn.init.zeros_(self.o_proj.weight)

        self.tiers = nn.ModuleList(
            [
                MemoryTier(
                    capacity=cap,
                    key_dim=self.inner_dim,
                    value_dim=self.inner_dim,
                    write_stride=stride,
                )
                for cap, stride in zip(tier_capacities, tier_strides)
            ]
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def reset(self) -> None:
        for t in self.tiers:
            t.reset()

    def stats(self) -> MemoryStats:
        return MemoryStats(
            tier_fill=[t.fill for t in self.tiers],
            tier_capacity=[t.capacity for t in self.tiers],
        )

    def effective_decays(self) -> torch.Tensor:
        return F.softplus(self.raw_decay)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    @torch.no_grad()
    def write(
        self,
        hidden_states: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """Store a batch of (hidden, timestamp) pairs into every tier.

        Tiers self-gate via their `write_stride`, so slow tiers only
        accept every Nth write.

        Args:
            hidden_states: (batch, seq, hidden)
            timestamps:    (batch, seq) float
        """
        if hidden_states.dim() != 3 or timestamps.dim() != 2:
            raise ValueError("hidden_states must be 3D, timestamps 2D")

        # Flatten across batch and seq: each token becomes one entry.
        b, s, _ = hidden_states.shape
        flat_hidden = hidden_states.reshape(b * s, -1)
        flat_ts = timestamps.reshape(b * s)

        keys = self.k_proj(flat_hidden).to(torch.float32)
        values = self.v_proj(flat_hidden).to(torch.float32)
        for tier in self.tiers:
            tier.write(keys, values, flat_ts)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend from current hidden states to stored memories.

        Returns an additive update (already multiplied by the gate):
            update = gate * o_proj(concat([tier_0_out, ..., tier_K-1_out]))

        Output shape matches `hidden_states`: (batch, seq, hidden).
        """
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states must be (batch, seq, hidden)")
        if timestamps.dim() != 2:
            raise ValueError("timestamps must be (batch, seq)")

        queries = self.q_proj(hidden_states)  # (b, s, inner)
        decays = self.effective_decays()

        tier_outputs: list[torch.Tensor] = []
        for i, tier in enumerate(self.tiers):
            out = tier(
                queries=queries,
                current_time=timestamps,
                decay_rate=decays[i],
            )
            tier_outputs.append(out.to(hidden_states.dtype))

        concat = torch.cat(tier_outputs, dim=-1)  # (b, s, K * inner)
        projected = self.o_proj(concat)  # (b, s, hidden)
        return self.gate * projected
