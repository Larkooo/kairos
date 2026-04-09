"""Config for the Kairos Gemma 3 model.

Extends Gemma3TextConfig with the extra knobs our temporal modules need:
  - continuous-time positional embedding (Fourier features on log-time)
  - per-head exponential attention decay over elapsed real time
  - multi-timescale external memory bank

Defaults are chosen so a freshly constructed KairosGemmaConfig loaded on
top of stock Gemma weights produces outputs identical to the stock model:
temporal contributions are gated by parameters initialised near zero.
"""

from __future__ import annotations

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


class KairosGemmaConfig(Gemma3TextConfig):
    """Gemma 3 text config with temporal extensions.

    Extra fields:
        temporal_enabled:        master switch; if False the model behaves
                                 exactly like stock Gemma 3.
        temporal_time_scale:     divide raw timestamps (seconds) by this
                                 value before computing Fourier features.
                                 Default 1.0 = seconds.
        temporal_num_frequencies: number of log-spaced Fourier frequencies
                                 for the continuous-time embedding.
        temporal_min_period:     shortest period (in scaled time units)
                                 covered by the Fourier features.
        temporal_max_period:     longest period covered.
        temporal_decay_enabled:  add per-head exponential decay bias to
                                 attention logits based on elapsed real
                                 time between query and key tokens.
        temporal_decay_init:     initial value (in 1/time_scale units) of
                                 the per-head decay rate. 0.0 means no
                                 decay at init.
        temporal_decay_per_head: if True, one decay rate per attention
                                 head; otherwise one rate per layer.
        memory_enabled:          enable multi-timescale memory bank.
        memory_num_tiers:        number of memory tiers (fast / medium /
                                 slow).
        memory_tier_sizes:       list of max entries per tier.
        memory_tier_decays:      list of decay rates (per tier) for the
                                 LRU/relevance score.
        memory_query_layers:     which transformer layer indices query the
                                 memory bank. Empty list disables.
    """

    model_type = "kairos_gemma"

    def __init__(
        self,
        temporal_enabled: bool = True,
        temporal_time_scale: float = 1.0,
        temporal_num_frequencies: int = 32,
        temporal_min_period: float = 1.0,
        temporal_max_period: float = 10_000_000.0,
        temporal_decay_enabled: bool = True,
        temporal_decay_init: float = 0.0,
        temporal_decay_per_head: bool = True,
        memory_enabled: bool = False,
        memory_num_tiers: int = 3,
        memory_tier_sizes: tuple[int, ...] = (64, 256, 1024),
        memory_tier_decays: tuple[float, ...] = (1e-1, 1e-3, 1e-5),
        memory_query_layers: tuple[int, ...] = (),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.temporal_enabled = temporal_enabled
        self.temporal_time_scale = temporal_time_scale
        self.temporal_num_frequencies = temporal_num_frequencies
        self.temporal_min_period = temporal_min_period
        self.temporal_max_period = temporal_max_period
        self.temporal_decay_enabled = temporal_decay_enabled
        self.temporal_decay_init = temporal_decay_init
        self.temporal_decay_per_head = temporal_decay_per_head
        self.memory_enabled = memory_enabled
        self.memory_num_tiers = memory_num_tiers
        self.memory_tier_sizes = tuple(memory_tier_sizes)
        self.memory_tier_decays = tuple(memory_tier_decays)
        self.memory_query_layers = tuple(memory_query_layers)


# Backward-compatible alias for older imports.
TemporalGemmaConfig = KairosGemmaConfig
