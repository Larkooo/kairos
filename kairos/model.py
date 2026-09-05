# Portions adapted from the Gemma 3 implementation in Hugging Face Transformers.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use those portions except in compliance with the License.
# A copy is in licenses/APACHE-2.0.txt and at https://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Modifications add timestamp inputs, temporal bias, external memory, and cache handling.

"""Kairos Gemma 3 causal LM.

Subclasses the stock `Gemma3TextModel` / `Gemma3ForCausalLM` and wires
in the temporal modules:

  * a continuous-time token embedding added to the input embeddings
  * a per-layer, per-head temporal decay bias added to attention logits

We deliberately keep the rest of Gemma's architecture untouched so that
pretrained weights can be loaded straight from a stock Gemma checkpoint
via `KairosGemmaForCausalLM.from_gemma_pretrained(...)`. At init the
temporal modules contribute approximately zero, so the wrapped model
matches the base model's outputs until it is fine-tuned.

Design notes:
  * RoPE position_ids are still computed as integer token indices.
    Timestamps augment, not replace, positional signal.
  * Temporal attention decay is applied by *adding* a negative bias to
    the stock causal/sliding-window mask. This works with the eager
    attention backend; we force eager to keep the bias path simple and
    backend-agnostic. SDPA/flash could be supported later by threading
    the bias through `attn_mask` arguments explicitly.
  * Masks are created once (stock behaviour); the per-layer decay bias
    is added inside the layer loop so each layer can learn its own
    decay rates.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3TextModel,
    _bidirectional_window_overlay,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from .config import KairosGemmaConfig
from .memory_bank import MultiTimescaleMemory
from .temporal_attention import TemporalDecayBias
from .temporal_embeddings import ContinuousTimeEmbedding


def _zero_init_temporal_modules(root: nn.Module) -> None:
    """Re-zero temporal parameters after the generic Gemma `post_init`.

    `post_init()` calls `_init_weights` on every submodule, which applies
    the default nn.Linear initialisation to our `ContinuousTimeEmbedding`
    projection and overwrites the zero weights we set in its `__init__`.
    We rerun the zero init ourselves so the temporal contribution is
    exactly zero at construction time.
    """
    for module in root.modules():
        if isinstance(module, ContinuousTimeEmbedding):
            nn.init.zeros_(module.proj.weight)
            nn.init.zeros_(module.proj.bias)
        elif isinstance(module, TemporalDecayBias):
            nn.init.zeros_(module.gate)
        elif isinstance(module, MultiTimescaleMemory):
            nn.init.ones_(module.gate)
            nn.init.zeros_(module.o_proj.weight)


def _expand_mask_for_heads(
    mask: torch.Tensor, num_heads: int
) -> torch.Tensor:
    """Expand a (batch, 1, q, kv) mask to (batch, heads, q, kv) so we can
    add per-head biases without breaking the causal/sliding semantics."""
    if mask is None:
        return None
    if mask.dim() != 4:
        raise ValueError(f"expected 4D mask, got shape {tuple(mask.shape)}")
    if mask.shape[1] == num_heads:
        return mask
    if mask.shape[1] != 1:
        raise ValueError(
            f"mask head dim must be 1 or {num_heads}, got {mask.shape[1]}"
        )
    return mask.expand(mask.shape[0], num_heads, mask.shape[2], mask.shape[3])


class KairosGemmaTextModel(Gemma3TextModel):
    """Gemma3 text model + continuous-time embedding + temporal decay."""

    config_class = KairosGemmaConfig

    def __init__(self, config: KairosGemmaConfig):
        if config._attn_implementation not in (None, "eager"):
            raise ValueError("Kairos temporal attention currently requires the eager backend")
        config._attn_implementation = "eager"
        super().__init__(config)
        self.temporal_config = config

        if config.temporal_enabled:
            self.time_embed = ContinuousTimeEmbedding(
                hidden_size=config.hidden_size,
                num_frequencies=config.temporal_num_frequencies,
                min_period=config.temporal_min_period,
                max_period=config.temporal_max_period,
                time_scale=config.temporal_time_scale,
                zero_init=True,
            )
        else:
            self.time_embed = None

        if config.temporal_decay_enabled:
            self.temporal_decay_layers = nn.ModuleList(
                [
                    TemporalDecayBias(
                        num_heads=config.num_attention_heads,
                        time_scale=config.temporal_time_scale,
                        init_decay=config.temporal_decay_init,
                        per_head=config.temporal_decay_per_head,
                    )
                    for _ in range(config.num_hidden_layers)
                ]
            )
        else:
            self.temporal_decay_layers = None

        # Multi-timescale memory. One shared memory bank; `memory_query_layers`
        # lists the transformer layer indices that should read from it.
        if config.memory_enabled:
            head_dim = getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
            self.memory = MultiTimescaleMemory(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                head_dim=head_dim,
                tier_capacities=config.memory_tier_sizes,
                tier_strides=tuple(max(1, 4 ** i) for i in range(len(config.memory_tier_sizes))),
                tier_decays=config.memory_tier_decays,
            )
            self.memory_query_layers = set(config.memory_query_layers)
        else:
            self.memory = None
            self.memory_query_layers = set()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        timestamps: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        use_cache = self.config.use_cache if use_cache is None else use_cache
        current_length = inputs_embeds.shape[1]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if timestamps is not None:
            expected = (inputs_embeds.shape[0], past_length + current_length)
            if tuple(timestamps.shape) != expected:
                raise ValueError(f"timestamps must have shape {expected}, including the cached prefix")
            if not torch.isfinite(timestamps).all():
                raise ValueError("timestamps must be finite")
            timestamps = timestamps.to(inputs_embeds.device)

        # Add continuous-time embedding to the input embeddings.
        if self.time_embed is not None and timestamps is not None:
            time_feats = self.time_embed(timestamps[:, -current_length:]).to(dtype=inputs_embeds.dtype)
            inputs_embeds = inputs_embeds + time_feats

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )
            position_ids = position_ids.unsqueeze(0)

        # Build the stock causal mask dict (Gemma3 uses one mask per layer
        # type: full attention + sliding window).
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = mask_kwargs.copy()

            if self.config.use_bidirectional_attention:
                mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(
                    True, dtype=torch.bool
                )
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                    self.config.sliding_window
                )

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(
                    **sliding_mask_kwargs
                ),
            }

        # Determine the q_len / kv_len situation for the decay bias.
        # During prefill both equal seq_len. During cached decode the kv
        # length is past + current. The decay module takes the full
        # timestamps tensor and a q_len hint.
        if timestamps is not None and self.temporal_decay_layers is not None:
            q_len = inputs_embeds.shape[1]
            # If a cache is in use, the caller is responsible for passing
            # the concatenated (past + current) timestamps so we can
            # compute |t_q - t_kv| properly.
            expected_kv = timestamps.shape[1]
            if expected_kv < q_len:
                raise ValueError(
                    "timestamps length must be >= current input length; "
                    f"got timestamps {expected_kv} vs inputs {q_len}"
                )

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states, position_ids, layer_type
            )

        num_heads = self.config.num_attention_heads
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            layer_mask = causal_mask_mapping[layer_type]

            # Add per-layer temporal decay bias into this layer's mask.
            if (
                timestamps is not None
                and self.temporal_decay_layers is not None
                and layer_mask is not None
            ):
                bias = self.temporal_decay_layers[i](
                    timestamps, q_len=hidden_states.shape[1]
                )  # (b, h, q, kv)
                bias = bias.to(dtype=layer_mask.dtype, device=layer_mask.device)
                # Sliding-window caches retain only the latest key positions.
                bias = bias[..., -layer_mask.shape[-1]:]
                expanded = _expand_mask_for_heads(layer_mask, num_heads)
                layer_mask = expanded + bias

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings[layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

            # Optional memory-bank read at this layer.
            if (
                self.memory is not None
                and i in self.memory_query_layers
                and timestamps is not None
            ):
                # Use *current-chunk* timestamps (last q_len of the
                # concatenated tensor) as the query times.
                q_len = hidden_states.shape[1]
                mem_update = self.memory(
                    hidden_states,
                    timestamps[:, -q_len:],
                )
                hidden_states = hidden_states + mem_update

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class KairosGemmaForCausalLM(Gemma3ForCausalLM):
    """Causal LM head on top of `KairosGemmaTextModel`."""

    config_class = KairosGemmaConfig

    def __init__(self, config: KairosGemmaConfig):
        if config._attn_implementation not in (None, "eager"):
            raise ValueError("Kairos temporal attention currently requires the eager backend")
        config._attn_implementation = "eager"
        # Skip Gemma3ForCausalLM.__init__ (which instantiates Gemma3TextModel)
        # so we can substitute our temporal variant.
        super(Gemma3ForCausalLM, self).__init__(config)
        self.model = KairosGemmaTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        # post_init() re-randomises all nn.Linear weights, including the
        # zero-init projection inside ContinuousTimeEmbedding. Re-zero
        # so the temporal contribution is exactly zero at construction.
        _zero_init_temporal_modules(self)

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1,
    ):
        updated = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        timestamps = updated.get("timestamps")
        if timestamps is not None:
            # Generated answer tokens share the time of the final prompt token.
            updated["timestamps"] = torch.cat(
                [timestamps, timestamps[:, -1:].expand(-1, num_new_tokens)], dim=1
            )
        return updated

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        timestamps: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            timestamps=timestamps,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_gemma_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        temporal_enabled: bool = True,
        temporal_decay_enabled: bool = True,
        temporal_decay_init: float = 0.0,
        temporal_time_scale: float = 1.0,
        temporal_num_frequencies: int = 32,
        temporal_min_period: float = 1.0,
        temporal_max_period: float = 10_000_000.0,
        temporal_decay_per_head: bool = True,
        memory_enabled: bool = False,
        memory_tier_sizes: tuple[int, ...] = (64, 256, 1024),
        memory_tier_decays: tuple[float, ...] = (1e-1, 1e-3, 1e-5),
        memory_query_layers: tuple[int, ...] = (),
        attn_implementation: str = "eager",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "KairosGemmaForCausalLM":
        """Load stock Gemma 3 weights into a KairosGemma model.

        We force `attn_implementation="eager"` by default because the
        per-layer decay bias is added to the dense attention mask, which
        is straightforward in eager mode. Other backends can be wired
        later.
        """
        # 1) Load the stock config and extend it into a KairosGemmaConfig.
        if attn_implementation != "eager":
            raise ValueError("Kairos temporal attention currently requires the eager backend")
        config_kwargs = {
            key: kwargs[key] for key in
            ("cache_dir", "force_download", "local_files_only", "revision", "token", "subfolder")
            if key in kwargs
        }
        base_config = Gemma3TextConfig.from_pretrained(
            pretrained_model_name_or_path, **config_kwargs
        )
        config_dict = base_config.to_dict()
        # Strip fields that KairosGemmaConfig adds back with its own defaults.
        for k in [
            "temporal_enabled",
            "temporal_time_scale",
            "temporal_num_frequencies",
            "temporal_min_period",
            "temporal_max_period",
            "temporal_decay_enabled",
            "temporal_decay_init",
            "temporal_decay_per_head",
            "memory_enabled",
            "memory_num_tiers",
            "memory_tier_sizes",
            "memory_tier_decays",
            "memory_query_layers",
        ]:
            config_dict.pop(k, None)
        temporal_config = KairosGemmaConfig(
            temporal_enabled=temporal_enabled,
            temporal_time_scale=temporal_time_scale,
            temporal_num_frequencies=temporal_num_frequencies,
            temporal_min_period=temporal_min_period,
            temporal_max_period=temporal_max_period,
            temporal_decay_enabled=temporal_decay_enabled,
            temporal_decay_init=temporal_decay_init,
            temporal_decay_per_head=temporal_decay_per_head,
            memory_enabled=memory_enabled,
            memory_num_tiers=len(memory_tier_sizes),
            memory_tier_sizes=memory_tier_sizes,
            memory_tier_decays=memory_tier_decays,
            memory_query_layers=memory_query_layers,
            **config_dict,
        )
        temporal_config._attn_implementation = attn_implementation

        # 2) Load stock weights into a stock Gemma3ForCausalLM.
        load_kwargs = dict(kwargs)
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        load_kwargs["attn_implementation"] = attn_implementation
        base_model = Gemma3ForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **load_kwargs
        )

        # 3) Instantiate the temporal model and copy tensors over.
        temporal_model = cls(temporal_config)
        if torch_dtype is not None:
            temporal_model = temporal_model.to(dtype=torch_dtype)

        missing, unexpected = temporal_model.load_state_dict(
            base_model.state_dict(), strict=False
        )
        # `missing` should only contain the new temporal parameters.
        allowed_missing_prefixes = (
            "model.time_embed.",
            "model.temporal_decay_layers.",
            "model.memory.",
        )
        unexpected_missing = [
            k for k in missing if not k.startswith(allowed_missing_prefixes)
        ]
        if unexpected_missing:
            raise RuntimeError(
                "Unexpected missing keys when loading Gemma weights into "
                f"KairosGemma: {unexpected_missing[:10]}"
            )
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading Gemma weights: {unexpected[:10]}"
            )

        # Free the intermediate base model.
        del base_model
        return temporal_model


# Backward-compatible aliases for older imports.
TemporalGemmaTextModel = KairosGemmaTextModel
TemporalGemmaForCausalLM = KairosGemmaForCausalLM
