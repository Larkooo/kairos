import unittest
import tempfile

import torch

from kairos import KairosGemmaConfig, KairosGemmaForCausalLM
from kairos.memory_bank import MemoryTier, MultiTimescaleMemory
from kairos.temporal_embeddings import ContinuousTimeEmbedding
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM


def tiny_model(**overrides):
    config = KairosGemmaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, layer_types=["sliding_attention", "full_attention"],
        sliding_window=8, **overrides,
    )
    return KairosGemmaForCausalLM(config).eval()


class TemporalModelTest(unittest.TestCase):
    def test_local_stock_checkpoint_matches_at_initialization(self):
        config = Gemma3TextConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, layer_types=["sliding_attention", "full_attention"], sliding_window=8,
        )
        config._attn_implementation = "eager"
        stock = Gemma3ForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            stock.save_pretrained(directory)
            model = KairosGemmaForCausalLM.from_gemma_pretrained(
                directory, local_files_only=True,
            ).eval()
            ids = torch.tensor([[1, 2, 3]])
            with torch.no_grad():
                expected = stock(input_ids=ids).logits
                actual = model(input_ids=ids, timestamps=torch.tensor([[0., 1., 60.]])).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cached_decode_across_sliding_window(self):
        model = tiny_model()
        ids = torch.arange(1, 14).unsqueeze(0)
        timestamps = torch.arange(13, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            for decay in model.model.temporal_decay_layers:
                decay.gate.fill_(.3)
            expected = model(input_ids=ids, timestamps=timestamps, use_cache=False).logits[:, -1:]
            prefix = model(input_ids=ids[:, :-1], timestamps=timestamps[:, :-1], use_cache=True)
            actual = model(input_ids=ids[:, -1:], timestamps=timestamps,
                           past_key_values=prefix.past_key_values, use_cache=True).logits
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    def test_cached_decode_matches_full_forward(self):
        torch.manual_seed(7)
        model = tiny_model()
        with torch.no_grad():
            model.model.time_embed.proj.weight.normal_(std=.01)
            for decay in model.model.temporal_decay_layers:
                decay.gate.fill_(.3)
            ids = torch.tensor([[1, 2, 3, 4]])
            timestamps = torch.tensor([[0., 2., 3., 10.]])
            full = model(input_ids=ids, timestamps=timestamps, use_cache=False)
            first = model(input_ids=ids[:, :-1], timestamps=timestamps[:, :-1], use_cache=True)
            last = model(input_ids=ids[:, -1:], timestamps=timestamps,
                         past_key_values=first.past_key_values, use_cache=True)
        self.assertEqual(last.logits.shape, (1, 1, 32))
        torch.testing.assert_close(full.logits[:, -1:], last.logits, rtol=1e-4, atol=1e-5)

    def test_generate_extends_timestamps(self):
        model = tiny_model()
        with torch.no_grad():
            result = model.generate(
                torch.tensor([[1, 2, 3]]), timestamps=torch.tensor([[0., 2., 3.]]),
                max_new_tokens=3, do_sample=False, eos_token_id=None, pad_token_id=0,
            )
        self.assertEqual(result.shape, (1, 6))

    def test_bfloat16_temporal_embedding(self):
        embedding = ContinuousTimeEmbedding(8).to(torch.bfloat16)
        out = embedding(torch.tensor([[0., 1.]]))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(out).all())

    def test_default_memory_can_learn_from_zero_output(self):
        torch.manual_seed(9)
        memory = MultiTimescaleMemory(
            8, 1, 8, tier_capacities=(4,), tier_strides=(1,), tier_decays=(.1,),
        )
        memory.write(torch.randn(1, 2, 8), torch.tensor([[0., 1.]]))
        output = memory(torch.randn(1, 1, 8), torch.tensor([[2.]]))
        self.assertEqual(output.abs().sum().item(), 0)
        (output - 1).square().mean().backward()
        self.assertGreater(memory.o_proj.weight.grad.abs().sum().item(), 0)

    def test_future_memories_are_excluded(self):
        memory = MemoryTier(4, 2, 2)
        memory.write(torch.ones(2, 2), torch.tensor([[1., 2.], [100., 100.]]),
                     torch.tensor([1., 10.]))
        result = memory(torch.ones(1, 2, 2), torch.tensor([[0., 2.]]), torch.tensor(.1))
        torch.testing.assert_close(result, torch.tensor([[[0., 0.], [1., 2.]]]))

    def test_memory_dtype_conversion_preserves_timestamps(self):
        memory = MemoryTier(4, 2, 2)
        memory.write(torch.ones(1, 2), torch.ones(1, 2), torch.tensor([100001.]))
        memory.to(dtype=torch.bfloat16)
        self.assertEqual(memory.timestamps.dtype, torch.float32)
        self.assertEqual(memory.timestamps[0].item(), 100001.)

    def test_timestamps_must_include_cached_prefix(self):
        model = tiny_model()
        with self.assertRaisesRegex(ValueError, "timestamps must have shape"):
            model(input_ids=torch.tensor([[1, 2]]), timestamps=torch.tensor([[1.]]))
