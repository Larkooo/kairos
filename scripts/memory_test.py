"""Unit + integration tests for the multi-timescale memory bank.

Exercises the memory module in isolation (no Gemma needed) and then
loads a full KairosGemma with `memory_enabled=True` to verify the
memory path integrates cleanly with the transformer forward pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kairos.memory_bank import MultiTimescaleMemory  # noqa: E402
from kairos.model import KairosGemmaForCausalLM  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


MODEL_ID = "unsloth/gemma-3-270m"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_empty_memory_returns_zero() -> None:
    print("[memory:empty] running")
    device = _device()
    mem = MultiTimescaleMemory(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        tier_capacities=(8, 32, 128),
        tier_strides=(1, 2, 4),
        tier_decays=(0.1, 0.01, 0.001),
    ).to(device)

    hidden = torch.randn(2, 3, 64, device=device)
    ts = torch.tensor([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]], device=device)

    # Empty memory + zero-init output projection & gate => exactly zero
    out = mem(hidden, ts)
    assert out.shape == hidden.shape
    assert out.abs().max().item() == 0.0, out.abs().max().item()
    print("[memory:empty] OK")


def test_writes_land_in_tiers() -> None:
    print("[memory:write] running")
    device = _device()
    mem = MultiTimescaleMemory(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        tier_capacities=(8, 32, 128),
        tier_strides=(1, 2, 4),
        tier_decays=(0.1, 0.01, 0.001),
    ).to(device)

    # Write 16 entries (one batch of 2 x 8 tokens).
    hidden = torch.randn(2, 8, 64, device=device)
    ts = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8)
    mem.write(hidden, ts)

    stats = mem.stats()
    print(f"[memory:write] tier fills = {stats.tier_fill}")

    # Tier 0 (stride 1) sees all 16; capacity 8 => wraps, fill == 8
    assert stats.tier_fill[0] == 8, stats.tier_fill
    # Tier 1 (stride 2) sees 8 writes; capacity 32 => fill 8
    assert stats.tier_fill[1] == 8, stats.tier_fill
    # Tier 2 (stride 4) sees 4 writes; capacity 128 => fill 4
    assert stats.tier_fill[2] == 4, stats.tier_fill
    print("[memory:write] OK")


def test_retrieval_nonzero_after_write_with_gate() -> None:
    print("[memory:retrieve] running")
    device = _device()
    mem = MultiTimescaleMemory(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        tier_capacities=(8, 32, 128),
        tier_strides=(1, 2, 4),
        tier_decays=(0.1, 0.01, 0.001),
    ).to(device)

    # Initialise the output proj to something non-trivial and open the gate.
    with torch.no_grad():
        torch.nn.init.normal_(mem.o_proj.weight, mean=0.0, std=0.1)
        mem.gate.fill_(1.0)

    hidden = torch.randn(1, 4, 64, device=device)
    ts = torch.arange(4, dtype=torch.float32, device=device).unsqueeze(0)
    mem.write(hidden, ts)

    out = mem(hidden, ts)
    assert out.abs().max().item() > 0, "retrieved update should be non-zero"
    print(f"[memory:retrieve] update abs max = {out.abs().max().item():.3e}")
    print("[memory:retrieve] OK")


def test_memory_gradients_flow() -> None:
    print("[memory:grad] running")
    device = _device()
    mem = MultiTimescaleMemory(
        hidden_size=32,
        num_heads=2,
        head_dim=16,
        tier_capacities=(4,),
        tier_strides=(1,),
        tier_decays=(0.0,),
    ).to(device)

    with torch.no_grad():
        torch.nn.init.normal_(mem.o_proj.weight, mean=0.0, std=0.1)
        mem.gate.fill_(1.0)

    hidden = torch.randn(1, 4, 32, device=device, requires_grad=False)
    ts = torch.arange(4, dtype=torch.float32, device=device).unsqueeze(0)
    mem.write(hidden, ts)

    query = torch.randn(1, 2, 32, device=device)
    q_ts = torch.tensor([[5.0, 6.0]], device=device)
    out = mem(query, q_ts)
    loss = out.pow(2).mean()
    loss.backward()

    assert mem.q_proj.weight.grad is not None
    assert mem.o_proj.weight.grad is not None
    assert mem.gate.grad is not None
    assert mem.gate.grad.abs().item() > 0
    print(
        f"[memory:grad] q_proj ||g||={mem.q_proj.weight.grad.norm():.3e}  "
        f"o_proj ||g||={mem.o_proj.weight.grad.norm():.3e}  "
        f"gate ||g||={mem.gate.grad.abs().item():.3e}"
    )
    print("[memory:grad] OK")


def test_full_model_with_memory_equivalence() -> None:
    """Loading KairosGemma with memory_enabled=True but empty memory
    and zero-init gate must match a baseline temporal model exactly."""
    print("[memory:model] running")
    device = _device()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = torch.float32

    baseline = KairosGemmaForCausalLM.from_gemma_pretrained(
        MODEL_ID, torch_dtype=dtype, temporal_decay_init=0.0
    ).to(device).eval()

    with_mem = KairosGemmaForCausalLM.from_gemma_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        temporal_decay_init=0.0,
        memory_enabled=True,
        memory_tier_sizes=(16, 64, 256),
        memory_tier_decays=(0.1, 0.01, 0.001),
        memory_query_layers=(5, 11, 17),  # read at each full-attention layer
    ).to(device).eval()

    enc = tok("Time passes and things change", return_tensors="pt").to(device)
    ids = enc["input_ids"]
    ts = torch.arange(ids.shape[1], dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        l_base = baseline(input_ids=ids, timestamps=ts).logits
        l_mem = with_mem(input_ids=ids, timestamps=ts).logits

    diff = (l_base - l_mem).abs().max().item()
    print(f"[memory:model] max |baseline - with_mem| = {diff:.3e}")
    # Empty memory + zero-init gate + zero-init o_proj => identical
    assert diff == 0.0, f"unexpected drift with empty memory path: {diff}"

    # Now write some memories, open the gate, and see the outputs change.
    with torch.no_grad():
        hidden_proxy = with_mem.model.embed_tokens(ids).float()
        with_mem.model.memory.write(hidden_proxy, ts)
        with_mem.model.memory.gate.fill_(0.5)
        torch.nn.init.normal_(
            with_mem.model.memory.o_proj.weight, mean=0.0, std=0.02
        )

    with torch.no_grad():
        l_after = with_mem(input_ids=ids, timestamps=ts).logits
    delta = (l_base - l_after).abs().max().item()
    print(f"[memory:model] max |baseline - memory_active| = {delta:.3e}")
    assert delta > 1e-3, f"memory contribution not reaching outputs: {delta}"
    print("[memory:model] OK")


def main() -> None:
    test_empty_memory_returns_zero()
    test_writes_land_in_tiers()
    test_retrieval_nonzero_after_write_with_gate()
    test_memory_gradients_flow()
    test_full_model_with_memory_equivalence()
    print("\nALL MEMORY TESTS PASSED")


if __name__ == "__main__":
    main()
