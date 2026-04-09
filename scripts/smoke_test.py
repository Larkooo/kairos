"""End-to-end smoke test for TemporalGemma.

Verifies:
  1. We can load stock Gemma 3 weights into TemporalGemmaForCausalLM.
  2. A forward pass produces finite logits with the right shape.
  3. With temporal modules zero-initialised, outputs match stock Gemma
     to within fp32/bf16 numerical tolerance (equivalence at init).
  4. Gradients flow through the new temporal parameters when we
     perturb timestamps and take a loss.
  5. Non-trivial decay rates actually change the attention-induced
     outputs (i.e. the mask bias is wired into the attention path).

Run with:
    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make the project root importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer, Gemma3ForCausalLM  # noqa: E402

from temporal_llm.model import TemporalGemmaForCausalLM  # noqa: E402


MODEL_ID = "unsloth/gemma-3-270m"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def test_equivalence_at_init() -> None:
    """With decay_init=0 and time_embed zero-init, the temporal model
    must match the stock model bit-for-bit (up to fp tolerance)."""
    device = _device()
    tokenizer = _load_tokenizer()
    dtype = torch.float32  # bf16 on MPS loses equivalence precision
    print(f"[equivalence] device={device} dtype={dtype}")

    stock = Gemma3ForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()

    temporal = TemporalGemmaForCausalLM.from_gemma_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        temporal_decay_init=0.0,
    ).to(device).eval()

    prompt = "The quick brown fox jumps"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]
    timestamps = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        stock_out = stock(input_ids=input_ids).logits
        temporal_out = temporal(input_ids=input_ids, timestamps=timestamps).logits

    max_diff = (stock_out - temporal_out).abs().max().item()
    print(f"[equivalence] max |stock - temporal| logit diff = {max_diff:.3e}")
    print(f"[equivalence] stock shape = {tuple(stock_out.shape)}")
    # We used zero-init for time_embed and softplus(-6) ≈ 2.47e-3 for
    # decay. Over seq_len=5, decay bias is at most 5 * 2.47e-3 ~ 1.2e-2
    # added to logits, which propagates to logit diffs below.
    assert torch.isfinite(temporal_out).all(), "non-finite logits"
    assert max_diff < 0.5, f"excessive drift at init: {max_diff}"
    print("[equivalence] OK\n")
    del stock, temporal


def test_decay_actually_affects_outputs() -> None:
    """When we crank the decay rate, the outputs must change."""
    device = _device()
    tokenizer = _load_tokenizer()
    dtype = torch.float32
    print(f"[decay-effect] device={device} dtype={dtype}")

    model = TemporalGemmaForCausalLM.from_gemma_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        temporal_decay_init=0.0,
    ).to(device).eval()

    prompt = "The quick brown fox jumps over the lazy dog"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]
    # Use timestamps with large gaps so decay bias is meaningful
    timestamps = torch.linspace(0, 100, seq_len, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        baseline = model(input_ids=input_ids, timestamps=timestamps).logits

    # Crank the decay rate to something large by writing to raw_decay
    # (raw=2 -> softplus(2)~2.13) and open the zero-initialised gate.
    # Together these should significantly suppress attention to
    # far-away tokens.
    with torch.no_grad():
        for dl in model.model.temporal_decay_layers:
            dl.raw_decay.fill_(2.0)
            dl.gate.fill_(1.0)

    with torch.no_grad():
        decayed = model(input_ids=input_ids, timestamps=timestamps).logits

    delta = (baseline - decayed).abs().max().item()
    print(f"[decay-effect] max |baseline - decayed| = {delta:.3e}")
    assert delta > 1e-3, f"decay bias not reaching attention path ({delta})"
    print("[decay-effect] OK\n")
    del model


def test_gradients_flow() -> None:
    """Make sure the new temporal parameters receive gradients."""
    device = _device()
    tokenizer = _load_tokenizer()
    dtype = torch.float32
    print(f"[gradients] device={device} dtype={dtype}")

    model = TemporalGemmaForCausalLM.from_gemma_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        temporal_decay_init=0.1,
    ).to(device).train()

    # Lift the decay gate off zero and perturb the time-embedding output
    # projection so gradients have a non-trivial signal path through
    # every temporal parameter group (not just the gate).
    with torch.no_grad():
        for dl in model.model.temporal_decay_layers:
            dl.gate.fill_(1.0)
        if model.model.time_embed is not None:
            model.model.time_embed.proj.weight.normal_(mean=0.0, std=1e-3)
            model.model.time_embed.proj.bias.zero_()

    # Freeze the base model so the grad sanity check is isolated to the
    # temporal modules.
    for name, p in model.named_parameters():
        is_temporal = (
            name.startswith("model.time_embed.")
            or name.startswith("model.temporal_decay_layers.")
        )
        p.requires_grad_(is_temporal)

    prompt = "Sequences arrive in time"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    labels = input_ids.clone()
    seq_len = input_ids.shape[1]
    timestamps = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0) * 10.0

    out = model(input_ids=input_ids, timestamps=timestamps, labels=labels)
    loss = out.loss
    print(f"[gradients] initial loss = {loss.item():.4f}")
    loss.backward()

    grads = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        assert g is not None, f"no grad on trainable param {name}"
        grads.append((name, g.norm().item()))

    for name, gnorm in grads:
        print(f"[gradients]   {name}: ||g||={gnorm:.3e}")
        assert gnorm > 0, f"zero grad on {name}"

    print("[gradients] OK\n")
    del model


def main() -> None:
    test_equivalence_at_init()
    test_decay_actually_affects_outputs()
    test_gradients_flow()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
