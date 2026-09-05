# Kairos

Timestamp-aware Gemma 3 experiments, with temporal embeddings, attention bias,
and an explicit memory bank.

Use it to prototype models for event logs or conversations where elapsed time
matters: two adjacent tokens can describe events a second or a week apart.
The package also provides a structured memory store, a task-aware heuristic
ranker, and a consolidator for recording facts, decisions, and unresolved work.

This is a research prototype. The mechanisms are implemented and tested; no
task-level accuracy improvement or general memory capability is established.

## Install and try

Python 3.12 or newer:

```sh
git clone https://github.com/Larkooo/kairos.git
cd kairos
uv venv --python 3.12
uv pip install -e .
uv run python examples/temporal_forward.py
uv run python -m unittest discover -s tests -v
```

The example runs a small, randomly initialized model on CPU. It checks an
event-time input and cached continuation without downloading model weights.
Its token IDs are synthetic; the output is not meaningful generated text.

For experiments with a pretrained checkpoint:

```python
import torch
from kairos import KairosGemmaForCausalLM

model = KairosGemmaForCausalLM.from_gemma_pretrained(
    "/path/to/a/local/gemma-3-checkpoint",
    local_files_only=True,
    torch_dtype=torch.float32,
)
```

Supply `timestamps` with shape `(batch, tokens)` alongside input IDs. Use
elapsed seconds relative to an episode's start, rather than large Unix
timestamps. During cached decoding, include timestamps for the full prefix
and the new input. During `generate`, answer tokens retain the last prompt
timestamp. New observed events should be passed with their actual event time.
The temporal modules need training before they affect the pretrained model.

## Memory

Enable `memory_enabled=True` and select `memory_query_layers` to add memory
reads. Write hidden states and their timestamps explicitly through
`model.model.memory.write(hidden_states, timestamps)`; call `reset()` between
independent episodes. Future entries are excluded from earlier queries.

Storage is a shared FIFO bank with different sampling strides across tiers.
It is not isolated per batch element: use a separate model or reset the bank
for unrelated sessions. Writes detach stored contents from autograd; the read
path can train, while write-time key/value projections do not receive gradients
through stored entries. Memory contents are not saved in model checkpoints.
The structured store and ranker are separate interfaces, not automatic model
training or a persistent database.

## Validation and scope

The offline tests cover cached/full-forward equivalence, generation timestamp
extension, mixed precision, memory gradient flow, future-entry filtering, and
the structured memory interfaces. Optional checkpoint smoke tests are in
`scripts/smoke_test.py` and `scripts/memory_test.py`; they can download the
configured checkpoint and are separate from the offline suite.

Only eager attention is supported. The Transformers range is restricted to
the version family exercised here because the wrapper uses Gemma internals.
Timestamps augment RoPE; they do not replace token positions. The trainable
temporal-bias gate is signed, so a learned model is not constrained to forget
monotonically. Neither this interface nor its tests demonstrate improved
reasoning, continual learning, or biological fidelity.

## License

Original Kairos code is MIT-licensed. Portions of `kairos/model.py` adapt the
Gemma 3 implementation from Hugging Face Transformers and retain its Apache 2.0
notice; see [NOTICE](NOTICE) and [the upstream license](licenses/APACHE-2.0.txt).
Model weights are obtained separately and retain their own terms.
