"""Run an event-time forward pass and cached continuation without model downloads."""
import torch

from kairos import KairosGemmaConfig, KairosGemmaForCausalLM


def main():
    torch.manual_seed(7)
    config = KairosGemmaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
    )
    model = KairosGemmaForCausalLM(config).eval()
    # Three synthetic events, separated by a minute and then an hour.
    ids = torch.tensor([[1, 2, 3]])
    times = torch.tensor([[0., 60., 3660.]])
    with torch.no_grad():
        first = model(input_ids=ids, timestamps=times, use_cache=True)
        next_event = model(
            input_ids=torch.tensor([[4]]),
            timestamps=torch.tensor([[0., 60., 3660., 3661.]]),
            past_key_values=first.past_key_values, use_cache=True,
        )
    print(f"Event logits: {tuple(first.logits.shape)}")
    print(f"Cached continuation: {tuple(next_event.logits.shape)}")
    print("Synthetic inputs, random weights; no model-quality claim.")


if __name__ == "__main__":
    main()
