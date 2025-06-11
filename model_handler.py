import torch
import torch.nn.functional as F


def nucleus_sample(probits: torch.Tensor, prob_threshold: float):
    sorted_probs, sorted_indices = torch.sort(probits, dim=-1, descending=True)
    prev_cumulative_probs = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    sorted_indices_to_remove = prev_cumulative_probs >= prob_threshold
    sorted_probs[sorted_indices_to_remove] = 0
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    sampled_idx = sorted_indices.gather(dim=-1, index=sampled_sorted_idx)
    return sampled_idx


def greedy_sample(probits: torch.Tensor):
    return probits.argmax(dim=-1, keepdim=True)


def stream(model, input_ids: torch.Tensor, max_length, prob_threshold: float, temperature: float):
    # TODO: KV-cache to avoid quadratic computational complexity in `max_length`
    output_ids = input_ids.clone()
    for _ in range(max_length):
        with torch.no_grad():
            next_token_logits = model(output_ids)[:, -1, :]
            # TODO: If repetitions resurface as an issue: repetition_penalty and no_repeat_ngram_size
            next_token_shaped_logits = next_token_logits / temperature
            next_token_probits = F.softmax(next_token_shaped_logits, dim=-1)
            next_token_id = nucleus_sample(probits=next_token_probits, prob_threshold=prob_threshold)
            output_ids = torch.cat([output_ids, next_token_id], dim=1)
            yield next_token_id.item()


def print_stream(
    model, tokenizer, prompt: str, device, max_length=50, prob_threshold: float = 0.95, temperature: float = 1.0
):
    print(prompt, end="", flush=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    for token in stream(
        model=model, input_ids=input_ids, max_length=max_length, prob_threshold=prob_threshold, temperature=temperature
    ):
        if token == tokenizer.eos_token_id:
            break
        print(tokenizer.decode(token), end="", flush=True)
    print("", flush=True)