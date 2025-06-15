from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import numpy as np
import matplotlib.pyplot as plt


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


def train(
    model,
    optimizer,
    tokenizer,
    tokenized_train_ds,
    tokenized_eval_ds,
    device,
    train_batch_size: int = 64,
    num_epochs: int = 100,
    warmup_steps: int = 2000,
    log_period: int | None = 25,
    stream_period: int | None = 100,
    eval_period: int | None = 250,
    checkpoint_period: int = 50,
    checkpoint_path: Path | None = None,
    stream_prompt: str = "In 1814, the",
) -> tuple[list[float], list[float]]:
    """Trains `model` (in-place) and returns training and eval losses."""

    # if torch.cuda.device_count() > 1:
    #     model = nn.parallel.DistributedDataParallel(model)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        tokenized_train_ds,
        shuffle=True,
    )

    train_dl = DataLoader(
        tokenized_train_ds,
        batch_size=train_batch_size,
        sampler=train_sampler,
        collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt"),
    )
    validation_dl = DataLoader(
        tokenized_eval_ds,
        batch_size=train_batch_size * 2,
        shuffle=False,
        collate_fn=lambda x: tokenizer.pad(x, return_tensors="pt"),
    )

    total_steps = num_epochs * len(train_dl)
    decay_steps = total_steps - warmup_steps
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=0.0),
        ],
        milestones=[warmup_steps],
    )

    model.train()
    train_losses = []
    eval_losses = []
    for epoch_i in range(num_epochs):
        train_sampler.set_epoch(epoch_i)
        for batch_i, batch in enumerate(train_dl):
            X: torch.Tensor = batch.input_ids.to(device)[:, :-1].contiguous()
            y: torch.Tensor = batch.input_ids.to(device)[:, 1:].contiguous()
            logits = model(X)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                y.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if log_period is not None and batch_i % log_period == 0:
                train_losses.append(loss.item())
                print(f"Batch {batch_i + 1}/{len(train_dl)} in epoch {epoch_i + 1}/{num_epochs}: Loss {loss.item()}")

            if stream_period is not None and batch_i % stream_period == 0:
                model.eval()
                print_stream(model=model, tokenizer=tokenizer, prompt=stream_prompt, device=device, max_length=50)
                print("", flush=True)
                model.train()

            if eval_period is not None and batch_i % eval_period == 0:
                with torch.no_grad():
                    model.eval()
                    avg_val_loss = torch.Tensor([0.0]).to(device)
                    for validation_batch in validation_dl:
                        X_val: torch.Tensor = validation_batch.input_ids.to(device)[:, :-1].contiguous()
                        y_val: torch.Tensor = validation_batch.input_ids.to(device)[:, 1:].contiguous()
                        val_logits = model(X_val)
                        avg_val_loss += nn.functional.cross_entropy(
                            val_logits.view(-1, val_logits.shape[-1]),
                            y_val.view(-1),
                            ignore_index=tokenizer.pad_token_id,
                        ) / len(validation_dl)
                    eval_losses.append(avg_val_loss.item())
                    print(f"Avg. validation Loss {avg_val_loss.item()}")
                    model.train()

            if batch_i % checkpoint_period == 0 and checkpoint_path is not None:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                path = checkpoint_path / f"epoch_{epoch_i + 1}_batch_{batch_i + 1}"
                print(f"Saving state dict checkpoint to '{path}'.")
                torch.save(model.state_dict(), path)

        print("=" * 40 + f"COMPLETED EPOCH {epoch_i + 1}/{num_epochs}" + "=" * 40)

        train_loss_batch_i = np.arange(len(train_losses)) * log_period
        eval_loss_batch_i = np.arange(len(eval_losses)) * eval_period
        plt.plot(train_loss_batch_i, train_losses, "--o", label="Train Loss")
        plt.plot(eval_loss_batch_i, eval_losses, "--o", label="Eval Loss")
        plt.xlabel("Batch number")
        plt.ylabel("Loss")
        plt.title(f"Loss over first {epoch_i + 1} epoch(s)")
        plt.legend()
        plt.grid()
        plt.show()

    return train_losses, eval_losses
