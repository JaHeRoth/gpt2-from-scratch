from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk


def tokenize(batch, tokenizer, context_length: int) -> dict[str, list[torch.Tensor]]:
    # TODO: Sequence packing
    outputs = tokenizer(
        batch["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    return {
        "input_ids": [
            input_ids
            for length, input_ids in zip(outputs["length"], outputs["input_ids"])
            if length == context_length
        ]
    }


def load_preprocessed(hf_path: str, hf_name: str, tokenizer, context_length: int):
    dataset = load_dataset(hf_path, hf_name)

    local_path = Path(f"outputs/{hf_path}__{hf_name}__tokenized.hf")  # TODO: Use absolute path
    if local_path.exists():
        tokenized_ds = load_from_disk(str(local_path))
    else:
        tokenized_ds = dataset.map(
            lambda batch: tokenize(batch, tokenizer, context_length),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        tokenized_ds.save_to_disk(local_path)
    return dataset, tokenized_ds
