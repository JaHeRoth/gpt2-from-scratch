from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk


def tokenize(batch, tokenizer, context_length: int) -> dict[str, list[torch.Tensor]]:
    # +1 since last token of sequences won't be used as input (since has no next token)
    row_length = context_length + 1
    raw_tokenized_batch = tokenizer(batch["text"])
    filtered = [
        input_ids
        for input_ids in raw_tokenized_batch.data["input_ids"]
        if len(input_ids) != 0
    ]
    flattened = [tokenizer.eos_token_id] + [
        token_id
        for input_ids in filtered
        for token_id in input_ids + [tokenizer.eos_token_id]
    ]
    foldable = flattened + [tokenizer.pad_token_id] * ((-len(flattened)) % row_length)
    return {
        "input_ids": torch.tensor(foldable).view(-1, row_length)
    }


def load_preprocessed(hf_path: str, hf_name: str, tokenizer, context_length: int):
    dataset = load_dataset(hf_path, hf_name, split="train")
    train_val_split = dataset.train_test_split(test_size=0.01)

    local_path = Path(f"outputs/{hf_path}__{hf_name}__tokenized.hf")
    if local_path.exists():
        tokenized_ds = load_from_disk(str(local_path))
    else:
        tokenized_ds = train_val_split.map(
            lambda batch: tokenize(batch, tokenizer, context_length),
            batched=True,
            remove_columns=train_val_split["train"].column_names,
        )
        tokenized_ds.save_to_disk(local_path)
    return dataset, tokenized_ds
