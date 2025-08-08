from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from datasets.formatting.formatting import LazyDict
from transformers import PreTrainedTokenizerBase
from datasets.exceptions import DatasetNotFoundError


def tokenize(
    batch: LazyDict, tokenizer: PreTrainedTokenizerBase, context_length: int
) -> dict[str, torch.Tensor]:
    dtype = torch.uint16 if len(tokenizer) < 2 ** 16 else torch.uint32
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
        "input_ids": torch.tensor(foldable, dtype=dtype).view(-1, row_length)
    }


def load_preprocessed(
    author: str, dataset: str, subset: str, sampled_percent: int, test_size: int, context_length: int, tokenizer: PreTrainedTokenizerBase
) -> DatasetDict:
    tokenized_name = f"{author}__{dataset}__{subset}__{sampled_percent}__{test_size}__{context_length}__tokenized"
    local_path = Path(f"outputs/datasets/{tokenized_name}.hf")

    if local_path.exists():
        print("Found tokenized dataset locally, so loading that.")
        return load_from_disk(str(local_path))

    try:
        print("Attempting to load tokenized dataset from HuggingFace Hub.")
        return load_dataset(f"jaheroth/{tokenized_name}")
    except DatasetNotFoundError:
        print(f"Could not find tokenized dataset locally or on HuggingFace Hub, so will recreate.")

    ds = load_dataset(f"{author}/{dataset}", subset, split=f"train[:{sampled_percent}%]")
    train_val_split = ds.train_test_split(test_size=test_size)
    tokenized_ds = train_val_split.map(
        lambda batch: tokenize(batch, tokenizer, context_length),
        remove_columns=train_val_split["train"].column_names,
        load_from_cache_file=False,
        batched=True,
    )

    tokenized_ds.save_to_disk(local_path)
    return tokenized_ds
