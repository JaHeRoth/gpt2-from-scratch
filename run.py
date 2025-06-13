from utilities.data_handler import load_preprocessed
from transformers import AutoTokenizer
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from pathlib import Path
from utilities.model_handler import train
from utilities.models import MyGPT


def prep():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # TODO: Use BooksCorpus dataset and context_length=512 (what was used in GPT paper)
    context_length = 128
    _, tokenized_ds = load_preprocessed(
        hf_path="wikitext", hf_name="wikitext-103-v1", tokenizer=tokenizer, context_length=context_length
    )

    return tokenizer, tokenized_ds


def run():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"{device=}")

    tokenizer, tokenized_ds = prep()

    # Using hyperparams of GPT paper (although we use a different dataset)
    model = MyGPT(d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, vocab_size=tokenizer.vocab_size,
                  device=device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        betas=(0.9, 0.98),
        eps=1e-9,
        lr=2.5e-4,
    )
    # TODO: DistributedDataParallel instead of DataParallel
    train(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        tokenized_train_ds=tokenized_ds["train"],
        tokenized_eval_ds=tokenized_ds["validation"],
        device=device,
        # TODO: Consider renaming from checkpoint, to avoid confusion with torch.util.checkpoint?
        checkpoint_path=Path(f"checkpoints/{int(time.time())}"),
    )


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # where rank 0 lives
    os.environ["MASTER_PORT"] = "29500"  # any free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def worker(rank, world_size, tokenizer, tokenized_ds):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    # Using hyperparams of GPT paper (although we use a different dataset)
    model = MyGPT(
        d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, vocab_size=tokenizer.vocab_size, device=device
    )
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        betas=(0.9, 0.98),
        eps=1e-9,
        lr=2.5e-4,
    )
    train(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        tokenized_train_ds=tokenized_ds["train"],
        tokenized_eval_ds=tokenized_ds["validation"],
        device=device,
        train_batch_size=32,
        # We disable these for all but rank 0, to avoid cluttering the output
        checkpoint_path=Path(f"checkpoints/{int(time.time())}") if rank == 0 else None,
        log_period=50 if rank == 0 else None,
        stream_period=250 if rank == 0 else None,
        eval_period=1000 if rank == 0 else None,
    )


def distributed_run():
    tokenizer, tokenized_ds = prep()
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs")
    mp.spawn(worker, nprocs=n_gpus, args=(n_gpus, tokenizer, tokenized_ds))

if __name__ == "__main__":
    # TODO: Unify these somehow
    # run()
    distributed_run()
