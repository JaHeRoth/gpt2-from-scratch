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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # where rank 0 lives
    os.environ["MASTER_PORT"] = "29500"  # any free port
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def worker(rank, world_size, tokenizer, tokenized_ds):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Using hyperparams of GPT paper (although we use a different dataset)
    model = MyGPT(
        d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, vocab_size=tokenizer.vocab_size, device=device
    )
    if world_size > 1 and torch.cuda.is_available():
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
        device=device,  # TODO: Can we skip passing device, and bet on default device being correct in mp.spawn?
        train_batch_size=32,
        run_id=str(int(time.time())),
        # We disable these for all but rank 0, to avoid cluttering the output
        log_period=50 if rank == 0 else None,
        stream_period=250 if rank == 0 else None,
        eval_period=1000 if rank == 0 else None,
        checkpoint_period=100 if rank == 0 else None,
    )

    cleanup()


def run():
    tokenizer, tokenized_ds = prep()

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Running on {world_size} GPUs")
    else:
        world_size = 1
        print(f"Running on a single CPU")

    mp.spawn(worker, nprocs=world_size, args=(world_size, tokenizer, tokenized_ds))

if __name__ == "__main__":
    run()
