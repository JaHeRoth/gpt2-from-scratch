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
from utilities.models import TransformerEncoderGPT


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
    model = TransformerEncoderGPT(
        d_model=768, nhead=12, num_layers=12, dim_feedforward=3072, vocab_size=tokenizer.vocab_size, device=device
    )
    # We choose to always use DDP, to avoid downstream if-statements for the rare case of single-device training.
    model = DDP(
        model,
        device_ids=[rank] if torch.cuda.is_available() else None,
    )

    # TODO: Replace with AdamW with weight_decay on all parameters but bias and LayerNorm gain ones, as in GPT paper
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
        train_batch_size=128,
        run_id=str(int(time.time())),
        # We disable these for all but rank 0, to avoid cluttering the output
        make_outputs=rank == 0,
        log_period=50,
        stream_period=250,
        eval_period=500,
        checkpoint_period=100,
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
