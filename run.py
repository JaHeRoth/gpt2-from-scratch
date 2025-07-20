from datetime import timedelta

from utilities.data_handler import load_preprocessed
from transformers import AutoTokenizer
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import time
from pathlib import Path
from utilities.model_handler import train
from utilities.models import ParametersGPT2
from utilities import optimizers

context_length = 512

def prep():
    tokenizer_path = Path("outputs/tokenizer")
    if not tokenizer_path.exists():
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        AutoTokenizer.from_pretrained(
            "openai-community/gpt2",
            pad_token="<|pad|>",
            unk_token="<unk>",  # Because it appears often in the dataset
        ).save_pretrained(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


    _, tokenized_ds = load_preprocessed(
        hf_path="wikitext", hf_name="wikitext-103-v1", tokenizer=tokenizer, context_length=context_length
    )

    return tokenizer, tokenized_ds


def setup(rank, world_size, failed_attempts=0):
    master_port = 29500 + failed_attempts
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # where rank 0 lives
    os.environ["MASTER_PORT"] = str(master_port)  # any free port
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except RuntimeError as err:
        if failed_attempts == 100:
            print(f"Failed 100 times in a row to setup, so giving up")
            raise err
        print(f"Failed to init process group with master port {master_port}, "
              f"so retrying with master port {master_port + 1}. Encountered error: {err}")
        setup(rank, world_size, failed_attempts + 1)


def worker(rank, world_size, tokenizer, tokenized_ds):
    setup(rank, world_size)
    try:
        torch.manual_seed(42)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Hyperparameters are inspired by GPT and GPT-3 papers
        model = ParametersGPT2(
            d_model=768,
            nhead=12,
            num_layers=12,
            dim_feedforward=3072,
            vocab_size=len(tokenizer),
            context_length=context_length,
            eos_token_id=tokenizer.eos_token_id,
            dropout_p=0.1,
            device=device,
        )
        model.compile()
        # We choose to always use DDP, to avoid downstream if-statements for the rare case of single-device training.
        model = DistributedDataParallel(
            model,
            device_ids=[rank] if torch.cuda.is_available() else None,
        )

        decaying_params = [param for param in model.parameters() if len(param.shape) >= 2]
        non_decaying_params = [param for param in model.parameters() if len(param.shape) < 2]
        optimizer = optimizers.AdamW(
            params=[
                {"params": decaying_params, "weight_decay": 0.01},
                {"params": non_decaying_params, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.98),
            eps=1e-9,
            lr=2.5e-4,
        )

        tokens_per_update = 524288
        train_batch_size = 128
        gradient_accumulation_steps = max(1, tokens_per_update // world_size // train_batch_size // context_length)
        if rank == 0:
            print(f"{gradient_accumulation_steps=}")
        train(
            model=model,
            optimizer=optimizer,
            tokenizer=tokenizer,
            tokenized_train_ds=tokenized_ds["train"],
            tokenized_eval_ds=tokenized_ds["validation"],
            device=device,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=500,  # Non-standard
            num_epochs=25,  # Non-standard
            run_id=str(int(time.time())),
            # We disable these for all but rank 0, to avoid cluttering the output
            make_outputs=rank == 0,
            stream_prompt=f"{tokenizer.eos_token}She first",
            log_period=5,
            stream_period=250,
            eval_period=500,
            checkpoint_period=100,
        )
    finally:
        print(f"Worker {rank}/{world_size} waiting to clean up after itself.", flush=True)
        dist.barrier()
        print(f"Worker {rank}/{world_size} cleaning up after itself.", flush=True)
        dist.destroy_process_group()
        torch.cuda.empty_cache()


def run():
    tokenizer, tokenized_ds = prep()

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Running on {world_size} GPUs")
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # So that all GPUs exit upon program exit
        mp.spawn(worker, nprocs=world_size, args=(world_size, tokenizer, tokenized_ds))
    else:
        print(f"Running on a single CPU")
        # Not running through mp.spawn makes debugging easier
        worker(rank=0, world_size=1, tokenizer=tokenizer, tokenized_ds=tokenized_ds)


if __name__ == "__main__":
    run()
