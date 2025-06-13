from utilities.data_handler import load_preprocessed
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from pathlib import Path
from utilities.model_handler import train
from utilities.models import MyGPT


def run():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"{device=}")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # TODO: Use BooksCorpus dataset and context_length=512 (what was used in GPT paper)
    context_length = 128
    dataset, tokenized_ds = load_preprocessed(
        hf_path="wikitext", hf_name="wikitext-103-v1", tokenizer=tokenizer, context_length=context_length
    )

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
        checkpoint_path=Path(f"checkpoints/{int(time.time())}"),
    )


def worker(rank, n_gpus):
    print("=" * 80 + f"{rank=}\n{n_gpus=}")


def distributed_run():
    n_gpus = torch.cuda.device_count()
    mp.spawn(worker, nprocs=n_gpus, args=(n_gpus,))

if __name__ == "__main__":
    run()
    # distributed_run()
