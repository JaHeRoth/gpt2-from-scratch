# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
# Commented out because we yet again find mps to be drastically slower
# elif torch.backends.mps.is_available():
#     torch._dynamo.disable()  # https://github.com/pytorch/pytorch/issues/149184
#     device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"{device=}")

# %%
from transformers import AutoTokenizer

context_length = 2
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")  # TODO: Replace by simpler tokenizer?

# %%
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-v1")

# %%
for i in range(10):
    print(dataset["train"][i])

# %%
tokenizer([dataset["train"][i]["text"] for i in range(10)], padding=True, truncation=True, max_length=10)

# %%
max(len(tokens) for tokens in tokenizer([dataset["train"][i]["text"] for i in range(len(dataset["train"]))])["input_ids"])

# %%
len(dataset["train"])

# %%
dataset

# %%
outputs = tokenizer(
    dataset["train"][:2]["text"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

outputs


# %%
def tokenize(batch):
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

tokenized_ds = dataset.map(
    tokenize, batched=True, remove_columns=dataset["train"].column_names
)
tokenized_ds

# %%
tokenizer

# %%
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

# TODO: Start smaller
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# %%
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

# %%
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# %%
data_collator([tokenized_ds["train"][i] for i in range(2)])

# %%
out = data_collator([tokenized_ds["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

# %%
dataset

# %%
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="mygpt2",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    # evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=False,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    # tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
)

# %%
