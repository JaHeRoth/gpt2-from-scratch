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
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sshleifer/tiny-gpt2"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

inputs = tokenizer(["Hi, my name is what my name is "], return_tensors="pt")
outputs = model.generate(**inputs, max_length=30)
for s in tokenizer.batch_decode(outputs):
    print(s)

# %%
from transformers import pipeline

p = pipeline("text-generation", model=model_name, device_map=device)
p("Hi, my name ", max_length=50, truncation=True)

# %%
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

model_name = "sshleifer/tiny-gpt2"
dataset_name = "rotten_tomatoes"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset(dataset_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
tokenized_ds = dataset.map(tokenize_dataset, batched=True)

training_args = TrainingArguments(
    output_dir="tiny-gpt2-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# %%
trainer.train()

# %%
