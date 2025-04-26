# %%
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    torch._dynamo.disable()
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.set_default_device(device)
print(f"Using device: {torch.get_default_device()}")

# %%
from pathlib import Path
import requests
import zipfile
import os

DATA_PATH = Path("data")
PATH = DATA_PATH / "nlp_from_scratch"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://download.pytorch.org/tutorial/"
FILENAME = "data.zip"

# %%
if not PATH.exists():
    FILE_PATH = PATH / FILENAME

    with open(FILE_PATH, "wb") as f:
        f.write(
            requests.get(URL + FILENAME).content
        )

    with zipfile.ZipFile(FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(PATH)

    os.remove(FILE_PATH)
    os.listdir(PATH)

# %%
with open(PATH / "data" / "names" / "Arabic.txt") as f:
    print(f"First line: {f.readline()}")
    print(f"Second line: {f.readline()}")
    print(f"Third line: {f.readline()}")

# %%
import string
import unicodedata

allowed_chars: str = string.ascii_letters + " .,;'"
n_letters = len(allowed_chars)

def unicode_to_ascii(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_chars
    )


print(f"converting 'Ślusàrski!' to {unicode_to_ascii('Ślusàrski!')}")

# %%
def letter_to_index(letter) -> int | None:
    try:
        return allowed_chars.index(letter)
    except ValueError:
        return None


def line_to_tensor(line: str) -> torch.Tensor:
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        if (index := letter_to_index(letter)) is not None:
            tensor[i][0][index] = 1
    return tensor


print(f"The letter 'a' becomes {line_to_tensor('a')}")
print(f"The name 'Ahn' becomes {line_to_tensor('Ahn')}")

# %%
from io import open
import glob
import os
import time

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class NamesDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()  # Does nothing, but stops the complaining
        self.data_dir = data_dir
        self.load_time = time.localtime
        labels_set = set()

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        filenames = glob.glob(os.path.join(data_dir, '*.txt'))
        for filename in tqdm(filenames):
            label = os.path.splitext(os.path.basename(filename))[0]
            labels_set.add(label)
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    self.data.append(name)
                    self.data_tensors.append(line_to_tensor(name))
                    self.labels.append(label)

        self.labels_uniq = list(labels_set)
        for idx, label in enumerate(self.labels):
            self.labels_tensors.append(
                torch.tensor([self.labels_uniq.index(label)])
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.labels_tensors[idx],
            self.data_tensors[idx],
            self.labels[idx],
            self.data[idx],
        )


alldata = NamesDataset(PATH / "data" / "names")
print(f"loaded {len(alldata)} items of data")
print(f"example = {alldata[0]}")

# %%
generator = torch.Generator(device=device).manual_seed(42)
train_set, test_set = torch.utils.data.random_split(
    dataset=alldata, lengths=[0.8, 0.2], generator=generator
)

print(f"{len(train_set)=}; {len(test_set)=}")

# %%
import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        _, hidden = self.rnn(line_tensor)
        # Seems `hidden.size=[1, batch_size, hidden_size]`
        output = self.h2o(hidden[0])
        return self.softmax(output)


n_hidden = 128
rnn = CharRNN(n_letters, n_hidden, len(alldata.labels_uniq))
print(rnn)

# %%
def label_from_output(output: torch.Tensor, output_labels):
    _, top_idx = output.topk(1)
    label_idx = top_idx.item()
    return output_labels[label_idx], label_idx


input = line_to_tensor("Albert")
output = rnn(input)
print(f"{output=}")
print(f"{label_from_output(output, alldata.labels_uniq)=}")

# %%
import random
import numpy as np

def train(
    rnn,
    training_data,
    n_epochs=10,
    batch_size=64,
    report_every=50,
    lr=0.2,
    criterion=nn.NLLLoss(),
):
    overall_loss = 0
    all_losses = []

    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)

    start_time = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(n_epochs):
        # Cannot use DataLoader since names have different length. I hate this
        #  though, since we get no inter-name parallelism in the forward pass
        shuffled_indices = list(range(len(training_data)))
        random.shuffle(shuffled_indices)
        batches = np.array_split(
            shuffled_indices, len(shuffled_indices) // batch_size
        )

        for batch in batches:
            batch_loss = 0
            for idx in batch:
                label_tensor, data_tensor, _, _ = training_data[idx]
                output = rnn(data_tensor)
                # NLLLoss takes as input an array of scores and a scalar index
                #  of the true label, bts plumbing these together correctly
                batch_loss += criterion(output, label_tensor) / len(batch)

            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            overall_loss += batch_loss.item() / len(batches)

        all_losses.append(overall_loss)
        if iter % report_every == 0 or iter == n_epochs - 1:
            print(f"{iter:<{len(str(n_epochs))}} ({iter / n_epochs:.0%}): \t average batch loss = {all_losses[-1]}")
            overall_loss = 0

    print(f"Training finished in {time.time() - start_time:.2f} seconds")
    return all_losses


all_losses = train(rnn, train_set, n_epochs=27, lr=0.15, report_every=5)

# %%
import matplotlib.pyplot as plt

plt.plot(all_losses)
plt.xlabel("Epoch")
plt.ylabel("Average loss per observation")
plt.show()

# %%
import matplotlib.ticker as ticker

def evaluate(rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval()
    with torch.no_grad():
        for observation in testing_data:
            label_tensor, data_tensor, label, name = observation
            output = rnn(data_tensor)
            pred, pred_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][pred_i] += 1

    for i in range(len(classes)):
        if (s := confusion[i].sum()) > 0:
            confusion[i] = confusion[i] / s

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


evaluate(rnn, test_set, classes=alldata.labels_uniq)

# %%
