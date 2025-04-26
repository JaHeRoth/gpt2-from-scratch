# %%
import torch

# Found this to be much faster than mps for Part 1
device = torch.device("cpu")

# %%
from io import open
import glob
import os
import unicodedata
import string

all_letters: str = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker

def find_files(path):
    return glob.glob(path)

def unicode_to_ascii(s) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def read_lines(filename):
    with open(filename, encoding='utf-8') as f:
        return [unicode_to_ascii(line.strip()) for line in f]

category_lines = {}
all_categories = []
for filename in find_files('data/nlp_from_scratch/data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    category_lines[category] = read_lines(filename)

n_categories = len(all_categories)

if n_categories == 0:
    raise ValueError("Failed to extract any categories from the data.")

print('# categories:', n_categories, all_categories)
print(unicode_to_ascii("O'Néàl"))

# %%
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.o2o = nn.Linear(output_size + hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, category_oh, input, hidden):
        combined = torch.cat((category_oh, input, hidden), 1)
        hidden_output = self.i2o(combined)
        hidden = self.i2h(combined)
        out_combined = torch.cat((hidden_output, hidden), 1)
        raw_logit = self.o2o(out_combined)
        fuzzy_logit = self.dropout(raw_logit)
        log_probit = self.log_softmax(fuzzy_logit)
        return log_probit, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# %%
import random

def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    return category, line

def ohe_category(cat):
    idx = all_categories.index(cat)
    oh = torch.zeros(1, n_categories)
    oh[0][idx] = 1
    return oh

def ohe_line(line):
    oh = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        idx = all_letters.find(letter)
        oh[i][0][idx] = 1
    return oh

def line_to_targets(line):
    next_token_indices = [
        all_letters.find(letter) for letter in line[1:]
    ]
    next_token_indices.append(n_letters - 1)  # EOS
    return torch.LongTensor(next_token_indices)

def random_training_example():
    category, line = random_training_pair()
    category_oh = ohe_category(category)
    line_oh = ohe_line(line)
    targets = line_to_targets(line)
    return category_oh, line_oh, targets


random_training_example()

# %%
criterion = nn.NLLLoss()
lr = 0.001

def train(rnn, category_oh, line_oh, line_targets):
    line_targets = line_targets.unsqueeze(-1)
    hidden = rnn.init_hidden()
    opt = torch.optim.SGD(rnn.parameters(), lr=lr)
    opt.zero_grad()

    rnn.train()
    loss = torch.Tensor([0])
    for char_oh, char_target in zip(line_oh, line_targets):
        output, hidden = rnn(category_oh, char_oh, hidden)
        loss += criterion(output, char_target) / len(line_targets)

    loss.backward()
    opt.step()
    opt.zero_grad()

    return output, loss.item()

# %%
import time
import math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# %%
rnn = RNN(n_letters, 128, n_letters)

n_iters = 100_000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0

start = time.time()
for iter in range(n_iters):
    _, loss = train(
        rnn,
        *random_training_example()
    )
    total_loss += loss

    if iter % print_every == 0 or iter == n_iters - 1:
        print(
            f"{time_since(start)} ({iter} {iter / n_iters * 100:.0f}%) {loss:.4f}"
        )
    if (iter + 1) % plot_every == 0 or iter == n_iters - 1:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

# %%
import matplotlib.pyplot as plt

plt.plot(all_losses)
plt.xlabel(f'iter / {plot_every}')
plt.ylabel('NLLLoss')
plt.show()

# %%
def sample_name(rnn, category, start_char, max_length=20):
    with torch.no_grad():
        category_oh = ohe_category(category)
        out_chars = [start_char]
        hidden = rnn.init_hidden()
        for _ in range(max_length - 1):
            start_char_oh = ohe_line(out_chars[-1])
            log_probits, hidden = rnn(category_oh, start_char_oh[0], hidden)
            _, out_char_idx = log_probits.topk(1)

            if out_char_idx.item() == n_letters - 1:
                break
            else:
                out_chars.append(
                    all_letters[out_char_idx.item()]
                )

    return ''.join(out_chars)

def samples(rnn, category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample_name(rnn, category, start_letter))


print('-' * 20 + 'Italian' + '-' * 20)
samples(rnn, 'Italian', 'ITA')
print('-' * 20 + 'Greek' + '-' * 20)
samples(rnn, 'Greek', 'GRE')
print('-' * 20 + 'Russian' + '-' * 20)
samples(rnn, 'Russian', 'RUS')
print('-' * 20 + 'Arabic' + '-' * 20)
samples(rnn, 'Arabic', 'ARA')

# %%
