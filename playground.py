# %%
import random
from typing import Callable

import torch
from torch import nn
from torch._C.cpp.nn import Module
from tqdm import tqdm
from torch.optim import Optimizer

data = [[1, 2], [3, 5], [5, 8], [7, 11]]
x = torch.tensor(data)

# %%
in_dim = x.shape[1]
out_dim = 1
x_w_ones = torch.cat([torch.ones(len(x), 1), x], dim=1)
weights = torch.ones(in_dim + 1, out_dim)
z = x_w_ones @ weights
print(z)

# %%
# Training with manual weight updates
lr = 0.01
y = torch.tensor([[4], [11], [18], [25]]).float()

w = torch.rand(in_dim + 1, out_dim, requires_grad=True)
for i in tqdm(range(10_000)):
    z = x_w_ones @ w
    loss = torch.nn.functional.mse_loss(z, y)

    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
    w.grad.zero_()

    if i % 1000 == 0:
        print("-" * 80)
        print(w)
        print(x_w_ones @ w)
        print("-" * 80)

# %%
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# %%
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(cols * rows):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i + 1)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# %%
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# %%
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1)
    )
)

# %%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

# %%
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# %%
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# %%
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# %%
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# %%
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# %%
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10),
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# %%
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# %%
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# %%
import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = x @ w + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

# %%
z = x @ w + b
print(z.requires_grad)

with torch.no_grad():
    z = x @ w + b
print(z.requires_grad)

# %%
z = x @ w + b
print(z.requires_grad)
print(z.detach().requires_grad)

# %%
def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: Callable, optimizer: Optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_i, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_i % 100 == 0:
            loss, current = loss.item(), batch_i * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: Callable):
    num_batches = len(dataloader)
    test_loss, accuracy = 0, 0
    model.eval()
    with torch.no_grad():
        for batch_i, (X, y) in enumerate(dataloader):
            pred: torch.Tensor = model(X)
            test_loss += loss_fn(pred, y).item() / num_batches
            accuracy += (pred.argmax(1) == y).type(torch.float).mean().item() / num_batches
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")


model = NeuralNetwork()

learning_rate = 1e-3
batch_size = 64
epochs = 10

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# %%
import torch
import torchvision.models as models

model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model_weights.pth")

# %%
model = models.vgg16()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
print(model)

# %%
torch.save(model, "model.pth")
model = torch.load("model.pth", weights_only=False)
print(model)

# %%
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (FILE_PATH := PATH / FILENAME).exists():
    with open(FILE_PATH, "wb") as f:
        f.write(
            requests.get(URL + FILENAME).content
        )

# %%
import pickle
import gzip

with gzip.open(FILE_PATH, "rb") as f:
    mnist = pickle.load(f, encoding="latin-1")
    x_train, y_train = mnist[0]
    x_valid, y_valid = mnist[1]

# %%
%matplotlib inline

# %%
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(x_train[0].reshape(28, 28), cmap="gray")
plt.show()

# %%
import torch

x_train, y_train, x_valid, y_valid = (
    torch.tensor(data)
    for data in (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape

print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

# %%
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# %%
def log_softmax(x: torch.Tensor):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(x: torch.Tensor):
    return log_softmax(x @ weights + bias)

# %%
bs = 64

x_batch = x_train[:bs]
preds = model(x_batch)
print(preds[0], preds.shape)

# %%
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

# %%
y_batch = y_train[:bs]
print(loss_func(preds, y_batch))

# %%
def accuracy(out: torch.Tensor, y: torch.Tensor):
    preds = out.argmax(dim=1)
    return (preds == y).float().mean().item()

print(accuracy(preds, y_batch))

# %%
lr = 0.5
epochs = 2

for epoch in range(epochs):
    for i in range(math.ceil(n / bs)):
        # breakpoint()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= lr * weights.grad
            bias -= lr * bias.grad
            weights.grad.zero_()
            bias.grad.zero_()

# %%
print(loss_func(model(xb), yb), accuracy(model(xb), yb))

# %%
import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

# %%
from torch import nn

class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = MnistLogistic()

print(loss_func(model(xb), yb))

def fit():
    for epoch in range(epochs):
        for i in range(math.ceil(n / bs)):
            # breakpoint()
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                # Seems main value of subclassing nn.Module is encapsulation:
                #  not having to handle weight tensors individually
                for p in model.parameters():
                    p -= lr * p.grad
                model.zero_grad()

fit()
print(loss_func(model(xb), yb))

# %%
class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        # E.g. nn.Linear replaces parts of DAG with pre-made components
        self.linear = nn.Linear(784, 10)

    def forward(self, xb):
        return self.linear(xb)

model = MnistLogistic()
print(loss_func(model(xb), yb))
fit()
print(loss_func(model(xb), yb))

# %%
from torch import optim

def get_model_and_optimizer():
    model = MnistLogistic()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    return model, optimizer

model, opt = get_model_and_optimizer()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range(math.ceil(n / bs)):
        # breakpoint()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        # So we see optimizer replacing boilerplate code for freezing gradients, updating weights based on gradients
        #  and zeroing gradients.
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

# %%
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)

for epoch in range(epochs):
    for i in range(math.ceil(n / bs)):
        # In this case the reward for using TensorDataset is minimal, thinking main advantage comes in loading
        #  datasets not yet in memory (e.g. from URLs) and in combining with DataLoader
        xb, yb = train_ds[i * bs:(i + 1) * bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

# %%
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

for epoch in range(epochs):
    # So advantage is eliminating boilerplate around batching and shuffling of the data
    for xb, yb in train_dl:
        loss_func(model(xb), yb).backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

# %%
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
# Shuffling validation dataset has no effect on outcomes and just wastes time
# Since we're not keeping track of gradients for validation dataset the memory consumption is lower,
#  so we can afford larger batch sizes
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

model, opt = get_model_and_optimizer()

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_func(model(xb), yb).backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl) / len(valid_dl)
        print(epoch, valid_loss)

fit(5, model, loss_func, opt, train_dl, valid_dl)

# %%
class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


lr = 0.1
model = MnistCNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(5, model, loss_func, opt, train_dl, valid_dl)

# %%
class Lambda(nn.Module):
    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


# So Sequential lets us skip the whole subclassing of nn.Module
model = nn.Sequential(
    Lambda(lambda x: x.view(-1, 1, 28, 28)),
    nn.ReLU(),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1))
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(5, model, loss_func, opt, train_dl, valid_dl)

# %%
class WrappedDataLoader:
    def __init__(self, dl, func, device):
        self.dl = dl
        self.func = func
        self.device = device

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            b_on_device = (d.to(self.device) for d in b)
            yield self.func(*b_on_device)

def preprocess(x: torch.Tensor, y: torch.Tensor):
    return x.view(-1, 1, 28, 28), y

if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator().type
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

train_dl = WrappedDataLoader(
    DataLoader(train_ds, batch_size=bs, shuffle=True),
    preprocess,
    device,
)
valid_dl = WrappedDataLoader(
    DataLoader(valid_ds, batch_size=bs * 2),
    preprocess,
    device,
)

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)
model.to(device)
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(10, model, loss_func, opt, train_dl, valid_dl)


# %%
