# %%
import random

import torch
from torch import nn
from tqdm import tqdm

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
class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w = torch.rand(in_dim, out_dim)
        self.b = torch.ones(out_dim, 1)

    def forward(self, x):
        return x @ self.w + self.b


# %%
# Training with optimizer


# %%
