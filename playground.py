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
class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w = torch.rand(in_dim, out_dim, requires_grad=True)
        self.b = torch.ones(out_dim, 1, requires_grad=True)

    def forward(self, x):
        return x @ self.w + self.b


# %%
# Training with optimizer


# %%
