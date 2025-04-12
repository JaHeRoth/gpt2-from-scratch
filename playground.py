# %%
import torch
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
