import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.):
        super(AdamW, self).__init__(
            params=params,
            defaults={"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
        )
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {
                    "first_moment": torch.zeros_like(p),
                    "second_moment": torch.zeros_like(p),
                }
        self.t = 0

    @torch.no_grad()
    def step(self, closure=None):
        self.t += 1

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, beta1, beta2, eps, weight_decay = (
                group["lr"], group["betas"][0], group["betas"][1], group["eps"], group["weight_decay"]
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["first_moment"] = (
                    beta1 * self.state[p]["first_moment"]
                    + (1 - beta1) * p.grad
                )
                self.state[p]["second_moment"] = (
                    beta2 * self.state[p]["second_moment"]
                    + (1 - beta2) * (p.grad ** 2)
                )
                bias_corrected_first_moment = self.state[p]["first_moment"] / (1 - beta1 ** self.t)
                bias_corrected_second_moment = self.state[p]["second_moment"] / (1 - beta2 ** self.t)
                p.sub_(
                    lr * (
                        bias_corrected_first_moment / (bias_corrected_second_moment.sqrt() + eps)
                        + weight_decay * p
                    )
                )

        return loss
