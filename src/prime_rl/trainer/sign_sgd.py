from typing import Callable

import torch
from torch.optim import Optimizer


class SignSGD(Optimizer):
    """Sign-based SGD optimizer with minimal memory footprint.

    This optimizer uses the sign of gradients instead of storing momentum and variance,
    making it equivalent to AdamW with beta1=0 and beta2=0 (resetting optimizer state each step).

    Mathematical equivalence:
        AdamW: W = W - lr * m_t / sqrt(v_t + eps)
        With beta1=0, beta2=0: m_t = g_t, v_t = g_t^2
        Simplified: W = W - lr * g_t / sqrt(g_t^2 + eps)
        Ignoring eps: W = W - lr * sign(g_t)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Batch per (device, dtype): foreach ops need homogeneous lists,
            # and a few fused launches (vs thousands of per-param kernels)
            # keep the step's kernel tail from overlapping the grad frees and
            # collectives that follow it
            buckets: dict[tuple, list] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                buckets.setdefault((p.device, p.dtype), []).append(p)

            for params in buckets.values():
                grads = [p.grad for p in params]
                if group["weight_decay"] > 0.0:
                    torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
                signs = torch._foreach_sign(grads)
                torch._foreach_add_(params, signs, alpha=-group["lr"])

        return loss
