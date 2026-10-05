from typing import Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
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
            for p in group["params"]:
                if p.grad is None:
                    continue

                sign_grad = torch.sign(p.grad)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                p.add_(sign_grad, alpha=-group["lr"])

        return loss


class SignSGDInBackward:
    """`SignSGD` applied to each parameter as soon as its gradient is final, which then frees it.

    No step ever holds every gradient at once. The update is exact for one micro-batch per step:
    gradients are only rescaled (by the token count and clipping) after backward, and
    `sign(c * g) == sign(g)` for any `c > 0`. The gradient norm is accumulated on the way, for logging.
    """

    def __init__(self, optimizer: SignSGD, model: nn.Module):
        self.optimizer = optimizer
        self._group = {id(p): group for group in optimizer.param_groups for p in group["params"]}
        self._sum_of_squares = torch.zeros((), device=torch.cuda.current_device())
        self._handles = [
            p.register_post_accumulate_grad_hook(self._apply) for p in model.parameters() if id(p) in self._group
        ]

    @torch.no_grad()
    def _apply(self, param: torch.Tensor) -> None:
        grad = param.grad
        local = grad.to_local() if isinstance(grad, DTensor) else grad
        # Every rank adds its local sum of squares, so count each replica once.
        replicas = 1
        if isinstance(grad, DTensor):
            for size, placement in zip(grad.device_mesh.shape, grad.placements):
                replicas *= size if placement.is_replicate() else 1
        self._sum_of_squares += local.float().square().sum() / replicas
        group = self._group[id(param)]
        if group["weight_decay"] > 0.0:
            param.add_(param, alpha=-group["lr"] * group["weight_decay"])
        param.add_(torch.sign(grad), alpha=-group["lr"])
        param.grad = None

    def grad_norm(self, grad_scale: float) -> torch.Tensor:
        """The step's gradient norm after the loss scaling `scale_gradients_` would have applied."""
        total = self._sum_of_squares.clone()
        dist.all_reduce(total)
        self._sum_of_squares.zero_()
        return total.sqrt() * grad_scale
