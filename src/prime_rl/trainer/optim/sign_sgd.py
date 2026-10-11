from typing import Callable

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


@triton.jit
def _sign_sgd_kernel(param_ptr, grad_ptr, numel, decay, step, DECAY: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    if DECAY:
        param = tl.fma(decay, param, param)
    # torch.sign: 0 for 0 and NaN.
    sign = (grad > 0).to(param.dtype) - (grad < 0).to(param.dtype)
    tl.store(param_ptr + offsets, tl.fma(step, sign, param), mask=mask)


@torch.no_grad()
def sign_sgd_update_(param: torch.Tensor, grad: torch.Tensor, lr: float, weight_decay: float) -> None:
    """`param -= lr * weight_decay * param`, then `param -= lr * sign(grad)`, in place.

    On CUDA this is one pass over `param` and `grad` instead of four, bit for bit with the torch ops.
    """
    param = param.to_local() if isinstance(param, DTensor) else param
    grad = grad.to_local() if isinstance(grad, DTensor) else grad
    if not (param.is_cuda and param.is_contiguous() and grad.is_contiguous() and grad.dtype == param.dtype):
        sign_grad = torch.sign(grad)
        if weight_decay > 0.0:
            param.add_(param, alpha=-lr * weight_decay)
        param.add_(sign_grad, alpha=-lr)
        return
    numel = param.numel()
    if numel == 0:
        return
    block = 4096
    _sign_sgd_kernel[(triton.cdiv(numel, block),)](
        param, grad, numel, -lr * weight_decay, -lr, DECAY=weight_decay > 0.0, BLOCK=block, num_warps=8
    )


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

                sign_sgd_update_(p, p.grad, group["lr"], group["weight_decay"])

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
        sign_sgd_update_(param, grad, group["lr"], group["weight_decay"])
        param.grad = None

    def grad_norm(self, grad_scale: float) -> torch.Tensor:
        """The step's gradient norm after the loss scaling `scale_gradients_` would have applied."""
        total = self._sum_of_squares.clone()
        dist.all_reduce(total)
        self._sum_of_squares.zero_()
        return total.sqrt() * grad_scale
