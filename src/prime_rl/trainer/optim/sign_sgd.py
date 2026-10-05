from typing import Callable

import torch
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


@triton.jit
def _sign_sgd_kernel(param_ptr, grad_ptr, numel, lr, lr_weight_decay, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    param = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
    # Like torch.sign on CUDA: zero and NaN both map to 0.
    sign = tl.where(grad > 0, 1.0, tl.where(grad < 0, -1.0, 0.0))
    # `param * (1 - lr * wd)` would round the tiny decay away in fp32; subtract it instead.
    param = param - lr_weight_decay * param - lr * sign
    tl.store(param_ptr + offsets, param.to(param_ptr.dtype.element_ty), mask=mask)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class SignSGD(Optimizer):
    """Sign-based SGD optimizer with minimal memory footprint.

    This optimizer uses the sign of gradients instead of storing momentum and variance,
    making it equivalent to AdamW with beta1=0 and beta2=0 (resetting optimizer state each step).

    Mathematical equivalence:
        AdamW: W = W - lr * m_t / sqrt(v_t + eps)
        With beta1=0, beta2=0: m_t = g_t, v_t = g_t^2
        Simplified: W = W - lr * g_t / sqrt(g_t^2 + eps)
        Ignoring eps: W = W - lr * sign(g_t)

    On CUDA the decay and update run as one fused pass over each parameter, which matters when a
    rank holds billions of expert parameters.
    """

    # Rescaling every gradient by a positive factor leaves the update unchanged, so gradient
    # clipping only needs the norm, not the rescale.
    scale_invariant = True

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
            lr, weight_decay = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                param, grad = _local(p), _local(p.grad)
                if param.is_cuda and param.is_contiguous() and grad.is_contiguous():
                    numel = param.numel()
                    if numel:
                        BLOCK = 4096
                        _sign_sgd_kernel[(triton.cdiv(numel, BLOCK),)](
                            param, grad, numel, lr, lr * weight_decay, BLOCK=BLOCK
                        )
                    continue

                sign_grad = torch.sign(p.grad)

                if weight_decay > 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                p.add_(sign_grad, alpha=-lr)

        return loss
