from __future__ import annotations

import torch
from torch import Tensor


class _MatmulToFloat32Fn(torch.autograd.Function):
    """`lhs @ rhs` accumulated into float32, with a hand-written backward.

    `aten::mm.dtype` has no registered derivative, so the widened matmul cannot be used in an
    autograd-tracked forward without this wrapper.
    """

    @staticmethod
    def forward(ctx, lhs: Tensor, rhs: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(lhs, rhs)
        if lhs.is_cuda and lhs.dtype in (torch.float16, torch.bfloat16):
            return torch.mm(lhs, rhs, out_dtype=torch.float32)
        return torch.mm(lhs, rhs).to(torch.float32)

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor | None, Tensor | None]:
        lhs, rhs = ctx.saved_tensors
        needs_lhs, needs_rhs = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        grad_lhs = torch.mm(grad_out.to(rhs.dtype), rhs.t()) if needs_lhs else None
        grad_rhs = torch.mm(lhs.t(), grad_out.to(lhs.dtype)) if needs_rhs else None
        return grad_lhs, grad_rhs


def matmul_to_float32(lhs: Tensor, rhs: Tensor) -> Tensor:
    """Matmul two 2D tensors, accumulating into float32 instead of rounding to a half-precision input dtype."""
    return _MatmulToFloat32Fn.apply(lhs, rhs)


__all__ = ["matmul_to_float32"]
