"""Row-sparse backbone backward (stop-gradient through context tokens).

The forward pass is unchanged — every token still runs through every layer so
that context K/V exist for the kept queries to attend to. The backward pass is
made row-sparse: gradients only propagate through the rows some loss component
reads (the same ``keep_mask`` the LM head skipping uses), which cuts the
backbone's linear-layer backward GEMMs proportionally to the kept fraction.

Two pieces implement the semantics of a per-layer
``torch.where(keep, h, h.detach())`` without paying full-size backward GEMMs:

- ``mask_grad``: identity forward; backward zeroes context-row gradients.
  Placed at every decoder-layer input, it stops gradients from propagating
  into context hidden states (and from there into earlier layers).
- ``row_sparse_linear``: full-row forward (all rows are consumed downstream),
  backward compacted to kept rows: ``dW = dY[keep]^T @ X[keep]`` and
  ``dX[keep] = dY[keep] @ W`` (zeros elsewhere). Only valid for projections
  whose upstream gradient is exactly zero on context rows under the mask-grad
  cut: q/o and the dense MLP. The k/v projections must keep a full backward
  because ``dK``/``dV`` on context rows are nonzero (kept queries attend to
  them) and feed the k/v weight gradients — that is the signal that lets the
  model keep learning how to read context.

This changes the training gradient (the loss no longer shapes how context
tokens are represented); it is NOT a pure optimization like the LM head
skipping.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def backbone_keep_index(keep_mask: Tensor) -> Tensor | None:
    """Flattened indices of rows that keep gradients. None means keep all.

    An all-masked micro batch keeps one dummy row so every backward GEMM stays
    non-empty and weight grads exist on all ranks (mirrors the LM head skip).
    """
    index = keep_mask.reshape(-1).nonzero(as_tuple=True)[0]
    if index.numel() == keep_mask.numel():
        return None
    if index.numel() == 0:
        return torch.zeros(1, dtype=torch.long, device=keep_mask.device)
    return index


class _MaskGradFn(torch.autograd.Function):
    """Identity forward; backward multiplies the gradient by the keep mask."""

    @staticmethod
    def forward(ctx, x: Tensor, keep_mask: Tensor) -> Tensor:
        ctx.save_for_backward(keep_mask)
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (keep_mask,) = ctx.saved_tensors
        return grad_output * keep_mask.unsqueeze(-1), None


def mask_grad(x: Tensor, keep_mask: Tensor | None) -> Tensor:
    """Stop gradients on rows where ``keep_mask`` is False. Values unchanged."""
    if keep_mask is None or not torch.is_grad_enabled() or not x.requires_grad:
        return x
    return _MaskGradFn.apply(x, keep_mask)


class _RowSparseLinearFn(torch.autograd.Function):
    """Linear with a full-row forward and a kept-rows-only backward."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, keep_index: Tensor) -> Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        out = x_2d @ weight.t()
        ctx.save_for_backward(x_2d.index_select(0, keep_index), weight, keep_index)
        ctx.x_shape = x.shape
        return out.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_keep, weight, keep_index = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_keep = grad_2d.index_select(0, keep_index)

        grad_x = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_2d.new_zeros(grad_2d.shape[0], weight.shape[1])
            grad_x.index_copy_(0, keep_index, grad_keep @ weight)
            grad_x = grad_x.reshape(ctx.x_shape)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_keep.t() @ x_keep
        return grad_x, grad_weight, None


def row_sparse_linear(linear: nn.Linear, x: Tensor, keep_index: Tensor | None) -> Tensor:
    """Apply ``linear`` with a backward restricted to ``keep_index`` rows.

    Falls back to the plain linear when there is nothing to skip, when grad is
    disabled (eval), or when the module is not a plain bias-free ``nn.Linear``
    (LoRA / quantized wrappers own their backward).
    """
    if keep_index is None or not torch.is_grad_enabled() or type(linear) is not nn.Linear or linear.bias is not None:
        return linear(x)
    return _RowSparseLinearFn.apply(x, linear.weight, keep_index)
