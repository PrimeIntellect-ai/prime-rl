"""Custom ops of the fused per-rank dispatch with their fake implementations and autograd: `sum_slots` and its
transpose `scatter_slots` over a `SlotMap`. Each one's backward is the other one, with the routing weights' gradient
from the same pass. Off CUDA the ops run as plain PyTorch."""

from dataclasses import dataclass

import torch
from torch import Tensor

from prime_rl.trainer.distributed.per_rank import kernels


@dataclass(frozen=True)
class SlotMap:
    """Pairs the ``W`` slots of each of ``S`` segments with the rows of a ``[num_rows, C]`` tensor: slot (s, w) owns
    row ``index[s, w]`` (int32, ``[S, W]``), or none if it is -1. Every row belongs to one slot, except
    ``padding_rows``, which belong to none (int32, -1 entries are ignored)."""

    index: Tensor
    num_rows: int
    padding_rows: Tensor | None = None


def _slot_rows(src: Tensor, index: Tensor) -> Tensor:
    """``src[index]`` as fp32 ``[S, W, C]``, zero for slots without a row."""
    padded = torch.cat((src.float(), src.new_zeros((1, src.shape[1]), dtype=torch.float32)))
    return padded[torch.where(index >= 0, index.long(), src.shape[0])]


def _sum(src: Tensor, index: Tensor, weights: Tensor | None, other: Tensor | None) -> tuple[Tensor, Tensor]:
    """``[S, C]`` sums over the slots and, with ``other`` (``[S, C]``), the ``[S, W]`` dot products of every slot's row
    with its segment's row of ``other``."""
    if src.is_cuda:
        return kernels.sum_slots(src, index, weights, other)
    rows = _slot_rows(src, index)
    dots = rows.new_empty(0) if other is None else (rows * other.float().unsqueeze(1)).sum(dim=-1)
    if weights is not None:
        rows = rows * weights.float().unsqueeze(-1)
    return rows.sum(dim=1).to(src.dtype), dots


def _scatter(
    src: Tensor, index: Tensor, num_rows: int, weights: Tensor | None, other: Tensor | None, padding_rows: Tensor | None
) -> tuple[Tensor, Tensor]:
    """The ``[num_rows, C]`` rows the slots write and, with ``other`` (``[num_rows, C]``), the ``[S, W]`` dot products
    of every segment's row with its slots' rows of ``other``."""
    if src.is_cuda:
        return kernels.scatter_slots(src, index, num_rows, weights, other, padding_rows)
    valid = index >= 0
    values = src.unsqueeze(1).expand(-1, index.shape[1], -1)
    if weights is not None:
        values = (values.float() * weights.float().unsqueeze(-1)).to(src.dtype)
    out = src.new_zeros((num_rows, src.shape[1]))
    out[index[valid].long()] = values[valid]
    dots = src.new_empty(0, dtype=torch.float32)
    if other is not None:
        dots = (_slot_rows(other, index) * src.float().unsqueeze(1)).sum(dim=-1)
    return out, dots


def _dots_like(index: Tensor, wanted: bool) -> Tensor:
    return index.new_empty(index.shape if wanted else (0,), dtype=torch.float32)


@torch.library.custom_op("prime_rl::sum_slots", mutates_args=())
def _sum_slots(src: Tensor, index: Tensor, padding_rows: Tensor | None, weights: Tensor | None) -> Tensor:
    return _sum(src, index, weights, None)[0]


@_sum_slots.register_fake
def _sum_slots_fake(src: Tensor, index: Tensor, padding_rows: Tensor | None, weights: Tensor | None) -> Tensor:
    return src.new_empty((index.shape[0], src.shape[1]))


@torch.library.custom_op("prime_rl::sum_slots_backward", mutates_args=())
def _sum_slots_backward(
    grad_out: Tensor,
    index: Tensor,
    num_rows: int,
    padding_rows: Tensor | None,
    weights: Tensor | None,
    src: Tensor | None,
) -> tuple[Tensor, Tensor]:
    return _scatter(grad_out, index, num_rows, weights, src, padding_rows)


@_sum_slots_backward.register_fake
def _sum_slots_backward_fake(
    grad_out: Tensor,
    index: Tensor,
    num_rows: int,
    padding_rows: Tensor | None,
    weights: Tensor | None,
    src: Tensor | None,
) -> tuple[Tensor, Tensor]:
    return grad_out.new_empty((num_rows, grad_out.shape[1])), _dots_like(index, src is not None)


def _sum_slots_setup_context(ctx, inputs, output) -> None:
    src, index, padding_rows, weights = inputs
    weights_need_grad = weights is not None and ctx.needs_input_grad[3]
    ctx.num_rows = src.shape[0]
    ctx.save_for_backward(index, padding_rows, weights, src if weights_need_grad else None)


def _sum_slots_autograd_backward(ctx, grad_out: Tensor):
    index, padding_rows, weights, src = ctx.saved_tensors
    grad_src, grad_weights = _sum_slots_backward(grad_out.contiguous(), index, ctx.num_rows, padding_rows, weights, src)
    return grad_src, None, None, grad_weights if src is not None else None


_sum_slots.register_autograd(_sum_slots_autograd_backward, setup_context=_sum_slots_setup_context)


@torch.library.custom_op("prime_rl::scatter_slots", mutates_args=())
def _scatter_slots(
    src: Tensor, index: Tensor, num_rows: int, padding_rows: Tensor | None, weights: Tensor | None
) -> Tensor:
    return _scatter(src, index, num_rows, weights, None, padding_rows)[0]


@_scatter_slots.register_fake
def _scatter_slots_fake(
    src: Tensor, index: Tensor, num_rows: int, padding_rows: Tensor | None, weights: Tensor | None
) -> Tensor:
    return src.new_empty((num_rows, src.shape[1]))


@torch.library.custom_op("prime_rl::scatter_slots_backward", mutates_args=())
def _scatter_slots_backward(
    grad_out: Tensor, index: Tensor, weights: Tensor | None, src: Tensor | None
) -> tuple[Tensor, Tensor]:
    return _sum(grad_out, index, weights, src)


@_scatter_slots_backward.register_fake
def _scatter_slots_backward_fake(
    grad_out: Tensor, index: Tensor, weights: Tensor | None, src: Tensor | None
) -> tuple[Tensor, Tensor]:
    return grad_out.new_empty((index.shape[0], grad_out.shape[1])), _dots_like(index, src is not None)


def _scatter_slots_setup_context(ctx, inputs, output) -> None:
    src, index, _, _, weights = inputs
    weights_need_grad = weights is not None and ctx.needs_input_grad[4]
    ctx.save_for_backward(index, weights, src if weights_need_grad else None)


def _scatter_slots_autograd_backward(ctx, grad_out: Tensor):
    index, weights, src = ctx.saved_tensors
    grad_src, grad_weights = _scatter_slots_backward(grad_out.contiguous(), index, weights, src)
    return grad_src, None, None, None, grad_weights if src is not None else None


_scatter_slots.register_autograd(_scatter_slots_autograd_backward, setup_context=_scatter_slots_setup_context)


def sum_slots(src: Tensor, slots: SlotMap, weights: Tensor | None = None) -> Tensor:
    """``out[s] = sum(weights[s, w] * src[slots.index[s, w]] for the slots w that own a row)``, ``[S, C]`` from the
    ``[slots.num_rows, C]`` rows ``src``, accumulated in fp32. ``weights`` (``[S, W]``) are read by slot."""
    return _sum_slots(src.contiguous(), slots.index, slots.padding_rows, weights)


def scatter_slots(src: Tensor, slots: SlotMap, weights: Tensor | None = None) -> Tensor:
    """``out[slots.index[s, w]] = weights[s, w] * src[s]`` for the slots that own a row, ``[slots.num_rows, C]`` from
    the ``[S, C]`` rows ``src``; the padding rows are zero. ``weights`` (``[S, W]``) are read by slot."""
    return _scatter_slots(src.contiguous(), slots.index, slots.num_rows, slots.padding_rows, weights)
