"""Triton kernels of the fused per-rank dispatch and their launches. A slot map pairs the ``W`` slots of each of ``S``
segments with rows of a ``[num_rows, C]`` tensor: slot (s, w) owns row ``index[s, w]``, or none if it is -1. Both
kernels run one program per segment and column block, walk the segment's slots and skip the slots without a row: the
sum reads the slots' rows (accumulating in fp32), the scatter writes them. With dot products (the routing weights'
gradient: each slot's row dotted with its segment's row), a program covers a whole row."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _sum_slots_kernel(
    src_ptr,
    index_ptr,
    weights_ptr,
    other_ptr,
    out_ptr,
    dots_ptr,
    cols,
    slots,
    HAS_WEIGHTS: tl.constexpr,
    HAS_DOTS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """out[s] = sum_w weights[s, w] * src[index[s, w]]; dots[s, w] = other[s] . src[index[s, w]]."""
    segment = tl.program_id(0).to(tl.int64)
    c_offs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < cols
    w_offs = tl.arange(0, BLOCK_W)
    if HAS_DOTS:
        other = tl.load(other_ptr + segment * cols + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    dots = tl.zeros([BLOCK_W], dtype=tl.float32)
    total = tl.zeros([BLOCK_C], dtype=tl.float32)
    for w in range(slots):
        row = tl.load(index_ptr + segment * slots + w).to(tl.int64)
        if row >= 0:
            values = tl.load(src_ptr + row * cols + c_offs, mask=c_mask, other=0.0).to(tl.float32)
            if HAS_DOTS:
                dots = tl.where(w_offs == w, tl.sum(values * other, axis=0), dots)
            if HAS_WEIGHTS:
                values = values * tl.load(weights_ptr + segment * slots + w).to(tl.float32)
            total += values
    tl.store(out_ptr + segment * cols + c_offs, total.to(out_ptr.dtype.element_ty), mask=c_mask)
    if HAS_DOTS:
        tl.store(dots_ptr + segment * slots + w_offs, dots, mask=w_offs < slots)


@triton.jit
def _scatter_slots_kernel(
    src_ptr,
    index_ptr,
    weights_ptr,
    other_ptr,
    out_ptr,
    dots_ptr,
    cols,
    slots,
    HAS_WEIGHTS: tl.constexpr,
    HAS_DOTS: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """out[index[s, w]] = weights[s, w] * src[s]; dots[s, w] = src[s] . other[index[s, w]]."""
    segment = tl.program_id(0).to(tl.int64)
    c_offs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < cols
    w_offs = tl.arange(0, BLOCK_W)
    values = tl.load(src_ptr + segment * cols + c_offs, mask=c_mask, other=0)
    dots = tl.zeros([BLOCK_W], dtype=tl.float32)
    for w in range(slots):
        row = tl.load(index_ptr + segment * slots + w).to(tl.int64)
        if row >= 0:
            if HAS_WEIGHTS:
                weight = tl.load(weights_ptr + segment * slots + w).to(tl.float32)
                written = (values.to(tl.float32) * weight).to(out_ptr.dtype.element_ty)
            else:
                written = values
            tl.store(out_ptr + row * cols + c_offs, written, mask=c_mask)
            if HAS_DOTS:
                other = tl.load(other_ptr + row * cols + c_offs, mask=c_mask, other=0.0).to(tl.float32)
                dots = tl.where(w_offs == w, tl.sum(values.to(tl.float32) * other, axis=0), dots)
    if HAS_DOTS:
        tl.store(dots_ptr + segment * slots + w_offs, dots, mask=w_offs < slots)


@triton.jit
def _zero_rows_kernel(out_ptr, rows_ptr, cols, BLOCK_C: tl.constexpr):
    row = tl.load(rows_ptr + tl.program_id(0)).to(tl.int64)
    if row >= 0:
        c_offs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        zeros = tl.zeros([BLOCK_C], dtype=out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * cols + c_offs, zeros, mask=c_offs < cols)


def _launch(cols: int, dots: bool) -> tuple[int, int, int]:
    """Column blocks per row, columns per block and warps (about 8 bf16 columns per thread). Without dot products a
    block is the largest power of two up to 1024 that divides the row (no idle lanes), at least 128; with them, one
    block covers the row."""
    block = triton.next_power_of_2(cols)
    if not dots:
        block = min(1024, block)
        while block > 128 and cols % block:
            block //= 2
    return triton.cdiv(cols, block), block, min(8, max(1, block // 256))


def sum_slots(src: Tensor, index: Tensor, weights: Tensor | None, other: Tensor | None) -> tuple[Tensor, Tensor]:
    """The sum over every segment's slots, ``[S, C]`` in the dtype of ``src``, and with ``other`` (``[S, C]``) the fp32
    dot products ``[S, W]`` (otherwise an empty tensor)."""
    segments, slots = index.shape
    cols = src.shape[1]
    out = src.new_empty((segments, cols))
    dots = src.new_empty((segments, slots) if other is not None else (0,), dtype=torch.float32)
    if segments > 0:
        blocks, block_c, warps = _launch(cols, other is not None)
        _sum_slots_kernel[(segments, blocks)](
            src,
            index,
            weights if weights is not None else src,
            other if other is not None else src,
            out,
            dots,
            cols,
            slots,
            HAS_WEIGHTS=weights is not None,
            HAS_DOTS=other is not None,
            BLOCK_W=triton.next_power_of_2(slots),
            BLOCK_C=block_c,
            num_warps=warps,
        )
    return out, dots


def scatter_slots(
    src: Tensor,
    index: Tensor,
    num_rows: int,
    weights: Tensor | None,
    other: Tensor | None,
    padding_rows: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """The ``[num_rows, C]`` rows the segments write, zero in ``padding_rows``, and with ``other`` (``[num_rows, C]``)
    the fp32 dot products ``[S, W]`` (otherwise an empty tensor)."""
    segments, slots = index.shape
    cols = src.shape[1]
    out = src.new_empty((num_rows, cols))
    dots = src.new_empty((segments, slots) if other is not None else (0,), dtype=torch.float32)
    if segments > 0:
        blocks, block_c, warps = _launch(cols, other is not None)
        _scatter_slots_kernel[(segments, blocks)](
            src,
            index,
            weights if weights is not None else src,
            other if other is not None else src,
            out,
            dots,
            cols,
            slots,
            HAS_WEIGHTS=weights is not None,
            HAS_DOTS=other is not None,
            BLOCK_W=triton.next_power_of_2(slots),
            BLOCK_C=block_c,
            num_warps=warps,
        )
    if padding_rows is not None and padding_rows.numel() > 0:
        blocks, block_c, _ = _launch(cols, False)
        _zero_rows_kernel[(padding_rows.numel(), blocks)](out, padding_rows, cols, BLOCK_C=block_c)
    return out, dots
