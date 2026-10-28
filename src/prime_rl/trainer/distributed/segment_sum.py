"""Row segment sums for the expert-parallel token dispatch: out[s] = sum_j weights[j] * src[rows[j]] over the entries j of
segment s (CSR: `seg_ptr`), accumulated in fp32. Every row of `src` belongs to at most one segment, so the backward is
a gather with plain writes, deterministic, without atomics."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _segment_sum_kernel(
    src_ptr,
    rows_ptr,
    seg_ptr,
    weights_ptr,
    out_ptr,
    dim,
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    segment = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < dim
    start = tl.load(seg_ptr + segment)
    end = tl.load(seg_ptr + segment + 1)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for j in range(start, end):
        row = tl.load(rows_ptr + j).to(tl.int64)
        value = tl.load(src_ptr + row * dim + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_WEIGHTS:
            value = value * tl.load(weights_ptr + j).to(tl.float32)
        acc += value
    tl.store(out_ptr + segment.to(tl.int64) * dim + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op("prime_rl::segment_sum", mutates_args=())
def _segment_sum(src: Tensor, rows: Tensor, segment_of_row: Tensor, seg_ptr: Tensor, weights: Tensor | None) -> Tensor:
    num_segments, dim = seg_ptr.numel() - 1, src.shape[1]
    if src.device.type != "cuda":
        values = src[rows].float()
        if weights is not None:
            values = values * weights.float().unsqueeze(1)
        out = values.new_zeros((num_segments, dim)).index_add_(0, segment_of_row, values)
        return out.to(src.dtype)
    out = src.new_empty((num_segments, dim))
    if num_segments > 0:
        block = min(1024, triton.next_power_of_2(dim))
        _segment_sum_kernel[(num_segments, triton.cdiv(dim, block))](
            src.contiguous(),
            rows,
            seg_ptr,
            weights if weights is not None else rows,
            out,
            dim,
            HAS_WEIGHTS=weights is not None,
            BLOCK_D=block,
        )
    return out


@_segment_sum.register_fake
def _segment_sum_fake(
    src: Tensor, rows: Tensor, segment_of_row: Tensor, seg_ptr: Tensor, weights: Tensor | None
) -> Tensor:
    return src.new_empty((seg_ptr.shape[0] - 1, src.shape[1]))


def _segment_sum_setup_context(ctx, inputs, output) -> None:
    src, rows, segment_of_row, seg_ptr, weights = inputs
    ctx.num_rows = src.shape[0]
    ctx.has_weights = weights is not None
    ctx.weights_need_grad = weights is not None and ctx.needs_input_grad[4]
    ctx.save_for_backward(rows, segment_of_row, weights, src if ctx.weights_need_grad else None)


def _segment_sum_backward(ctx, grad_output: Tensor):
    rows, segment_of_row, weights, src = ctx.saved_tensors
    grad_rows = grad_output[segment_of_row]
    if ctx.has_weights:
        grad_rows = (grad_rows.float() * weights.float().unsqueeze(1)).to(grad_output.dtype)
    grad_src = grad_output.new_zeros((ctx.num_rows, grad_output.shape[1]))
    grad_src[rows] = grad_rows
    grad_weights = None
    if ctx.weights_need_grad:
        grad_weights = (grad_output[segment_of_row].float() * src[rows].float()).sum(dim=1).to(weights.dtype)
    return grad_src, None, None, None, grad_weights


_segment_sum.register_autograd(_segment_sum_backward, setup_context=_segment_sum_setup_context)


def segment_sum(
    src: Tensor, rows: Tensor, segment_of_row: Tensor, seg_ptr: Tensor, weights: Tensor | None = None
) -> Tensor:
    """`out[s] = sum(weights[j] * src[rows[j]] for j in range(seg_ptr[s], seg_ptr[s + 1]))`; `segment_of_row[j]` is
    the segment of entry j. Each row of `src` appears at most once in `rows`."""
    return _segment_sum(src, rows, segment_of_row, seg_ptr, weights)


class _GatherRows(torch.autograd.Function):
    """`src[index]` with -1 selecting a zero row. The backward sums the gradient rows that read each source row with
    `segment_sum` over the CSR (`rows`, `segment_of_row`, `seg_ptr`) of those readers."""

    @staticmethod
    def forward(ctx, src: Tensor, index: Tensor, rows: Tensor, segment_of_row: Tensor, seg_ptr: Tensor) -> Tensor:
        ctx.save_for_backward(rows, segment_of_row, seg_ptr)
        return torch.cat((src, src.new_zeros((1, src.shape[1]))))[index]

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        rows, segment_of_row, seg_ptr = ctx.saved_tensors
        grad_src = segment_sum(grad_output.reshape(-1, grad_output.shape[-1]), rows, segment_of_row, seg_ptr)
        return grad_src, None, None, None, None


def gather_rows(src: Tensor, index: Tensor, rows: Tensor, segment_of_row: Tensor, seg_ptr: Tensor) -> Tensor:
    """`src[index]` (-1 gives a zero row); (`rows`, `segment_of_row`, `seg_ptr`) list, per row of `src`, the positions
    of `index` that read it."""
    return _GatherRows.apply(src, index, rows, segment_of_row, seg_ptr)
