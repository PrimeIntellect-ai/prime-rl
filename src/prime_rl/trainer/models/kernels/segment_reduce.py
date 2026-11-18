import torch
import triton
import triton.language as tl


@triton.jit
def _segment_sum_kernel(x_ptr, offsets_ptr, out_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(offsets_ptr + row)
    end = tl.load(offsets_ptr + row + 1)
    s = tl.zeros([BLOCK], tl.float32)
    for block_start in range(start, end, BLOCK):
        idx = block_start + tl.arange(0, BLOCK)
        s += tl.load(x_ptr + idx, mask=idx < end, other=0.0)
    tl.store(out_ptr + row, tl.sum(s, axis=0))


@triton.jit
def _segment_max_and_tail_kernel(x_ptr, offsets_ptr, max_ptr, tail_ptr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(offsets_ptr + row)
    end = tl.load(offsets_ptr + row + 1)
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    for block_start in range(start, end, BLOCK):
        idx = block_start + tl.arange(0, BLOCK)
        m = tl.maximum(m, tl.load(x_ptr + idx, mask=idx < end, other=float("-inf")))
    row_max = tl.max(m, axis=0)
    below = tl.zeros([BLOCK], tl.float32)
    at_max = tl.zeros([BLOCK], tl.int32)
    for block_start in range(start, end, BLOCK):
        idx = block_start + tl.arange(0, BLOCK)
        valid = idx < end
        x = tl.load(x_ptr + idx, mask=valid, other=float("-inf"))
        below += tl.where(x < row_max, tl.exp(x - row_max), 0.0)
        at_max += (valid & (x == row_max)).to(tl.int32)
    tl.store(max_ptr + row, row_max)
    tl.store(tail_ptr + row, tl.sum(below, axis=0) + (tl.sum(at_max, axis=0) - 1).to(tl.float32))


def segment_max_and_tail(x: torch.Tensor, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Max and ``max_tail_spread`` tail (``sum exp(x - max)`` over all entries but one maximal one) of each CSR row
    ``x[offsets[i]:offsets[i + 1]]`` in a fixed order. An empty row gets max -inf and tail -1, so ``log1p(tail)`` is
    -inf."""
    row_max = torch.empty(offsets.shape[0] - 1, device=x.device, dtype=torch.float32)
    tail = torch.empty_like(row_max)
    if row_max.numel():
        _segment_max_and_tail_kernel[(row_max.numel(),)](x.float().contiguous(), offsets, row_max, tail, BLOCK=256)
    return row_max, tail


def segment_sum(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Sum of each CSR row ``x[offsets[i]:offsets[i + 1]]`` in a fixed order."""
    out = torch.empty(offsets.shape[0] - 1, device=x.device, dtype=torch.float32)
    if out.numel():
        _segment_sum_kernel[(out.numel(),)](x.float().contiguous(), offsets, out, BLOCK=256)
    return out


@triton.jit
def _csr_matmul_kernel(
    offsets_ptr, cols_ptr, vals_ptr, dense_ptr, out_ptr, H, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr
):
    row = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    start = tl.load(offsets_ptr + row)
    end = tl.load(offsets_ptr + row + 1)
    acc = tl.zeros([BLOCK_H], tl.float32)
    for block_start in range(start, end, BLOCK_K):
        k = block_start + tl.arange(0, BLOCK_K)
        cols = tl.load(cols_ptr + k, mask=k < end, other=0).to(tl.int64)
        vals = tl.load(vals_ptr + k, mask=k < end, other=0.0)
        tile = tl.load(
            dense_ptr + cols[:, None] * H + h[None, :], mask=(k < end)[:, None] & (h < H)[None, :], other=0.0
        )
        acc += tl.sum(vals[:, None] * tile.to(tl.float32), axis=0)
    tl.store(out_ptr + row.to(tl.int64) * H + h, acc.to(out_ptr.dtype.element_ty), mask=h < H)


def csr_matmul(offsets: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor, dense: torch.Tensor, out: torch.Tensor):
    """``out[i] = sum_k vals[k] * dense[cols[k]]`` over the CSR row ``offsets[i]:offsets[i + 1]``, accumulated in
    fp32 in a fixed order (deterministic). Writes every row of ``out``."""
    rows, hidden = out.shape
    if rows:
        grid = (rows, triton.cdiv(hidden, 256))
        _csr_matmul_kernel[grid](offsets, cols, vals, dense, out, hidden, BLOCK_K=16, BLOCK_H=256)
    return out
