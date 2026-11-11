import torch
import triton
import triton.language as tl


@triton.jit
def _segment_reduce_kernel(x_ptr, offsets_ptr, out_ptr, LOGSUMEXP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(offsets_ptr + row)
    end = tl.load(offsets_ptr + row + 1)
    if LOGSUMEXP:
        m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for block_start in range(start, end, BLOCK):
        idx = block_start + tl.arange(0, BLOCK)
        if LOGSUMEXP:
            x = tl.load(x_ptr + idx, mask=idx < end, other=float("-inf"))
            m_new = tl.maximum(m, x)
            finite = m_new > float("-inf")
            s = tl.where(finite, s * tl.exp(m - m_new) + tl.exp(x - m_new), 0.0)
            m = m_new
        else:
            s += tl.load(x_ptr + idx, mask=idx < end, other=0.0)
    if LOGSUMEXP:
        row_max = tl.max(m, axis=0)
        total = tl.sum(tl.where(m > float("-inf"), s * tl.exp(m - row_max), 0.0), axis=0)
        tl.store(out_ptr + row, row_max + tl.log(total))
    else:
        tl.store(out_ptr + row, tl.sum(s, axis=0))


def _segment_reduce(x: torch.Tensor, offsets: torch.Tensor, logsumexp: bool) -> torch.Tensor:
    out = torch.empty(offsets.shape[0] - 1, device=x.device, dtype=torch.float32)
    if out.numel():
        _segment_reduce_kernel[(out.numel(),)](x.float().contiguous(), offsets, out, LOGSUMEXP=logsumexp, BLOCK=256)
    return out


def segment_logsumexp(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Logsumexp of each CSR row ``x[offsets[i]:offsets[i + 1]]`` in a fixed order, -inf for empty rows."""
    return _segment_reduce(x, offsets, logsumexp=True)


def segment_sum(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Sum of each CSR row ``x[offsets[i]:offsets[i + 1]]`` in a fixed order."""
    return _segment_reduce(x, offsets, logsumexp=False)


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
