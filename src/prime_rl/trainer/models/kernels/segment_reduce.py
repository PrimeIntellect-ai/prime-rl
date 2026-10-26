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
