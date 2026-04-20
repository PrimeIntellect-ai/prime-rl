import torch
import triton
import triton.language as tl

BLOCK_SIZE = 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y



@triton.jit
def _grouped_fp8_block_quantize(
    x_ptr,
    out_ptr,
    sf_ptr,
    groups,
    rows,
    cols,
    stride_xg,
    stride_xm,
    stride_xn,
    stride_yg,
    stride_ym,
    stride_yn,
    stride_sg,
    stride_sm,
    stride_sn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_g = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (pid_g < groups) & (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
    x = tl.load(
        x_ptr + pid_g * stride_xg + row_offsets[:, None] * stride_xm + col_offsets[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x))
    # Floor matches the pre-Triton PyTorch kernel (main's quantize_to_fp8_blockwise).
    # Raising to 1e-4 was observable drift on blocks with amax < ~0.0448 — tiny
    # nonzero weights got zeroed out at dequant time.
    scale = tl.maximum(amax / 448.0, 1e-12)
    y = x / scale
    tl.store(
        out_ptr + pid_g * stride_yg + row_offsets[:, None] * stride_ym + col_offsets[None, :] * stride_yn,
        y.to(tl.float8e4nv),
        mask=mask,
    )
    tl.store(sf_ptr + pid_g * stride_sg + pid_m * stride_sm + pid_n * stride_sn, scale, mask=pid_g < groups)


def grouped_fp8_block_quantize(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 e4m3 blockwise quantization for a 3D tensor ``(groups, rows, cols)``.

    Every ``(gran_k, gran_k)`` tile shares a single fp32 scale. ``out`` and
    ``sf`` are written in place when provided; otherwise fresh buffers are
    allocated and returned.
    """
    assert x.dim() == 3
    groups, rows, cols = x.shape
    if out is None:
        out = torch.empty((groups, rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    if sf is None:
        sf = torch.empty((groups, ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE)), device=x.device, dtype=torch.float32)
    grid = (groups, ceil_div(rows, BLOCK_SIZE), ceil_div(cols, BLOCK_SIZE))
    _grouped_fp8_block_quantize[grid](
        x,
        out,
        sf,
        groups,
        rows,
        cols,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sf.stride(0),
        sf.stride(1),
        sf.stride(2),
        BLOCK_M=BLOCK_SIZE,
        BLOCK_N=BLOCK_SIZE,
        num_warps=8,
    )
    return out, sf


def fp8_block_quantize(
    x: torch.Tensor, out: torch.Tensor | None = None, sf: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D variant of :func:`grouped_fp8_block_quantize` (``groups=1``)."""
    assert x.dim() == 2
    out = out.unsqueeze(0) if out is not None else None
    sf = sf.unsqueeze(0) if sf is not None else None
    x = x.unsqueeze(0)
    q, s = grouped_fp8_block_quantize(x, out=out, sf=sf)
    return q.squeeze(0), s.squeeze(0)
