from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import torch
import triton
import triton.language as tl

FP8_MAX = tl.constexpr(448.0)
FP8_MIN = tl.constexpr(-448.0)
MIN_SCALE = 1e-4
GROUP_ALIGNMENT = 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def ue8m0_for_device(device: torch.device | None = None) -> bool:
    """DeepGEMM requires UE8M0 (power-of-2) scales on SM100; SM90 supports exact float
    scales, and UE8M0 there is a pure precision loss (it measurably raises the
    trainer/inference mismatch KL)."""
    return torch.cuda.get_device_capability(device)[0] >= 10


# ---------------------------------------------------------------------------
# Layout building
# ---------------------------------------------------------------------------


def build_grouped_layout(offs: torch.Tensor, *, total_m: int | None = None):
    assert offs.dim() == 1
    assert offs.dtype == torch.int32
    device = offs.device
    total_m = (total_m if total_m is not None else int(offs[-1].item())) if offs.numel() else 0
    starts_tensor = torch.empty_like(offs)
    if offs.numel() > 0:
        starts_tensor[0] = 0
        if offs.numel() > 1:
            starts_tensor[1:] = offs[:-1]
    actual_ms_tensor = offs - starts_tensor
    aligned_ms_tensor = ((actual_ms_tensor + GROUP_ALIGNMENT - 1) // GROUP_ALIGNMENT) * GROUP_ALIGNMENT
    padded_ends = aligned_ms_tensor.cumsum(0)
    block_starts_tensor = (padded_ends - aligned_ms_tensor) // GROUP_ALIGNMENT
    ks_tensor = aligned_ms_tensor.contiguous()
    padded_total_m = int(padded_ends[-1].item()) if offs.numel() else 0
    total_blocks = padded_total_m // GROUP_ALIGNMENT
    grouped_layout = torch.empty((padded_total_m,), dtype=torch.int32, device=device)
    block_to_group = torch.empty((total_blocks,), dtype=torch.int32, device=device)
    if offs.numel():
        _build_grouped_layout_triton(
            grouped_layout,
            block_to_group,
            starts_tensor,
            actual_ms_tensor,
            aligned_ms_tensor,
            block_starts_tensor,
        )
    return (
        total_m,
        padded_total_m,
        grouped_layout,
        block_to_group,
        ks_tensor,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
    )


def _build_grouped_layout_triton(
    grouped_layout: torch.Tensor,
    block_to_group: torch.Tensor,
    starts_tensor: torch.Tensor,
    actual_ms_tensor: torch.Tensor,
    aligned_ms_tensor: torch.Tensor,
    block_starts_tensor: torch.Tensor,
) -> None:
    _build_grouped_layout_kernel[(actual_ms_tensor.numel(),)](
        grouped_layout,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        aligned_ms_tensor,
        block_starts_tensor,
        BLOCK_M=128,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _build_grouped_layout_kernel(
    grouped_layout_ptr,
    block_to_group_ptr,
    starts_ptr,
    actual_ms_ptr,
    aligned_ms_ptr,
    block_starts_ptr,
    BLOCK_M: tl.constexpr,
):
    pid_g = tl.program_id(axis=0)
    actual_m = tl.load(actual_ms_ptr + pid_g)
    aligned_m = tl.load(aligned_ms_ptr + pid_g)
    block_start = tl.load(block_starts_ptr + pid_g)
    dst_start = block_start * BLOCK_M
    block_idx = 0
    while block_idx < (aligned_m // BLOCK_M):
        row_offsets = block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        values = tl.where(row_offsets < actual_m, pid_g, -1)
        tl.store(grouped_layout_ptr + dst_start + row_offsets, values)
        tl.store(block_to_group_ptr + block_start + block_idx, pid_g)
        block_idx += 1


@triton.jit
def _unpack_grouped_rows_kernel(
    x_ptr,
    block_to_group_ptr,
    starts_ptr,
    actual_ms_ptr,
    block_starts_ptr,
    out_ptr,
    cols,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_BLOCK_M: tl.constexpr,
):
    pid_blk = tl.program_id(axis=0)
    pid_sub = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    pid_g = tl.load(block_to_group_ptr + pid_blk)
    block_start = tl.load(block_starts_ptr + pid_g)
    dst_start = tl.load(starts_ptr + pid_g)
    actual_m = tl.load(actual_ms_ptr + pid_g)
    row_offsets = (pid_blk - block_start) * GROUP_BLOCK_M + pid_sub * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    src_rows_i64 = (pid_blk * GROUP_BLOCK_M + pid_sub * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    dst_rows_i64 = (dst_start + row_offsets).to(tl.int64)
    col_offsets_i64 = col_offsets.to(tl.int64)
    valid_rows = row_offsets < actual_m
    valid_cols = col_offsets < cols
    x = tl.load(
        x_ptr + src_rows_i64[:, None] * stride_xm + col_offsets_i64[None, :] * stride_xn,
        mask=valid_rows[:, None] & valid_cols[None, :],
        other=0.0,
    )
    tl.store(
        out_ptr + dst_rows_i64[:, None] * stride_ym + col_offsets_i64[None, :] * stride_yn,
        x,
        mask=valid_rows[:, None] & valid_cols[None, :],
    )


@triton.jit
def _fp8_scale(amax, USE_UE8M0: tl.constexpr):
    scale = tl.math.div_rn(tl.maximum(amax.to(tl.float32), 1e-10), FP8_MAX)
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    return scale


@triton.jit
def _fp8_quantize(x, scale):
    """``x / scale`` rounded to e4m3, bit-identical to ``div_rn`` followed by a [-448, 448] clamp.

    One Markstein correction step on ``x * (1 / scale)`` yields the correctly rounded quotient
    without a per-element IEEE division, and the satfinite fp8 conversion does the clamping.
    """
    rcp = tl.math.div_rn(1.0, scale)
    q = x * rcp
    y = tl.math.fma(tl.math.fma(-q, scale, x), rcp, q)
    # The correction step turns -0 into +0; q carries the sign of x.
    return tl.where(q == 0.0, q, y).to(tl.float8e4nv)


@triton.jit
def _per_token_fp8_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    rows,
    cols,
    stride_xm,
    stride_sk,
    USE_UE8M0: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    row_offsets = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    col_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = row_offsets < rows
    x_ptrs = x_ptr + row_offsets[:, None] * stride_xm + col_offsets[None, :]
    out_ptrs = out_ptr + row_offsets[:, None] * cols + col_offsets[None, :]
    sf_ptrs = sf_ptr + pid_k * stride_sk + row_offsets
    if EVEN_ROWS:
        x = tl.load(x_ptrs)
    else:
        x = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
    scale = _fp8_scale(tl.max(tl.abs(x), axis=1), USE_UE8M0)
    y = _fp8_quantize(x.to(tl.float32), scale[:, None])
    if EVEN_ROWS:
        tl.store(out_ptrs, y)
        tl.store(sf_ptrs, scale)
    else:
        tl.store(out_ptrs, y, mask=row_mask[:, None])
        tl.store(sf_ptrs, scale, mask=row_mask)


@triton.jit
def _per_token_dequant_kernel(
    q_ptr,
    sf_ptr,
    out_ptr,
    rows,
    cols,
    stride_sm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    row_offsets = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    col_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = row_offsets < rows
    offsets = row_offsets[:, None] * cols + col_offsets[None, :]
    q = tl.load(q_ptr + offsets, mask=row_mask[:, None], other=0.0)
    scale = tl.load(sf_ptr + row_offsets * stride_sm + pid_k, mask=row_mask, other=0.0)
    tl.store(out_ptr + offsets, (q.to(tl.float32) * scale[:, None]).to(tl.bfloat16), mask=row_mask[:, None])


@triton.jit
def _per_token_fp8_tp_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    rows,
    padded_rows,
    cols,
    stride_xm,
    USE_UE8M0: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Loads a row-major [BLOCK_M tokens, BLOCK_N channels] tile, scales each channel over the
    # tokens and stores the tile transposed; tokens past `rows` are written as zeros.
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + row_offsets.to(tl.int64)[:, None] * stride_xm + col_offsets[None, :]
    if EVEN_ROWS:
        x = tl.load(x_ptrs)
    else:
        x = tl.load(x_ptrs, mask=(row_offsets < rows)[:, None], other=0.0)
    scale = _fp8_scale(tl.max(tl.abs(x), axis=0), USE_UE8M0)
    y = _fp8_quantize(x.to(tl.float32), scale[None, :])
    tl.store(out_ptr + col_offsets.to(tl.int64)[:, None] * padded_rows + row_offsets[None, :], tl.trans(y))
    tl.store(sf_ptr + pid_m * cols + col_offsets, scale)


@triton.jit
def _grouped_per_token_fp8_kernel(
    x_ptr,
    block_to_group_ptr,
    starts_ptr,
    actual_ms_ptr,
    block_starts_ptr,
    out_ptr,
    sf_ptr,
    cols,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_sm,
    stride_sk,
    USE_UE8M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_BLOCK_M: tl.constexpr,
):
    pid_blk = tl.program_id(axis=0)
    pid_sub = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    pid_g = tl.load(block_to_group_ptr + pid_blk)
    src_start = tl.load(starts_ptr + pid_g)
    actual_m = tl.load(actual_ms_ptr + pid_g)
    block_start = tl.load(block_starts_ptr + pid_g)
    local_block = pid_blk - block_start
    row_offsets = local_block * GROUP_BLOCK_M + pid_sub * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    src_rows_i64 = (src_start + row_offsets).to(tl.int64)
    dst_rows_i64 = (pid_blk * GROUP_BLOCK_M + pid_sub * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    col_offsets_i64 = col_offsets.to(tl.int64)
    valid_rows = row_offsets < actual_m
    valid_cols = col_offsets < cols
    x = tl.load(
        x_ptr + src_rows_i64[:, None] * stride_xm + col_offsets_i64[None, :] * stride_xn,
        mask=valid_rows[:, None] & valid_cols[None, :],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    scale = tl.math.div_rn(amax, FP8_MAX)
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    y = tl.clamp(tl.math.div_rn(x, scale[:, None]), FP8_MIN, FP8_MAX)
    tl.store(
        out_ptr + dst_rows_i64[:, None] * stride_ym + col_offsets_i64[None, :] * stride_yn,
        y.to(tl.float8e4nv),
        mask=valid_rows[:, None] & valid_cols[None, :],
    )
    tl.store(sf_ptr + dst_rows_i64 * stride_sm + pid_k * stride_sk, scale, mask=valid_rows)


@triton.jit
def _grouped_per_channel_fp8_kernel(
    x_ptr,
    block_to_group_ptr,
    starts_ptr,
    actual_ms_ptr,
    aligned_ms_ptr,
    block_starts_ptr,
    out_ptr,
    sf_ptr,
    cols,
    stride_xm,
    stride_xn,
    stride_sf0,
    stride_sf1,
    USE_UE8M0: tl.constexpr,
    K_MAJOR: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_blk = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_g = tl.load(block_to_group_ptr + pid_blk)
    src_start = tl.load(starts_ptr + pid_g)
    actual_m = tl.load(actual_ms_ptr + pid_g)
    aligned_m = tl.load(aligned_ms_ptr + pid_g)
    block_start = tl.load(block_starts_ptr + pid_g)
    local_block = pid_blk - block_start
    row_offsets = local_block * BLOCK_K + tl.arange(0, BLOCK_K)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    src_rows_i64 = (src_start + row_offsets).to(tl.int64)
    row_offsets_i64 = row_offsets.to(tl.int64)
    col_offsets_i64 = col_offsets.to(tl.int64)
    valid_rows = row_offsets < actual_m
    valid_cols = col_offsets < cols
    x = tl.load(
        x_ptr + src_rows_i64[:, None] * stride_xm + col_offsets_i64[None, :] * stride_xn,
        mask=valid_rows[:, None] & valid_cols[None, :],
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-10)
    scale = tl.math.div_rn(amax, FP8_MAX)
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    y = tl.clamp(tl.math.div_rn(x, scale[None, :]), FP8_MIN, FP8_MAX)
    flat_base = block_start.to(tl.int64) * BLOCK_K * cols
    if K_MAJOR:
        out_ptrs = out_ptr + flat_base + col_offsets_i64[:, None] * aligned_m + row_offsets_i64[None, :]
        tl.store(
            out_ptrs,
            tl.trans(y).to(tl.float8e4nv),
            mask=valid_cols[:, None] & (row_offsets[None, :] < aligned_m),
        )
    else:
        out_ptrs = out_ptr + flat_base + row_offsets_i64[:, None] * cols + col_offsets_i64[None, :]
        tl.store(
            out_ptrs,
            y.to(tl.float8e4nv),
            mask=(row_offsets[:, None] < aligned_m) & valid_cols[None, :],
        )
    tl.store(
        sf_ptr + pid_blk * stride_sf0 + col_offsets_i64 * stride_sf1,
        scale,
        mask=valid_cols,
    )


@triton.jit
def _grouped_per_block_fp8_kernel(
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
    USE_UE8M0: tl.constexpr,
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
    scale = tl.maximum(amax / FP8_MAX, 1e-4)
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    y = x / scale
    tl.store(
        out_ptr + pid_g * stride_yg + row_offsets[:, None] * stride_ym + col_offsets[None, :] * stride_yn,
        y.to(tl.float8e4nv),
        mask=mask,
    )
    tl.store(sf_ptr + pid_g * stride_sg + pid_m * stride_sm + pid_n * stride_sn, scale, mask=pid_g < groups)


# ---------------------------------------------------------------------------
# Public quantization functions
# ---------------------------------------------------------------------------


def unpack_rows_triton(
    x: torch.Tensor,
    total_m: int,
    block_to_group: torch.Tensor,
    starts_tensor: torch.Tensor,
    actual_ms_tensor: torch.Tensor,
    block_starts_tensor: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty((total_m, x.size(1)), device=x.device, dtype=x.dtype)
    if total_m == 0:
        return out
    grid = (block_to_group.numel(), GROUP_ALIGNMENT // 32, ceil_div(x.size(1), 128))
    _unpack_grouped_rows_kernel[grid](
        x,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
        out,
        x.size(1),
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=32,
        BLOCK_N=128,
        GROUP_BLOCK_M=GROUP_ALIGNMENT,
        num_warps=4,
    )
    return out


def per_token_cast_to_fp8_triton(
    x: torch.Tensor, use_ue8m0: bool, gran_k: int = GROUP_ALIGNMENT
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token (1 x gran_k) fp8 cast of ``x``.

    Scales are returned MN-major and TMA-aligned, the layout DeepGEMM consumes without a
    transpose kernel of its own.
    """
    assert x.dim() == 2 and x.stride(1) == 1
    assert gran_k == GROUP_ALIGNMENT
    rows, cols = x.shape
    assert cols % gran_k == 0
    block_m = 32
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sf = torch.empty((cols // gran_k, ceil_div(rows, 4) * 4), device=x.device, dtype=torch.float32)
    _per_token_fp8_kernel[(ceil_div(rows, block_m), cols // gran_k)](
        x,
        out,
        sf,
        rows,
        cols,
        x.stride(0),
        sf.stride(0),
        USE_UE8M0=use_ue8m0,
        EVEN_ROWS=rows % block_m == 0,
        BLOCK_M=block_m,
        BLOCK_K=gran_k,
        num_warps=4,
    )
    return out, sf[:, :rows].T


def per_token_dequant_fp8_triton(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """bf16 rows from a per-token (1 x 128) fp8 cast; ``sf`` is ``[rows, cols / 128]``, token-major.

    Exact for power-of-two scales: an e4m3 value times a power of two is representable in bf16.
    """
    assert q.dim() == 2 and q.is_contiguous() and sf.stride(1) == 1
    rows, cols = q.shape
    assert sf.shape == (rows, cols // GROUP_ALIGNMENT)
    out = torch.empty(rows, cols, device=q.device, dtype=torch.bfloat16)
    if rows:
        block_m = 32
        _per_token_dequant_kernel[(ceil_div(rows, block_m), cols // GROUP_ALIGNMENT)](
            q, sf, out, rows, cols, sf.stride(0), BLOCK_M=block_m, BLOCK_K=GROUP_ALIGNMENT, num_warps=4
        )
    return out


def grouped_per_token_cast_to_fp8_triton(
    x: torch.Tensor,
    padded_total_m: int,
    block_to_group: torch.Tensor,
    starts_tensor: torch.Tensor,
    actual_ms_tensor: torch.Tensor,
    block_starts_tensor: torch.Tensor,
    use_ue8m0: bool,
    gran_k: int = GROUP_ALIGNMENT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert gran_k == GROUP_ALIGNMENT
    out = torch.empty((padded_total_m, x.size(1)), device=x.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty(
        (padded_total_m, ceil_div(x.size(1), gran_k)),
        device=x.device,
        dtype=torch.float32,
    )
    if block_to_group.numel() == 0:
        return out, sf
    grid = (block_to_group.numel(), GROUP_ALIGNMENT // 32, ceil_div(x.size(1), gran_k))
    _grouped_per_token_fp8_kernel[grid](
        x,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        block_starts_tensor,
        out,
        sf,
        x.size(1),
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        sf.stride(0),
        sf.stride(1),
        USE_UE8M0=use_ue8m0,
        BLOCK_M=32,
        BLOCK_K=gran_k,
        GROUP_BLOCK_M=GROUP_ALIGNMENT,
        num_warps=4,
    )
    return out, sf


def grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
    x: torch.Tensor,
    padded_total_m: int,
    block_to_group: torch.Tensor,
    starts_tensor: torch.Tensor,
    actual_ms_tensor: torch.Tensor,
    ks_tensor: torch.Tensor,
    block_starts_tensor: torch.Tensor,
    use_ue8m0: bool,
    gran_k: int = GROUP_ALIGNMENT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert gran_k == GROUP_ALIGNMENT
    out = torch.empty((padded_total_m * x.size(1),), device=x.device, dtype=torch.float8_e4m3fn)
    total_blocks = padded_total_m // gran_k
    sf = torch.empty((total_blocks, x.size(1)), device=x.device, dtype=torch.float32)
    if block_to_group.numel() == 0:
        return out, sf.T
    block_n = 16
    grid = (block_to_group.numel(), ceil_div(x.size(1), block_n))
    _grouped_per_channel_fp8_kernel[grid](
        x,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        ks_tensor,
        block_starts_tensor,
        out,
        sf,
        x.size(1),
        x.stride(0),
        x.stride(1),
        sf.stride(0),
        sf.stride(1),
        USE_UE8M0=use_ue8m0,
        K_MAJOR=True,
        BLOCK_K=gran_k,
        BLOCK_N=block_n,
        num_warps=2,
    )
    return out, sf.T


def grouped_per_channel_cast_to_fp8_rowmajor_triton(
    x: torch.Tensor,
    padded_total_m: int,
    block_to_group: torch.Tensor,
    starts_tensor: torch.Tensor,
    actual_ms_tensor: torch.Tensor,
    ks_tensor: torch.Tensor,
    block_starts_tensor: torch.Tensor,
    use_ue8m0: bool,
    gran_k: int = GROUP_ALIGNMENT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert gran_k == GROUP_ALIGNMENT
    out = torch.empty((padded_total_m, x.size(1)), device=x.device, dtype=torch.float8_e4m3fn)
    total_blocks = padded_total_m // gran_k
    sf = torch.empty((total_blocks, x.size(1)), device=x.device, dtype=torch.float32)
    if block_to_group.numel() == 0:
        return out, sf
    block_n = 64
    grid = (block_to_group.numel(), ceil_div(x.size(1), block_n))
    _grouped_per_channel_fp8_kernel[grid](
        x,
        block_to_group,
        starts_tensor,
        actual_ms_tensor,
        ks_tensor,
        block_starts_tensor,
        out,
        sf,
        x.size(1),
        x.stride(0),
        x.stride(1),
        sf.stride(0),
        sf.stride(1),
        USE_UE8M0=use_ue8m0,
        K_MAJOR=False,
        BLOCK_K=gran_k,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out, sf


def grouped_per_block_cast_to_fp8_triton(
    x: torch.Tensor, use_ue8m0: bool, gran_k: int = GROUP_ALIGNMENT
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    assert gran_k == GROUP_ALIGNMENT
    groups, rows, cols = x.shape
    out = torch.empty((groups, rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty(
        (groups, ceil_div(rows, gran_k), ceil_div(cols, gran_k)),
        device=x.device,
        dtype=torch.float32,
    )
    grid = (groups, ceil_div(rows, gran_k), ceil_div(cols, gran_k))
    _grouped_per_block_fp8_kernel[grid](
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
        USE_UE8M0=use_ue8m0,
        BLOCK_M=gran_k,
        BLOCK_N=gran_k,
        num_warps=8,
    )
    return out, sf


def per_block_cast_to_fp8_triton(
    x: torch.Tensor, use_ue8m0: bool, gran_k: int = GROUP_ALIGNMENT
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    out, sf = grouped_per_block_cast_to_fp8_triton(
        x.unsqueeze(0),
        use_ue8m0,
        gran_k,
    )
    return out[0], sf[0]


def stacked_per_block_cast_to_fp8_triton(
    weights: Sequence[torch.Tensor], use_ue8m0: bool, gran_k: int = GROUP_ALIGNMENT
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``per_block_cast_to_fp8_triton`` of the row-wise concatenation of ``weights``, without materializing it.

    Each weight's row count must be a multiple of ``gran_k``, so no block straddles two weights and the result
    equals the separate casts stacked."""
    assert gran_k == GROUP_ALIGNMENT
    cols = weights[0].shape[1]
    assert all(w.dim() == 2 and w.shape[1] == cols and w.shape[0] % gran_k == 0 for w in weights)
    rows = sum(w.shape[0] for w in weights)
    out = torch.empty((rows, cols), device=weights[0].device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((rows // gran_k, ceil_div(cols, gran_k)), device=weights[0].device, dtype=torch.float32)
    row = 0
    for w in weights:
        w_rows = w.shape[0]
        out_w, sf_w = out[row : row + w_rows], sf[row // gran_k : (row + w_rows) // gran_k]
        _grouped_per_block_fp8_kernel[(1, w_rows // gran_k, ceil_div(cols, gran_k))](
            w,
            out_w,
            sf_w,
            1,
            w_rows,
            cols,
            0,
            w.stride(0),
            w.stride(1),
            0,
            out_w.stride(0),
            out_w.stride(1),
            0,
            sf_w.stride(0),
            sf_w.stride(1),
            USE_UE8M0=use_ue8m0,
            BLOCK_M=gran_k,
            BLOCK_N=gran_k,
            num_warps=8,
        )
        row += w_rows
    return out, sf


def per_block_cast_to_fp8_tp_triton(
    x: torch.Tensor, use_ue8m0: bool, gran_k: int = GROUP_ALIGNMENT
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-fp8 cast of ``x.T`` without materializing the transpose."""
    assert x.dim() == 2
    assert gran_k == GROUP_ALIGNMENT
    rows, cols = x.shape
    x3 = x.unsqueeze(0)
    out = torch.empty((cols, rows), device=x.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((ceil_div(cols, gran_k), ceil_div(rows, gran_k)), device=x.device, dtype=torch.float32)
    grid = (1, ceil_div(rows, gran_k), ceil_div(cols, gran_k))
    _grouped_per_block_fp8_kernel[grid](
        x3,
        out,
        sf,
        1,
        rows,
        cols,
        x3.stride(0),
        x3.stride(1),
        x3.stride(2),
        # transposed output: x's element (row, col) lands at out[col, row]
        cols * rows,
        1,
        rows,
        # transposed scales: x's tile (pid_m, pid_n) lands at sf[pid_n, pid_m]
        ceil_div(cols, gran_k) * ceil_div(rows, gran_k),
        1,
        ceil_div(rows, gran_k),
        USE_UE8M0=use_ue8m0,
        BLOCK_M=gran_k,
        BLOCK_N=gran_k,
        num_warps=8,
    )
    return out, sf


def per_token_cast_to_fp8_tp_triton(
    x: torch.Tensor, use_ue8m0: bool, gran_k: int = GROUP_ALIGNMENT
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token fp8 cast of ``x.T`` without materializing the transpose.

    The token dimension (``x``'s rows) is zero-padded to a multiple of ``gran_k``, as DeepGEMM's
    (1, 1, 128) wgrad recipe requires: the result is ``(cols, padded_rows)``. Scales are returned
    MN-major, the layout DeepGEMM consumes without a transpose kernel of its own.
    """
    assert x.dim() == 2 and x.stride(1) == 1
    assert gran_k == GROUP_ALIGNMENT
    rows, cols = x.shape
    block_n = 32
    assert cols % block_n == 0
    padded_rows = ceil_div(rows, gran_k) * gran_k
    out = torch.empty((cols, padded_rows), device=x.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((padded_rows // gran_k, cols), device=x.device, dtype=torch.float32)
    _per_token_fp8_tp_kernel[(cols // block_n, padded_rows // gran_k)](
        x,
        out,
        sf,
        rows,
        padded_rows,
        cols,
        x.stride(0),
        USE_UE8M0=use_ue8m0,
        EVEN_ROWS=rows == padded_rows,
        BLOCK_M=gran_k,
        BLOCK_N=block_n,
        num_warps=2,
    )
    return out, sf.T
