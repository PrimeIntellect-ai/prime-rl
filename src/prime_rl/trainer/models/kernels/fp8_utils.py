from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

FP8_MAX = tl.constexpr(448.0)
FP8_MIN = tl.constexpr(-448.0)
FP8_MAX_RECIPROCAL = tl.constexpr(1.0 / 448.0)
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


def build_grouped_layout(offs: torch.Tensor, padded_total_m: int):
    """Map groups ending at `offs` onto 128-row blocks of a `padded_total_m`-row buffer, without a host sync.

    Rows and blocks past the last group are marked -1, so DeepGEMM and the grouped casts skip them.
    `padded_total_m` must cover every group rounded up to 128 rows; a device-side assert checks it.
    """
    assert offs.dim() == 1
    assert offs.dtype == torch.int32
    assert padded_total_m % GROUP_ALIGNMENT == 0
    starts_tensor = torch.cat((offs.new_zeros(1), offs[:-1]))[: offs.numel()]
    actual_ms_tensor = offs - starts_tensor
    aligned_ms_tensor = ((actual_ms_tensor + GROUP_ALIGNMENT - 1) // GROUP_ALIGNMENT) * GROUP_ALIGNMENT
    padded_ends = aligned_ms_tensor.cumsum(0)
    block_starts_tensor = (padded_ends - aligned_ms_tensor) // GROUP_ALIGNMENT
    ks_tensor = aligned_ms_tensor.contiguous()
    grouped_layout = torch.full((padded_total_m,), -1, dtype=torch.int32, device=offs.device)
    block_to_group = torch.full((padded_total_m // GROUP_ALIGNMENT,), -1, dtype=torch.int32, device=offs.device)
    if offs.numel():
        torch._assert_async(padded_ends[-1] <= padded_total_m)
        _build_grouped_layout_triton(
            grouped_layout,
            block_to_group,
            starts_tensor,
            actual_ms_tensor,
            aligned_ms_tensor,
            block_starts_tensor,
        )
    return (
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
def _per_token_fp8_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    rows,
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
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_offsets_i64 = row_offsets.to(tl.int64)
    col_offsets_i64 = col_offsets.to(tl.int64)
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
    x = tl.load(
        x_ptr + row_offsets_i64[:, None] * stride_xm + col_offsets_i64[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    scale = tl.math.div_rn(amax, FP8_MAX)
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    y = tl.clamp(tl.math.div_rn(x, scale[:, None]), FP8_MIN, FP8_MAX)
    tl.store(
        out_ptr + row_offsets_i64[:, None] * stride_ym + col_offsets_i64[None, :] * stride_yn,
        y.to(tl.float8e4nv),
        mask=mask,
    )
    tl.store(sf_ptr + row_offsets_i64 * stride_sm + pid_k * stride_sk, scale, mask=row_offsets < rows)


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
    block_group = tl.load(block_to_group_ptr + pid_blk)
    pid_g = tl.maximum(block_group, 0)
    src_start = tl.load(starts_ptr + pid_g)
    actual_m = tl.load(actual_ms_ptr + pid_g)
    block_start = tl.load(block_starts_ptr + pid_g)
    local_block = pid_blk - block_start
    row_offsets = local_block * GROUP_BLOCK_M + pid_sub * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    src_rows_i64 = (src_start + row_offsets).to(tl.int64)
    dst_rows_i64 = (pid_blk * GROUP_BLOCK_M + pid_sub * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    col_offsets_i64 = col_offsets.to(tl.int64)
    valid_rows = (row_offsets < actual_m) & (block_group >= 0)
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
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
    scale = amax * FP8_MAX_RECIPROCAL
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    y = x * (1.0 / scale)
    tl.store(
        out_ptr + pid_g * stride_yg + row_offsets[:, None] * stride_ym + col_offsets[None, :] * stride_yn,
        y.to(tl.float8e4nv),
        mask=mask,
    )
    tl.store(sf_ptr + pid_g * stride_sg + pid_m * stride_sm + pid_n * stride_sn, scale, mask=pid_g < groups)


# ---------------------------------------------------------------------------
# Public quantization functions
# ---------------------------------------------------------------------------


def per_token_cast_to_fp8_triton(
    x: torch.Tensor, use_ue8m0: bool, gran_k: int = GROUP_ALIGNMENT
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    assert gran_k == GROUP_ALIGNMENT
    rows, cols = x.shape
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sf = torch.empty((rows, ceil_div(cols, gran_k)), device=x.device, dtype=torch.float32)
    grid = lambda meta: (ceil_div(rows, meta["BLOCK_M"]), ceil_div(cols, meta["BLOCK_K"]))
    _per_token_fp8_kernel[grid](
        x,
        out,
        sf,
        rows,
        cols,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        sf.stride(0),
        sf.stride(1),
        USE_UE8M0=use_ue8m0,
        BLOCK_M=32,
        BLOCK_K=gran_k,
        num_warps=4,
    )
    return out, sf


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


@triton.jit
def _grouped_transpose_bytes_kernel(
    x_ptr,
    y_ptr,
    rows,
    cols,
    stride_xg,
    stride_xm,
    stride_xn,
    stride_yg,
    stride_ym,
    stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_g = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
    tile = tl.load(
        x_ptr + pid_g * stride_xg + row_offsets[:, None] * stride_xm + col_offsets[None, :] * stride_xn,
        mask=mask,
    )
    tl.store(
        y_ptr + pid_g * stride_yg + col_offsets[:, None] * stride_ym + row_offsets[None, :] * stride_yn,
        tl.trans(tile),
        mask=tl.trans(mask),
    )


def grouped_transpose_block_fp8_triton(x_fp8: torch.Tensor, sf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transpose a (G, M, N) 128x128 block-scaled FP8 tensor to (G, N, M), bytes and scales.

    A block's amax is invariant under transposition, so the result is bitwise the block cast of the
    transposed source.
    """
    assert x_fp8.dim() == 3 and x_fp8.dtype == torch.float8_e4m3fn
    groups, rows, cols = x_fp8.shape
    x_bytes = x_fp8.view(torch.uint8)
    out = torch.empty((groups, cols, rows), device=x_fp8.device, dtype=torch.uint8)
    block_m, block_n = 128, 128
    grid = (groups, ceil_div(rows, block_m), ceil_div(cols, block_n))
    _grouped_transpose_bytes_kernel[grid](
        x_bytes,
        out,
        rows,
        cols,
        x_bytes.stride(0),
        x_bytes.stride(1),
        x_bytes.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out.view(torch.float8_e4m3fn), sf.transpose(1, 2).contiguous()


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
    """Per-token fp8 cast of ``x.T`` without materializing the transpose."""
    assert x.dim() == 2
    assert gran_k == GROUP_ALIGNMENT
    rows, cols = x.shape
    out = torch.empty((cols, rows), device=x.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty((cols, ceil_div(rows, gran_k)), device=x.device, dtype=torch.float32)
    grid = lambda meta: (ceil_div(cols, meta["BLOCK_M"]), ceil_div(rows, meta["BLOCK_K"]))
    _per_token_fp8_kernel[grid](
        x,
        out,
        sf,
        cols,
        rows,
        # transposed read: the kernel's per-row amax reduces over x's rows
        x.stride(1),
        x.stride(0),
        out.stride(0),
        out.stride(1),
        sf.stride(0),
        sf.stride(1),
        USE_UE8M0=use_ue8m0,
        BLOCK_M=32,
        BLOCK_K=gran_k,
        num_warps=4,
    )
    return out, sf
