"""Blockwise fp8 cast launcher that writes into caller-supplied destinations."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    _grouped_per_block_fp8_kernel,
    ceil_div,
)


def grouped_per_block_cast_to_fp8(
    x: torch.Tensor,
    use_ue8m0: bool,
    *,
    out: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    assert (out is None) == (sf is None)
    groups, rows, cols = x.shape
    scale_rows, scale_cols = ceil_div(rows, GROUP_ALIGNMENT), ceil_div(cols, GROUP_ALIGNMENT)
    if out is None:
        out = torch.empty((groups, rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
        sf = torch.empty((groups, scale_rows, scale_cols), device=x.device, dtype=torch.float32)
    else:
        assert out.shape == (groups, rows, cols)
        assert out.dtype == torch.float8_e4m3fn
        assert out.device == x.device
        assert out.is_contiguous()
        assert sf.shape == (groups, scale_rows, scale_cols)
        assert sf.dtype == torch.float32
        assert sf.device == x.device
        assert sf.is_contiguous()
    grid = (groups, scale_rows, scale_cols)
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
        BLOCK_M=GROUP_ALIGNMENT,
        BLOCK_N=GROUP_ALIGNMENT,
        num_warps=8,
    )
    return out, sf


@triton.jit
def _grouped_per_block_fp8_both_layouts_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    out_t_ptr,
    sf_t_ptr,
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
    stride_ytg,
    stride_ytm,
    stride_ytn,
    stride_stg,
    stride_stm,
    stride_stn,
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
    mask_t = (pid_g < groups) & (col_offsets[:, None] < cols) & (row_offsets[None, :] < rows)
    x = tl.load(
        x_ptr + pid_g * stride_xg + row_offsets[:, None] * stride_xm + col_offsets[None, :] * stride_xn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x))
    scale = tl.maximum(amax / 448.0, 1e-4)
    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    y = x / scale
    tl.store(
        out_ptr + pid_g * stride_yg + row_offsets[:, None] * stride_ym + col_offsets[None, :] * stride_yn,
        y.to(tl.float8e4nv),
        mask=mask,
    )
    tl.store(sf_ptr + pid_g * stride_sg + pid_m * stride_sm + pid_n * stride_sn, scale, mask=pid_g < groups)
    # tl.trans(y) miscompiles on triton 3.7.1 when y also feeds the untransposed store.
    y_t = tl.trans(x) / scale
    tl.store(
        out_t_ptr + pid_g * stride_ytg + col_offsets[:, None] * stride_ytm + row_offsets[None, :] * stride_ytn,
        y_t.to(tl.float8e4nv),
        mask=mask_t,
    )
    tl.store(sf_t_ptr + pid_g * stride_stg + pid_n * stride_stm + pid_m * stride_stn, scale, mask=pid_g < groups)


def grouped_per_block_cast_to_fp8_both_layouts(
    x: torch.Tensor,
    use_ue8m0: bool,
    *,
    out: torch.Tensor | None = None,
    sf: torch.Tensor | None = None,
    out_t: torch.Tensor | None = None,
    sf_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One pass over ``x`` emitting both the row-major and the transposed blockwise fp8 layouts."""
    assert x.dim() == 3
    destinations = (out, sf, out_t, sf_t)
    assert all(tensor is None for tensor in destinations) or all(tensor is not None for tensor in destinations)
    groups, rows, cols = x.shape
    scale_rows, scale_cols = ceil_div(rows, GROUP_ALIGNMENT), ceil_div(cols, GROUP_ALIGNMENT)
    if out is None:
        out = torch.empty((groups, rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
        sf = torch.empty((groups, scale_rows, scale_cols), device=x.device, dtype=torch.float32)
        out_t = torch.empty((groups, cols, rows), device=x.device, dtype=torch.float8_e4m3fn)
        sf_t = torch.empty((groups, scale_cols, scale_rows), device=x.device, dtype=torch.float32)
    else:
        assert out.shape == (groups, rows, cols)
        assert out.dtype == torch.float8_e4m3fn
        assert out.device == x.device
        assert out.is_contiguous()
        assert sf.shape == (groups, scale_rows, scale_cols)
        assert sf.dtype == torch.float32
        assert sf.device == x.device
        assert sf.is_contiguous()
        assert out_t.shape == (groups, cols, rows)
        assert out_t.dtype == torch.float8_e4m3fn
        assert out_t.device == x.device
        assert out_t.is_contiguous()
        assert sf_t.shape == (groups, scale_cols, scale_rows)
        assert sf_t.dtype == torch.float32
        assert sf_t.device == x.device
        assert sf_t.is_contiguous()
    grid = (groups, scale_rows, scale_cols)
    _grouped_per_block_fp8_both_layouts_kernel[grid](
        x,
        out,
        sf,
        out_t,
        sf_t,
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
        out_t.stride(0),
        out_t.stride(1),
        out_t.stride(2),
        sf_t.stride(0),
        sf_t.stride(1),
        sf_t.stride(2),
        USE_UE8M0=use_ue8m0,
        BLOCK_M=GROUP_ALIGNMENT,
        BLOCK_N=GROUP_ALIGNMENT,
        num_warps=8,
    )
    return out, sf, out_t, sf_t
