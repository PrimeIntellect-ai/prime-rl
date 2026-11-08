from __future__ import annotations

import torch
import triton
import triton.language as tl

from prime_rl.trainer.models.kernels.fp8_utils import FP8_MAX, _per_token_fp8_kernel, ceil_div

MX_BLOCK = 32


def mx_cast(x: torch.Tensor, along_rows: bool) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    rows, cols = x.shape
    out = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=x.device)
    if along_rows:
        assert rows % MX_BLOCK == 0
        sf = torch.empty((cols, rows // MX_BLOCK), dtype=torch.float32, device=x.device)
        logical = (cols, rows)
        x_strides, out_strides = (x.stride(1), x.stride(0)), (out.stride(1), out.stride(0))
    else:
        assert cols % MX_BLOCK == 0
        sf = torch.empty((rows, cols // MX_BLOCK), dtype=torch.float32, device=x.device)
        logical = (rows, cols)
        x_strides, out_strides = (x.stride(0), x.stride(1)), (out.stride(0), out.stride(1))
    grid = lambda meta: (ceil_div(logical[0], meta["BLOCK_M"]), ceil_div(logical[1], meta["BLOCK_K"]))  # noqa: E731
    _per_token_fp8_kernel[grid](
        x,
        out,
        sf,
        logical[0],
        logical[1],
        *x_strides,
        *out_strides,
        sf.stride(0),
        sf.stride(1),
        USE_UE8M0=True,
        BLOCK_M=32,
        BLOCK_K=MX_BLOCK,
        num_warps=4,
    )
    return out, sf


def pack_ue8m0_k_major(sf: torch.Tensor) -> torch.Tensor:
    assert sf.dtype == torch.float32 and sf.size(-1) % 4 == 0
    return (sf.contiguous().view(torch.int32) >> 23).to(torch.uint8).view(torch.int32)


def mx_weight_sf(sf: torch.Tensor, mn: int, k: int) -> torch.Tensor:
    import deep_gemm
    from deep_gemm.mega import _transpose_sf_for_utccp

    num_groups = sf.shape[0]
    return _transpose_sf_for_utccp(deep_gemm.transform_sf_into_required_layout(sf, mn, k, (1, MX_BLOCK), num_groups))


@triton.jit
def _mx_cast_rows_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    rows,
    cols,
    stride_xe,
    stride_xr,
    stride_xc,
    stride_oe,
    stride_or,
    stride_oc,
    stride_se,
    stride_sc,
    stride_sk,
    FP8_MAX: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    e = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_r = tl.program_id(axis=2)
    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r64, c64 = r.to(tl.int64), c.to(tl.int64)
    mask = (r[:, None] < rows) & (c[None, :] < cols)
    x = tl.load(x_ptr + e * stride_xe + r64[:, None] * stride_xr + c64[None, :] * stride_xc, mask=mask, other=0.0).to(
        tl.float32
    )
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-10)
    scale = tl.exp2(tl.ceil(tl.log2(tl.math.div_rn(amax, FP8_MAX))))
    y = tl.clamp(tl.math.div_rn(x, scale[None, :]), -FP8_MAX, FP8_MAX)
    tl.store(
        out_ptr + e * stride_oe + r64[:, None] * stride_or + c64[None, :] * stride_oc, y.to(tl.float8e4nv), mask=mask
    )
    tl.store(sf_ptr + e * stride_se + c64 * stride_sc + pid_r * stride_sk, scale, mask=c < cols)


def cast_weight_along_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts, rows, cols = w.shape
    assert rows % MX_BLOCK == 0
    out = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    sf = torch.empty((num_experts, cols, rows // MX_BLOCK), dtype=torch.float32, device=w.device)
    block_c = 128
    grid = (num_experts, ceil_div(cols, block_c), rows // MX_BLOCK)
    _mx_cast_rows_kernel[grid](
        w,
        out,
        sf,
        rows,
        cols,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sf.stride(0),
        sf.stride(1),
        sf.stride(2),
        FP8_MAX=FP8_MAX,
        BLOCK_R=MX_BLOCK,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out, sf
