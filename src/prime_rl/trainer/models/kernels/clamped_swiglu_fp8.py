"""Clamped SwiGLU between FP8 blockwise linears, fused with the FP8 casts around it.

Forward: `h = silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)` from the packed `[gate | up]` GEMM output,
plus its 1 x 128 per-token FP8 cast (the down projection's input). Backward: `d[gate | up]` straight into the two
casts the gate and up projections' GEMMs read, per token (data gradient) and transposed per 128 tokens (weight
gradient), without materializing the bf16 gradient.

Every value is rounded where eager PyTorch rounds it (bf16 after each op, IEEE `exp` and division, the FMA nvcc
emits in `silu_backward`), and the casts are those of `fp8_utils`, so the results are bitwise those of the unfused
ops.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from prime_rl.trainer.models.kernels.fp8_utils import _fp8_quantize, _fp8_scale, ceil_div

BLOCK = 128


@triton.jit
def _clamped_swiglu(g, u, limit):
    """bf16 `silu(clamp(g, max=limit)) * clamp(u, -limit, limit)` as fp32 values, with the clamped inputs."""
    g = tl.minimum(g, limit, propagate_nan=tl.PropagateNan.ALL).to(tl.bfloat16).to(tl.float32)
    u = tl.minimum(tl.maximum(u, -limit, propagate_nan=tl.PropagateNan.ALL), limit, propagate_nan=tl.PropagateNan.ALL)
    u = u.to(tl.bfloat16).to(tl.float32)
    s = libdevice.div_rn(g, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    return g, u, s


@triton.jit
def _clamped_swiglu_fwd_kernel(
    gu_ptr,
    h_ptr,
    q_ptr,
    sf_ptr,
    rows,
    limit,
    stride_sk,
    I: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row_offsets = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    col_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = row_offsets < rows
    gu_ptrs = gu_ptr + row_offsets[:, None] * (2 * I) + col_offsets[None, :]
    g = tl.load(gu_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    u = tl.load(gu_ptrs + I, mask=row_mask[:, None], other=0.0).to(tl.float32)
    _, u, s = _clamped_swiglu(g, u, limit)
    h = (s * u).to(tl.bfloat16)
    out_offsets = row_offsets[:, None] * I + col_offsets[None, :]
    tl.store(h_ptr + out_offsets, h, mask=row_mask[:, None])
    scale = _fp8_scale(tl.max(tl.abs(h), axis=1), USE_UE8M0)
    tl.store(q_ptr + out_offsets, _fp8_quantize(h.to(tl.float32), scale[:, None]), mask=row_mask[:, None])
    tl.store(sf_ptr + pid_k * stride_sk + row_offsets, scale, mask=row_mask)


@triton.jit
def _clamped_swiglu_bwd_kernel(
    gu_ptr,
    dh_ptr,
    q_ptr,
    sf_ptr,
    qt_ptr,
    sft_ptr,
    rows,
    padded_rows,
    sf_rows,
    limit,
    I: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One [BLOCK_M tokens, BLOCK_K channels] tile of d[gate] (programs `pid_k < I / BLOCK_K`) or d[up]. Outputs are
    stacked `[gate, up]`: `q [2, rows, I]` / `sf [2, I / 128, sf_rows]` per token, `qt [2, I, padded_rows]` /
    `sft [2, padded_rows / 128, I]` transposed; padded tokens are zeros, as in `per_token_cast_to_fp8_tp_triton`."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    NK: tl.constexpr = I // BLOCK_K
    is_up = pid_k >= NK
    kk = pid_k % NK
    row_offsets = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    col_offsets = kk * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = row_offsets < rows
    gu_ptrs = gu_ptr + row_offsets[:, None] * (2 * I) + col_offsets[None, :]
    g_raw = tl.load(gu_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float32)
    u_raw = tl.load(gu_ptrs + I, mask=row_mask[:, None], other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + row_offsets[:, None] * I + col_offsets[None, :], mask=row_mask[:, None], other=0.0)
    dh = dh.to(tl.float32)
    g, u, s = _clamped_swiglu(g_raw, u_raw, limit)
    if is_up:
        du = (dh * s).to(tl.bfloat16).to(tl.float32)
        d = tl.where((u_raw >= -limit) & (u_raw <= limit), du, 0.0)
    else:
        ds = (dh * u).to(tl.bfloat16).to(tl.float32)
        sig = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-g))
        dg = libdevice.mul_rn(libdevice.mul_rn(ds, sig), libdevice.fma(g, 1.0 - sig, 1.0))
        d = tl.where(g_raw <= limit, dg.to(tl.bfloat16).to(tl.float32), 0.0)
    d = d.to(tl.bfloat16).to(tl.float32)
    half = is_up.to(tl.int64)

    scale = _fp8_scale(tl.max(tl.abs(d), axis=1), USE_UE8M0)
    q_offsets = half * rows * I + row_offsets[:, None] * I + col_offsets[None, :]
    tl.store(q_ptr + q_offsets, _fp8_quantize(d, scale[:, None]), mask=row_mask[:, None])
    tl.store(sf_ptr + half * (I // BLOCK_K) * sf_rows + kk * sf_rows + row_offsets, scale, mask=row_mask)

    scale_t = _fp8_scale(tl.max(tl.abs(d), axis=0), USE_UE8M0)
    y = _fp8_quantize(d, scale_t[None, :])
    qt_offsets = half * I * padded_rows + col_offsets.to(tl.int64)[:, None] * padded_rows + row_offsets[None, :]
    tl.store(qt_ptr + qt_offsets, tl.trans(y))
    tl.store(sft_ptr + half * (padded_rows // BLOCK_M) * I + pid_m * I + col_offsets, scale_t)


def clamped_swiglu_fp8(
    gate_up: torch.Tensor, limit: float, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`h` (bf16 `[T, I]`) from the packed `[T, 2I]` gate/up, its per-token FP8 values and its scales as stored,
    `[I / 128, ceil(T / 4) * 4]` (DeepGEMM reads `sf[:, :T].T`)."""
    assert gate_up.dim() == 2 and gate_up.is_contiguous()
    rows, I = gate_up.shape[0], gate_up.shape[1] // 2
    assert I % BLOCK == 0
    h = torch.empty(rows, I, device=gate_up.device, dtype=torch.bfloat16)
    q = torch.empty(rows, I, device=gate_up.device, dtype=torch.float8_e4m3fn)
    sf = torch.empty(I // BLOCK, ceil_div(rows, 4) * 4, device=gate_up.device, dtype=torch.float32)
    block_m = 32
    _clamped_swiglu_fwd_kernel[(ceil_div(rows, block_m), I // BLOCK)](
        gate_up,
        h,
        q,
        sf,
        rows,
        limit,
        sf.stride(0),
        I=I,
        USE_UE8M0=use_ue8m0,
        BLOCK_M=block_m,
        BLOCK_K=BLOCK,
        num_warps=4,
    )
    return h, q, sf


def clamped_swiglu_fp8_backward(
    grad_h: torch.Tensor, gate_up: torch.Tensor, limit: float, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """d[gate | up] cast for the projections' GEMMs: per token (`q [2, T, I]`, scales `[2, I / 128, ceil(T / 4) * 4]`)
    and transposed (`qt [2, I, T padded to 128]`, scales `[2, T padded / 128, I]`)."""
    assert gate_up.is_contiguous() and grad_h.is_contiguous()
    rows, I = gate_up.shape[0], gate_up.shape[1] // 2
    padded_rows = ceil_div(rows, BLOCK) * BLOCK
    sf_rows = ceil_div(rows, 4) * 4
    dev = gate_up.device
    q = torch.empty(2, rows, I, device=dev, dtype=torch.float8_e4m3fn)
    sf = torch.empty(2, I // BLOCK, sf_rows, device=dev, dtype=torch.float32)
    qt = torch.empty(2, I, padded_rows, device=dev, dtype=torch.float8_e4m3fn)
    sft = torch.empty(2, padded_rows // BLOCK, I, device=dev, dtype=torch.float32)
    _clamped_swiglu_bwd_kernel[(padded_rows // BLOCK, 2 * I // BLOCK)](
        gate_up,
        grad_h,
        q,
        sf,
        qt,
        sft,
        rows,
        padded_rows,
        sf_rows,
        limit,
        I=I,
        USE_UE8M0=use_ue8m0,
        BLOCK_M=BLOCK,
        BLOCK_K=BLOCK,
        num_warps=8,
    )
    return q, sf, qt, sft
