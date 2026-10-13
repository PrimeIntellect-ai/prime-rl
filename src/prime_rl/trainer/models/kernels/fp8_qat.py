"""vLLM block weight casting with shared FP8 stochastic rounding."""

import torch
import triton
import triton.language as tl

from prime_rl.trainer.models.kernels.fp8_utils import _stochastic_round_fp8


@triton.jit
def _weight_quant_kernel(
    X,
    OUT,
    SCALES,
    rounding_seed,
    rounding_salt,
    rounding_offset,
    stride_xg,
    rows,
    cols,
    stride_xm,
    stride_xn,
    USE_UE8M0: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    DEQUANTIZE: tl.constexpr,
    HAS_SALT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_n = tl.program_id(1)
    group = tl.program_id(2).to(tl.int64)
    X += group * stride_xg
    OUT += group * rows * cols
    SCALES += group * tl.cdiv(rows, BLOCK) * tl.cdiv(cols, BLOCK)
    m = block_m * BLOCK + tl.arange(0, BLOCK)
    n = block_n * BLOCK + tl.arange(0, BLOCK)
    mask = (m[:, None] < rows) & (n[None, :] < cols)
    x = tl.load(X + m[:, None] * stride_xm + n[None, :] * stride_xn, mask, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
    scale = amax * (1.0 / 448.0)
    if STOCHASTIC:
        stream = tl.load(rounding_seed)
        if HAS_SALT:
            stream ^= tl.load(rounding_salt)
        # Smaller random-generation tiles keep Philox state in registers.
        for sub in range(4):
            sm = block_m * BLOCK + sub * 32 + tl.arange(0, 32)
            sub_mask = (sm[:, None] < rows) & (n[None, :] < cols)
            sx = tl.load(X + sm[:, None] * stride_xm + n[None, :] * stride_xn, sub_mask, other=0).to(tl.float32)
            scaled = sx * tl.div_rn(1.0, scale)
            counter = (group * rows + sm[:, None]) * cols + n[None, :] + rounding_offset
            rounded = _stochastic_round_fp8(scaled, stream, counter)
            tl.store(OUT + sm[:, None] * cols + n[None, :], rounded, sub_mask)
        tl.store(SCALES + block_m * tl.cdiv(cols, BLOCK) + block_n, scale)
        return
    scaled = x * tl.div_rn(1.0, scale)
    nearest = scaled.to(tl.float8e4nv)
    if USE_UE8M0:
        # vLLM online casts first, then DeepGEMM requantizes the decoded weights.
        decoded = nearest.to(tl.float32) * scale
        amax = tl.maximum(tl.max(tl.abs(decoded)), 1e-4)
        scale = tl.exp2(tl.ceil(tl.log2(amax * (1.0 / 448.0))))
        nearest = (decoded * tl.div_rn(1.0, scale)).to(tl.float8e4nv)
    result = nearest.to(tl.float32) * scale if DEQUANTIZE else nearest
    tl.store(OUT + m[:, None] * cols + n[None, :], result, mask)
    tl.store(SCALES + block_m * tl.cdiv(cols, BLOCK) + block_n, scale)


@triton.jit
def _requantize_weight_kernel(
    Q, SCALES, OUT, OUT_SCALES, rows, cols, USE_UE8M0: tl.constexpr, DEQUANTIZE: tl.constexpr
):
    bm, bn = tl.program_id(0), tl.program_id(1)
    group = tl.program_id(2).to(tl.int64)
    m, n = bm * 128 + tl.arange(0, 128), bn * 128 + tl.arange(0, 128)
    offset = group * rows * cols + m[:, None] * cols + n[None, :]
    scale_offset = group * tl.cdiv(rows, 128) * tl.cdiv(cols, 128) + bm * tl.cdiv(cols, 128) + bn
    mask = (m[:, None] < rows) & (n[None, :] < cols)
    q = tl.load(Q + offset, mask, other=0.0).to(tl.float32)
    scale = tl.load(SCALES + scale_offset)
    decoded = q * scale
    if USE_UE8M0:
        amax = tl.maximum(tl.max(tl.abs(decoded)), 1e-4)
        scale = tl.exp2(tl.ceil(tl.log2(amax * (1.0 / 448.0))))
        q = (decoded * tl.div_rn(1.0, scale)).to(tl.float8e4nv).to(tl.float32)
    result = q * scale if DEQUANTIZE else q
    tl.store(OUT + offset, result, mask)
    tl.store(OUT_SCALES + scale_offset, scale)


def cast_qat_weight(
    weight: torch.Tensor,
    use_ue8m0: bool,
    *,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
    rounding_offset: int = 0,
    dequantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match vLLM online casting and optional DeepGEMM UE8M0 requantization.

    The seed, parameter salt, and global element offset use the shared FP8
    rounding contract. FSDP calls after gathering complete quantization blocks.
    """
    assert weight.ndim in (2, 3)
    rows, cols = weight.shape[-2:]
    out = torch.empty(weight.shape, device=weight.device, dtype=torch.bfloat16 if dequantize else torch.float8_e4m3fn)
    scales = torch.empty(
        (*weight.shape[:-2], triton.cdiv(rows, 128), triton.cdiv(cols, 128)),
        device=weight.device,
        dtype=torch.float32,
    )
    groups = weight.shape[0] if weight.ndim == 3 else 1
    stochastic_rounding = rounding_seed is not None
    needs_requant = stochastic_rounding and (use_ue8m0 or dequantize)
    first = (
        torch.empty_like(weight, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format)
        if needs_requant
        else out
    )
    first_scales = torch.empty_like(scales) if needs_requant else scales
    _weight_quant_kernel[(triton.cdiv(rows, 128), triton.cdiv(cols, 128), groups)](
        weight,
        first,
        first_scales,
        rounding_seed,
        rounding_salt,
        rounding_offset,
        weight.stride(0) if weight.ndim == 3 else 0,
        rows,
        cols,
        weight.stride(-2),
        weight.stride(-1),
        USE_UE8M0=use_ue8m0,
        STOCHASTIC=stochastic_rounding,
        DEQUANTIZE=dequantize,
        HAS_SALT=rounding_salt is not None,
        BLOCK=128,
        num_warps=8,
        enable_fp_fusion=False,
    )
    if needs_requant:
        _requantize_weight_kernel[(triton.cdiv(rows, 128), triton.cdiv(cols, 128), groups)](
            first,
            first_scales,
            out,
            scales,
            rows,
            cols,
            USE_UE8M0=use_ue8m0,
            DEQUANTIZE=dequantize,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return out, scales
