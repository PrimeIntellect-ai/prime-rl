"""E4M3 values with one RCEIL UE8M0 scale per row and 32 columns."""

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 32


@triton.jit
def _stochastic_round(x, seed, index):
    magnitude = tl.minimum(tl.abs(x), 448.0, propagate_nan=tl.PropagateNan.ALL)
    exponent = ((magnitude.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    spacing_exponent = tl.maximum(exponent - 3, -9)
    spacing = ((spacing_exponent + 127) << 23).to(tl.float32, bitcast=True)
    inverse_spacing = ((127 - spacing_exponent) << 23).to(tl.float32, bitcast=True)
    units = magnitude * inverse_spacing
    lower_units = tl.floor(units)
    rounded = (lower_units + (tl.rand(seed, index.to(tl.uint64)) < units - lower_units).to(tl.float32)) * spacing
    sign = x.to(tl.int32, bitcast=True) & -2147483648
    return (rounded.to(tl.int32, bitcast=True) | sign).to(tl.float32, bitcast=True).to(tl.float8e4nv)


@triton.jit
def _quantize_kernel(
    X,
    Q,
    S,
    SEED,
    SALT,
    rounding_offset,
    rows,
    cols,
    stride_g,
    stride_m,
    stride_k,
    STOCHASTIC: tl.constexpr,
    TILE_M: tl.constexpr,
):
    group = tl.program_id(2).to(tl.int64)
    m = tl.program_id(0) * TILE_M + tl.arange(0, TILE_M)
    k = tl.program_id(1) * 32 + tl.arange(0, 32)
    mask = m[:, None] < rows
    x = tl.load(X + group * stride_g + m[:, None].to(tl.int64) * stride_m + k[None, :] * stride_k, mask, 0)
    x = x.to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.where(tl.sum((x != x).to(tl.int32), axis=1) > 0, float("nan"), amax)
    scale = tl.inline_asm_elementwise(
        "cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
        constraints="=h,r",
        args=[amax * (1.0 / 448.0)],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
    # Match the online serving cast's zero-scale convention.
    reciprocal = ((254 - scale.to(tl.int32)) << 23).to(tl.float32, bitcast=True)
    reciprocal = tl.where(scale == 0, 0.0, reciprocal)
    reciprocal = tl.where(scale == 255, float("nan"), reciprocal)
    offset = (group * rows + m[:, None].to(tl.int64)) * cols + k[None, :]
    scaled = x * reciprocal[:, None]
    if STOCHASTIC:
        seed = tl.load(SEED).to(tl.uint64) ^ tl.load(SALT).to(tl.uint64)
        q = _stochastic_round(scaled, seed, offset + rounding_offset)
    else:
        q = scaled.to(tl.float8e4nv)
    tl.store(Q + offset, q, mask)
    tl.store(S + (group * rows + m.to(tl.int64)) * (cols // 32) + tl.program_id(1), scale, m < rows)


@triton.jit
def _dequantize_kernel(Q, S, OUT, size, TILE: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    q = tl.load(Q + offset, offset < size, 0.0).to(tl.float32)
    scale = tl.load(S + offset // 32, offset < size, 0).to(tl.int32)
    decoded = (scale << 23).to(tl.float32, bitcast=True)
    decoded = tl.where(scale == 0, 2.0**-127, decoded)
    decoded = tl.where(scale == 255, float("nan"), decoded)
    tl.store(OUT + offset, q * decoded, offset < size)


def quantize(
    x: torch.Tensor,
    *,
    rounding_seed: torch.Tensor | None = None,
    rounding_salt: torch.Tensor | None = None,
    rounding_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 and UE8M0 scales, optionally using a Philox stream for weight rounding.

    The stream is keyed by seed XOR salt and the logical row-major element index
    plus offset. Supplying the same stream reproduces forward rounding in backward.
    """
    if x.dtype != torch.bfloat16 or x.ndim not in (2, 3) or x.shape[-1] % BLOCK_SIZE:
        raise ValueError("MXFP8 requires 2D/3D BF16 input with its last dimension divisible by 32")
    if rounding_seed is not None:
        for key in (rounding_seed, rounding_salt):
            if key is None or key.dtype != torch.int64 or key.ndim != 0 or key.device != x.device:
                raise ValueError("Stochastic rounding requires scalar int64 seed and salt on the input device")
    rows, cols = x.shape[-2:]
    groups = x.shape[0] if x.ndim == 3 else 1
    q = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((*x.shape[:-1], cols // BLOCK_SIZE), device=x.device, dtype=torch.uint8)
    if x.numel():
        _quantize_kernel[(triton.cdiv(rows, 16), cols // BLOCK_SIZE, groups)](
            x,
            q,
            scales,
            rounding_seed,
            rounding_salt,
            rounding_offset,
            rows,
            cols,
            x.stride(0) if x.ndim == 3 else 0,
            x.stride(-2),
            x.stride(-1),
            STOCHASTIC=rounding_seed is not None,
            TILE_M=16,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, scales


def dequantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode contiguous MXFP8 payload and scales into BF16."""
    if q.dtype != torch.float8_e4m3fn or scales.dtype != torch.uint8:
        raise ValueError("Expected E4M3 payload and uint8 UE8M0 scales")
    if q.shape[-1] % BLOCK_SIZE or scales.shape != (*q.shape[:-1], q.shape[-1] // BLOCK_SIZE):
        raise ValueError("MXFP8 payload and scale shapes do not match")
    if not q.is_contiguous() or not scales.is_contiguous():
        raise ValueError("MXFP8 dequantization requires contiguous payload and scales")
    out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    if q.numel():
        _dequantize_kernel[(triton.cdiv(q.numel(), 1024),)](q, scales, out, q.numel(), TILE=1024)
    return out
