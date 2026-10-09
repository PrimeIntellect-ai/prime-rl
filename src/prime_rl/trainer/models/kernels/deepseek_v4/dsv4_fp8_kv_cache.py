"""Quantize-dequantize round trip through vLLM's `fp8_ds_mla` KV cache for DeepSeek-V4.

vLLM stores each 512-wide KV vector as 448 FP8 e4m3 channels, one UE8M0 (power of two) scale per
64-channel tile, and the 64 trailing rope channels in bf16, and its attention reads keys and values only
from that cache. `dsv4_fp8_kv_round_trip` reproduces the values the cache hands back, with a
straight-through gradient, so the trainer can score tokens against the same keys and values.

The two cache writers differ only in where the scale floor sits:

- the sliding-window insert floors the tile's absolute maximum: `scale = 2^ceil(log2(max(amax, 1e-4) / 448))`
- the compressor insert floors the scale itself: `scale = 2^ceil(log2(max(amax / 448, 1e-4)))`
"""

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
QUANT_TILE = 64
SWA_KV_MIN_AMAX = 1e-4
COMPRESSED_KV_MIN_SCALE = 1e-4


@triton.jit
def _fp8_kv_round_trip_kernel(
    X,
    OUT,
    head_dim: tl.constexpr,
    nope_tiles: tl.constexpr,
    min_amax: tl.constexpr,
    min_scale: tl.constexpr,
    fp8_max: tl.constexpr,
    TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """Round-trip the first `nope_tiles` tiles of one row through FP8 and copy the rest."""
    row = tl.program_id(0).to(tl.int64)
    tiles = tl.arange(0, NUM_TILES)
    offsets = tiles[:, None] * TILE + tl.arange(0, TILE)[None, :]
    mask = offsets < head_dim
    x = tl.load(X + row * head_dim + offsets, mask=mask, other=0.0).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(x), axis=1), min_amax)
    scale_raw = tl.maximum(tl.math.div_rn(amax, fp8_max), min_scale)
    biased_exponent = ((scale_raw.to(tl.uint32, bitcast=True) + 0x7FFFFF) >> 23) & 0xFF
    scale = (biased_exponent << 23).to(tl.float32, bitcast=True)
    inv_scale = ((254 - biased_exponent) << 23).to(tl.float32, bitcast=True)

    quantized = tl.clamp(x * inv_scale[:, None], -fp8_max, fp8_max).to(tl.float8e4nv)
    round_trip = quantized.to(tl.float32) * scale[:, None]
    y = tl.where((tiles < nope_tiles)[:, None], round_trip, x)
    tl.store(OUT + row * head_dim + offsets, y.to(OUT.dtype.element_ty), mask=mask)


@torch.library.custom_op("prime_rl::dsv4_fp8_kv_round_trip", mutates_args=())
def dsv4_fp8_kv_round_trip(x: torch.Tensor, rope_dim: int, min_amax: float, min_scale: float) -> torch.Tensor:
    """The value vLLM's `fp8_ds_mla` cache returns for `x`, in `x`'s dtype.

    Each `QUANT_TILE`-wide tile of the leading `head_dim - rope_dim` channels becomes
    `e4m3(x / scale) * scale` with `scale = 2^ceil(log2(max(max(amax, min_amax) / 448, min_scale)))`,
    rounding to nearest even. The trailing `rope_dim` channels pass through.

    Args:
        x: `(..., head_dim)`, `head_dim - rope_dim` a multiple of `QUANT_TILE`.
        rope_dim: trailing channels the cache stores unquantized.
        min_amax: floor on each tile's absolute maximum.
        min_scale: floor on each tile's scale before it is rounded up to a power of two.
    """
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim
    assert nope_dim % QUANT_TILE == 0, f"{nope_dim=} is not a multiple of {QUANT_TILE}"
    assert min_amax > 0 or min_scale > 0, "an all-zero tile needs a positive floor"
    x = x.contiguous()
    out = torch.empty_like(x)
    rows = x.numel() // head_dim
    if rows == 0:
        return out
    _fp8_kv_round_trip_kernel[(rows,)](
        x,
        out,
        head_dim,
        nope_dim // QUANT_TILE,
        min_amax,
        min_scale,
        FP8_MAX,
        TILE=QUANT_TILE,
        NUM_TILES=triton.next_power_of_2(triton.cdiv(head_dim, QUANT_TILE)),
    )
    return out


@dsv4_fp8_kv_round_trip.register_fake
def _dsv4_fp8_kv_round_trip_fake(x: torch.Tensor, rope_dim: int, min_amax: float, min_scale: float) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _dsv4_fp8_kv_round_trip_autograd_backward(ctx, grad: torch.Tensor):
    return grad, None, None, None


dsv4_fp8_kv_round_trip.register_autograd(_dsv4_fp8_kv_round_trip_autograd_backward)


def dsv4_fp8_swa_kv_round_trip(kv: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """`kv` as vLLM's sliding-window cache insert stores it, with the amax floored at `SWA_KV_MIN_AMAX`."""
    return dsv4_fp8_kv_round_trip(kv, rope_dim, SWA_KV_MIN_AMAX, 0.0)


def dsv4_fp8_compressed_kv_round_trip(compressed_kv: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """`compressed_kv` as vLLM's compressor stores it, with the scale floored at `COMPRESSED_KV_MIN_SCALE`."""
    return dsv4_fp8_kv_round_trip(compressed_kv, rope_dim, 0.0, COMPRESSED_KV_MIN_SCALE)
