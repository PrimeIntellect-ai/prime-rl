"""Fake quantization (quantize, then dequantize) of DeepSeek-V4 keys and values as vLLM's `fp8_ds_mla` KV cache stores them.

vLLM stores each 512-wide KV vector as 448 FP8 e4m3 channels, one UE8M0 (power of two) scale per
64-channel tile, and the 64 trailing rope channels in bf16, and its attention reads keys and values only
from that cache. `dsv4_kv_fake_quant` reproduces the values the cache hands back, with a
straight-through gradient, so the trainer can score tokens against the same keys and values.

The two cache writers differ only in where the scale floor sits:

- the sliding-window insert floors the tile's absolute maximum: `scale = 2^ceil(log2(max(amax, 1e-4) / 448))`
- the compressor insert floors the scale itself: `scale = 2^ceil(log2(max(amax / 448, 1e-4)))`

With an NVFP4 compressed KV cache (prime-rl's `nvfp4_ds_mla`-style record for the ratio-4 compressor),
`dsv4_nvfp4_kv_fake_quant` reproduces what sparse decode reads: the NVFP4 record, requantized to the
fp8 record FlashMLA's decode kernel takes.
"""

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
QUANT_TILE = 64
VLLM_SWA_KV_AMAX_FLOOR = 1e-4
VLLM_COMPRESSED_KV_SCALE_FLOOR = 1e-4
NVFP4_GROUP = 16
NVFP4_MIN_SCALE = 0.001953125  # 2^-9


@triton.jit
def _kv_fake_quant_kernel(
    X,
    OUT,
    head_dim: tl.constexpr,
    nope_tiles: tl.constexpr,
    amax_floor: tl.constexpr,
    scale_floor: tl.constexpr,
    fp8_max: tl.constexpr,
    TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    """Fake-quantize the first `nope_tiles` tiles of one row to FP8 and copy the rest."""
    row = tl.program_id(0).to(tl.int64)
    tiles = tl.arange(0, NUM_TILES)
    offsets = tiles[:, None] * TILE + tl.arange(0, TILE)[None, :]
    mask = offsets < head_dim
    x = tl.load(X + row * head_dim + offsets, mask=mask, other=0.0).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(x), axis=1), amax_floor)
    scale_raw = tl.maximum(tl.math.div_rn(amax, fp8_max), scale_floor)
    biased_exponent = ((scale_raw.to(tl.uint32, bitcast=True) + 0x7FFFFF) >> 23) & 0xFF
    scale = (biased_exponent << 23).to(tl.float32, bitcast=True)
    inv_scale = ((254 - biased_exponent) << 23).to(tl.float32, bitcast=True)

    quantized = tl.clamp(x * inv_scale[:, None], -fp8_max, fp8_max).to(tl.float8e4nv)
    dequantized = quantized.to(tl.float32) * scale[:, None]
    y = tl.where((tiles < nope_tiles)[:, None], dequantized, x)
    tl.store(OUT + row * head_dim + offsets, y.to(OUT.dtype.element_ty), mask=mask)


@torch.library.custom_op("prime_rl::dsv4_kv_fake_quant", mutates_args=())
def dsv4_kv_fake_quant(x: torch.Tensor, rope_dim: int, amax_floor: float, scale_floor: float) -> torch.Tensor:
    """The value vLLM's `fp8_ds_mla` cache returns for `x`, in `x`'s dtype.

    Each `QUANT_TILE`-wide tile of the leading `head_dim - rope_dim` channels becomes
    `e4m3(x / scale) * scale` with `scale = 2^ceil(log2(max(max(amax, amax_floor) / 448, scale_floor)))`,
    rounding to nearest even. The trailing `rope_dim` channels pass through.

    Args:
        x: `(..., head_dim)`, `head_dim - rope_dim` a multiple of `QUANT_TILE`.
        rope_dim: trailing channels the cache stores unquantized.
        amax_floor: floor on each tile's absolute maximum.
        scale_floor: floor on each tile's scale before it is rounded up to a power of two.
    """
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim
    assert nope_dim % QUANT_TILE == 0, f"{nope_dim=} is not a multiple of {QUANT_TILE}"
    assert amax_floor > 0 or scale_floor > 0, "an all-zero tile needs a positive floor"
    x = x.contiguous()
    out = torch.empty_like(x)
    rows = x.numel() // head_dim
    if rows == 0:
        return out
    _kv_fake_quant_kernel[(rows,)](
        x,
        out,
        head_dim,
        nope_dim // QUANT_TILE,
        amax_floor,
        scale_floor,
        FP8_MAX,
        TILE=QUANT_TILE,
        NUM_TILES=triton.next_power_of_2(triton.cdiv(head_dim, QUANT_TILE)),
    )
    return out


@dsv4_kv_fake_quant.register_fake
def _dsv4_kv_fake_quant_fake(x: torch.Tensor, rope_dim: int, amax_floor: float, scale_floor: float) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _dsv4_kv_fake_quant_autograd_backward(ctx, grad: torch.Tensor):
    return grad, None, None, None


dsv4_kv_fake_quant.register_autograd(_dsv4_kv_fake_quant_autograd_backward)


def dsv4_swa_kv_fake_quant(kv: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """`kv` as vLLM's sliding-window cache insert stores it, with the amax floored at `VLLM_SWA_KV_AMAX_FLOOR`."""
    return dsv4_kv_fake_quant(kv, rope_dim, VLLM_SWA_KV_AMAX_FLOOR, 0.0)


def dsv4_compressed_kv_fake_quant(compressed_kv: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """`compressed_kv` as vLLM's compressor stores it, with the scale floored at `VLLM_COMPRESSED_KV_SCALE_FLOOR`."""
    return dsv4_kv_fake_quant(compressed_kv, rope_dim, 0.0, VLLM_COMPRESSED_KV_SCALE_FLOOR)


@triton.jit
def _nvfp4_kv_fake_quant_kernel(
    X,
    OUT,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    GROUP: tl.constexpr,
    TILE: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """One row as decode reads it from the NVFP4 cache: NoPE e2m1 with an e4m3 scale per `GROUP`, requantized to e4m3
    with a power-of-two scale per `TILE`; RoPE as unscaled e4m3. The expressions mirror the cache's writer and its
    fp8 conversion for decode, so the rounding (including approximate division, log2 and exp2) matches."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, WIDTH)
    x = tl.load(X + row * head_dim + col, mask=col < head_dim, other=0.0).to(tl.float32)

    groups = tl.reshape(x, (WIDTH // GROUP, GROUP))
    amax = tl.max(tl.abs(groups), axis=1)
    scale = tl.clamp(amax / 6.0, 0.001953125, 448.0).to(tl.float8e4nv).to(tl.float32)
    a = tl.abs(groups / scale[:, None])
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    exponent = code >> 1
    mantissa = code & 1
    bits = tl.where(exponent == 0, mantissa * 0x3F000000, ((126 + exponent) << 23) | (mantissa << 22))
    magnitude = bits.to(tl.float32, bitcast=True)
    nvfp4 = tl.where(groups < 0, -magnitude, magnitude) * scale[:, None]

    tiles = tl.reshape(nvfp4, (WIDTH // TILE, TILE))
    tile_exponent = tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(tiles), axis=1), 1e-4) / 448.0))
    fp8 = tl.clamp(tiles * tl.exp2(-tile_exponent)[:, None], -448.0, 448.0).to(tl.float8e4nv)
    nope = tl.reshape(fp8.to(tl.float32) * tl.exp2(tile_exponent)[:, None], (WIDTH,))

    rope = tl.clamp(x, -448.0, 448.0).to(tl.float8e4nv).to(tl.float32)
    y = tl.where(col < nope_dim, nope, rope)
    tl.store(OUT + row * head_dim + col, y.to(OUT.dtype.element_ty), mask=col < head_dim)


@torch.library.custom_op("prime_rl::dsv4_nvfp4_kv_fake_quant", mutates_args=())
def dsv4_nvfp4_kv_fake_quant(x: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """The value sparse decode reads for `x` from an NVFP4 compressed KV cache, in `x`'s dtype.

    The leading `head_dim - rope_dim` channels are stored as e2m1 with one e4m3 scale
    `e4m3(clamp(amax / 6, 2^-9, 448))` per `NVFP4_GROUP` channels, then requantized for decode to e4m3 with
    one power-of-two scale `2^ceil(log2(max(amax, 1e-4) / 448))` per `QUANT_TILE` channels; the trailing
    `rope_dim` channels are stored as unscaled e4m3.

    Args:
        x: `(..., head_dim)`, `head_dim` a power of two and `head_dim - rope_dim` a multiple of `QUANT_TILE`.
        rope_dim: trailing channels stored as unscaled e4m3.
    """
    head_dim = x.shape[-1]
    assert head_dim == triton.next_power_of_2(head_dim), f"{head_dim=} is not a power of two"
    assert (head_dim - rope_dim) % QUANT_TILE == 0, f"{head_dim - rope_dim=} is not a multiple of {QUANT_TILE}"
    x = x.contiguous()
    out = torch.empty_like(x)
    rows = x.numel() // head_dim
    if rows == 0:
        return out
    _nvfp4_kv_fake_quant_kernel[(rows,)](
        x, out, head_dim, head_dim - rope_dim, GROUP=NVFP4_GROUP, TILE=QUANT_TILE, WIDTH=head_dim, num_warps=4
    )
    return out


@dsv4_nvfp4_kv_fake_quant.register_fake
def _dsv4_nvfp4_kv_fake_quant_fake(x: torch.Tensor, rope_dim: int) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _dsv4_nvfp4_kv_fake_quant_autograd_backward(ctx, grad: torch.Tensor):
    return grad, None


dsv4_nvfp4_kv_fake_quant.register_autograd(_dsv4_nvfp4_kv_fake_quant_autograd_backward)
