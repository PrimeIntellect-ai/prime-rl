"""DeepSeek V4.1's compressed KV entries as vLLM's ``nvfp4_ds_mla`` cache stores them.

vLLM 0.33 writes each compressed entry (``vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py``,
``_rope_quant_insert_nvfp4_kernel``) as a 288-byte record: the 512 channels, RoPE included, in 32 tiles of 16, each
tile ``e2m1(x / s)`` with one E4M3 scale ``s = e4m3(clamp(amax * fp32(1 / 6), 2^-9, 448))`` and no per-tensor scale
on top. The division is IEEE round-to-nearest (``div_rn``) and the E2M1 cast rounds to nearest even, saturating at 6
(``cvt.rn.satfinite.e2m1x2``). FlashMLA's mega-attention kernel reads ``e2m1 * s`` back. The sliding-window record
stays V4.1's MXFP8 one (``dsv4_kv_fake_quant`` with 32-wide tiles).
"""

import torch

NVFP4_TILE = 16
E4M3_MIN_NORMAL = 2.0**-9
E4M3_MAX = 448.0
# Triton's ``amax * (1.0 / 6.0)`` multiplies by the float32 constant, which is not the same as dividing by 6
ONE_SIXTH = torch.tensor(1.0 / 6.0, dtype=torch.float32).item()


def e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """``x`` (float32) rounded to the nearest E2M1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6), ties to the even code, and
    saturated at 6, as ``cvt.rn.satfinite.e2m1x2`` does."""
    a = x.abs()
    # Interval edges; a tie goes to the neighbour with an even code (0, 1, 2, 4 have even codes; 0.5, 1.5, 3, 6 odd)
    out = torch.full_like(a, 6.0)
    out = torch.where(a <= 5.0, 4.0, out)
    out = torch.where(a < 3.5, 3.0, out)
    out = torch.where(a <= 2.5, 2.0, out)
    out = torch.where(a < 1.75, 1.5, out)
    out = torch.where(a <= 1.25, 1.0, out)
    out = torch.where(a < 0.75, 0.5, out)
    out = torch.where(a <= 0.25, 0.0, out)
    return torch.copysign(out, x)


def nvfp4_ds_mla_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The record's E2M1 values (as float32) and per-16 E4M3 scales for ``x`` (``(..., 512)``)."""
    head_dim = x.shape[-1]
    assert head_dim % NVFP4_TILE == 0, f"{head_dim=} is not a multiple of {NVFP4_TILE}"
    tiles = x.float().reshape(*x.shape[:-1], head_dim // NVFP4_TILE, NVFP4_TILE)
    amax = tiles.abs().amax(dim=-1, keepdim=True)
    scale = (amax * ONE_SIXTH).clamp(E4M3_MIN_NORMAL, E4M3_MAX).to(torch.float8_e4m3fn)
    values = e2m1_round(tiles / scale.float())
    return values, scale


class _StraightThrough(torch.autograd.Function):
    """Returns the rounded tensor as is and passes the gradient through. ``x + (rounded - x).detach()`` would not:
    in BF16 the difference is inexact once rounding saturates (``x`` many times the record's largest value)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, rounded: torch.Tensor) -> torch.Tensor:
        return rounded

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad, None


def dsv41_nvfp4_kv_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """The value vLLM's ``nvfp4_ds_mla`` compressed cache returns for ``x``, in ``x``'s dtype, with a
    straight-through gradient. E2M1 times an E4M3 scale has at most four significant bits, so BF16 holds it exactly."""
    with torch.no_grad():
        values, scale = nvfp4_ds_mla_quant(x)
        rounded = (values * scale.float()).reshape(x.shape).to(x.dtype)
    return _StraightThrough.apply(x, rounded)
