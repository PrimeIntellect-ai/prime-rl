"""Simulate vLLM's FP8 KV cache on the trainer side so both see the same degraded K/V.

vLLM stores each cached K/V row with the leading nope channels in `float8_e4m3fn`, one
power-of-two scale per 64-wide block per token, and the trailing RoPE'd channels in bfloat16.
This mirrors the arithmetic of vLLM's
`csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

FP8_E4M3_MAX = 448.0
QUANT_BLOCK = 64
# The KV-insert kernel floors the block amax at 1e-4, not at the 1e-10 its per-token
# activation quant uses.
AMAX_FLOOR = 1e-4


def _quantize_dequantize_blocks(nope: Tensor) -> Tensor:
    nope_dim = nope.shape[-1]
    num_blocks = (nope_dim + QUANT_BLOCK - 1) // QUANT_BLOCK
    padded_dim = num_blocks * QUANT_BLOCK
    # Zero padding cannot raise a block's amax, so a partial trailing block reduces over
    # exactly its real elements.
    padded = nope if padded_dim == nope_dim else F.pad(nope, (0, padded_dim - nope_dim))

    blocked = padded.unflatten(-1, (num_blocks, QUANT_BLOCK))
    amax = blocked.abs().amax(dim=-1, keepdim=True).clamp_min(AMAX_FLOOR)
    exponent = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    inv_scale = torch.exp2(-exponent)

    scaled = (blocked * inv_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    dequantized = scaled.to(torch.float8_e4m3fn).float() * torch.exp2(exponent)
    return dequantized.flatten(-2)[..., :nope_dim]


class _Fp8RoundTrip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nope: Tensor) -> Tensor:
        # The kernel rounds its fp32 registers to bfloat16 before taking the block amax, so the
        # scale it picks depends on the rounded values.
        return _quantize_dequantize_blocks(nope.float().bfloat16().float()).to(nope.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


def fake_quantize_kv_cache(kv: Tensor, rope_dim: int) -> Tensor:
    """Round-trip the nope channels of `kv` through vLLM's FP8 cache, straight-through backward."""
    nope_dim = kv.shape[-1] - rope_dim
    quantized_nope = _Fp8RoundTrip.apply(kv[..., :nope_dim])
    return torch.cat([quantized_nope, kv[..., nope_dim:]], dim=-1)


__all__ = ["fake_quantize_kv_cache"]
