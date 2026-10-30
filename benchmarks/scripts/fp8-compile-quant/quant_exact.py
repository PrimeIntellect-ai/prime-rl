"""Torch-op FP8 quantizers that match the Triton kernels bit for bit, via a correctly rounded division op."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from quant_torch import AMAX_FLOOR, BLOCK, FP8_MAX, TMA_ALIGNMENT_FP32

from prime_rl.trainer.models.kernels.fp8_torch_quant import div_rn


def fp8_max_like(t: torch.Tensor) -> torch.Tensor:
    return torch.full((), FP8_MAX, dtype=t.dtype, device=t.device)


def quant_1x128_exact(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`quant_1x128` with correctly rounded scale and element divisions, bitwise equal to `per_token_cast_to_fp8_triton`."""
    tokens, channels = x.shape
    groups = x.float().reshape(tokens, channels // BLOCK, BLOCK)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=AMAX_FLOOR)
    scale = div_rn(amax, fp8_max_like(amax))
    x_fp8 = div_rn(groups, scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8.view(tokens, channels), scale.view(tokens, channels // BLOCK)


def quant_128x1_tp_exact(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`quant_128x1_tp` with correctly rounded scale and element divisions, bitwise equal to `per_token_cast_to_fp8_tp_triton`."""
    tokens, channels = x.shape
    padded_tokens = (tokens + BLOCK - 1) // BLOCK * BLOCK
    token_groups = padded_tokens // BLOCK
    x = F.pad(x, (0, 0, 0, padded_tokens - tokens)).float()
    amax = x.reshape(token_groups, BLOCK, channels).abs().amax(dim=1).clamp(min=AMAX_FLOOR)
    scale = div_rn(amax, fp8_max_like(amax))
    tma_aligned_channels = (channels + TMA_ALIGNMENT_FP32 - 1) // TMA_ALIGNMENT_FP32 * TMA_ALIGNMENT_FP32
    if tma_aligned_channels != channels:
        scale = F.pad(scale, (0, tma_aligned_channels - channels))[:, :channels]
    scale_t = scale.t()
    x_t_groups = x.t().reshape(channels, token_groups, BLOCK)
    x_fp8_t = div_rn(x_t_groups, scale_t.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8_t.contiguous().view(channels, padded_tokens), scale_t
