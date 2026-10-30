"""FP8 activation quantizers written in plain torch ops, so Inductor can fuse them into the producing kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
AMAX_FLOOR = 1e-10
BLOCK = 128
TMA_ALIGNMENT_FP32 = 4


def quant_1x128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token cast in 1x128 groups with `per_token_cast_to_fp8_triton(x, use_ue8m0=False)`'s outputs and layout."""
    tokens, channels = x.shape
    groups = x.float().reshape(tokens, channels // BLOCK, BLOCK)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=AMAX_FLOOR)
    scale = amax / FP8_MAX
    x_fp8 = (groups / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return x_fp8.view(tokens, channels), scale.view(tokens, channels // BLOCK)


def quant_128x1_tp(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast of `x.T` in 128-token groups after zero-padding tokens to a multiple of 128, in `per_token_cast_to_fp8_tp_triton`'s layout."""
    tokens, channels = x.shape
    padded_tokens = (tokens + BLOCK - 1) // BLOCK * BLOCK
    token_groups = padded_tokens // BLOCK
    x = F.pad(x, (0, 0, 0, padded_tokens - tokens)).float()
    amax = x.reshape(token_groups, BLOCK, channels).abs().amax(dim=1).clamp(min=AMAX_FLOOR)
    scale = amax / FP8_MAX
    tma_aligned_channels = (channels + TMA_ALIGNMENT_FP32 - 1) // TMA_ALIGNMENT_FP32 * TMA_ALIGNMENT_FP32
    if tma_aligned_channels != channels:
        scale = F.pad(scale, (0, tma_aligned_channels - channels))[:, :channels]
    scale_t = scale.t()
    x_t_groups = x.t().reshape(channels, token_groups, BLOCK)
    x_fp8_t = (x_t_groups / scale_t.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # Materializing the 3D [channels, token_groups, BLOCK] view, not the 2D one, lets Inductor fuse the cast into the amax.
    return x_fp8_t.contiguous().view(channels, padded_tokens), scale_t
