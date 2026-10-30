"""FP8 activation quantizers written in plain torch ops, so Inductor can fuse them into the producing kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
AMAX_FLOOR = 1e-10
BLOCK = 128
TMA_ALIGNMENT_FP32 = 4
WORD_BYTES = 8


def quant_1x128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token cast in 1x128 groups, as `per_token_cast_to_fp8_triton(x, use_ue8m0=False)` with FP8 values shaped `[tokens, channels / 128, 128]`."""
    tokens, channels = x.shape
    groups = x.float().reshape(tokens, channels // BLOCK, BLOCK)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp(min=AMAX_FLOOR)
    scale = amax / FP8_MAX
    x_fp8 = (groups / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # An opaque op that takes a 2D view makes Inductor store the view in its own kernel, so the 3D groups go to the op.
    return x_fp8, scale.view(tokens, channels // BLOCK)


def quant_128x1_tp(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast of `x.T` in 128-token groups after zero-padding tokens to a multiple of 128, as `per_token_cast_to_fp8_tp_triton` with FP8 values shaped `[channels, padded_tokens / 128, 128]`."""
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
    return x_fp8_t.contiguous(), scale_t


def _stored(x: torch.Tensor) -> torch.Tensor:
    """Return `x` through a pair of no-op dtype views, which make Inductor store it to memory instead of recomputing it in each reader."""
    return x.contiguous().view(torch.int32).view(x.dtype)


def quant_grouped_128x1_kmajor(x: torch.Tensor, offs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel cast in 128-row blocks in `grouped_per_channel_cast_to_fp8_sm90_kmajor_triton`'s layout, with rows past `offs[-1]` as one extra trailing group."""
    rows, channels = x.shape
    row_blocks = rows // BLOCK
    x_float = _stored(x).float()
    amax = x_float.reshape(row_blocks, BLOCK, channels).abs().amax(dim=1).clamp(min=AMAX_FLOOR)
    scale_t = (amax / FP8_MAX).t()
    x_t_blocks = x_float.t().reshape(channels, row_blocks, BLOCK)
    x_fp8_t = (x_t_blocks / scale_t.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    # The int64 view realizes x_fp8_t in the amax kernel, so the group scatter below moves coalesced 8-byte words.
    x_fp8_t_words = x_fp8_t.contiguous().view(torch.int64)

    group_ends = torch.cat((offs, offs.new_full((1,), rows))).long()
    group_starts = torch.cat((offs.new_zeros(1), offs)).long()
    word = torch.arange(rows // WORD_BYTES, device=x.device)
    word_group = torch.bucketize(word * WORD_BYTES, group_ends, right=True)
    group_start = group_starts[word_group] // WORD_BYTES
    group_size = group_ends[word_group] // WORD_BYTES - group_start
    channel = torch.arange(channels, device=x.device).unsqueeze(-1)
    destination = group_start * channels + channel * group_size + (word - group_start)
    destination = destination.view(channels, row_blocks, BLOCK // WORD_BYTES)
    out = torch.index_put(x_fp8_t_words.new_empty(rows * channels // WORD_BYTES), (destination,), x_fp8_t_words)
    return out.view(torch.float8_e4m3fn), scale_t
