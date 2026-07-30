from __future__ import annotations

import torch

from . import _C
from .mxfp8 import pack_scales_blocked, quantize_activation_mxfp8, quantize_weight_mxfp8

__all__ = [
    "fused_moe_bf16",
    "fused_moe_mxfp8",
    "moe_align",
    "pack_scales_blocked",
    "quantize_activation_mxfp8",
    "quantize_weight_mxfp8",
]


@torch.library.register_fake("prime_moe::moe_align")
def _moe_align_fake(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int,
    bpc: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ()


@torch.library.register_fake("prime_moe::fused_moe_bf16")
def _fused_moe_bf16_fake(
    x: torch.Tensor,
    w: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int,
    block_n: int,
    warp_n: int,
    stages: int,
    bpc: int,
) -> None:
    return


@torch.library.register_fake("prime_moe::fused_moe_mxfp8")
def _fused_moe_mxfp8_fake(
    x: torch.Tensor,
    x_scales: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int,
    block_n: int,
    warp_n: int,
    stages: int,
    bpc: int,
) -> None:
    return


def moe_align(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int = 128,
    bpc: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.prime_moe.moe_align(topk_ids, num_experts, block_m, bpc)


def fused_moe_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int = 128,
    block_n: int = 64,
    warp_n: int = 4,
    stages: int = 4,
    bpc: int = 1,
) -> torch.Tensor:
    torch.ops.prime_moe.fused_moe_bf16(
        x,
        w,
        w2,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        block_n,
        warp_n,
        stages,
        bpc,
    )
    return out


def fused_moe_mxfp8(
    x: torch.Tensor,
    x_scales: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    w2: torch.Tensor,
    w2_scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    top_k: int,
    block_m: int = 128,
    block_n: int = 64,
    warp_n: int = 4,
    stages: int = 4,
    bpc: int = 1,
) -> torch.Tensor:
    torch.ops.prime_moe.fused_moe_mxfp8(
        x,
        x_scales,
        w,
        w_scales,
        w2,
        w2_scales,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        top_k,
        block_m,
        block_n,
        warp_n,
        stages,
        bpc,
    )
    return out
