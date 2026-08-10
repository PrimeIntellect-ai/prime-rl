from __future__ import annotations

import torch

from . import _C  # noqa: F401  # imported for its side effect: registers the prime_moe ops
from .mxfp8 import pack_scales_blocked

__all__ = [
    "fused_moe_bf16",
    "fused_moe_mxfp8",
    "moe_align",
    "pack_scales_blocked",
]


@torch.library.register_fake("prime_moe::moe_align")
def _moe_align_fake(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int,
    bpc: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    pad_to = block_m * bpc
    max_padded = (topk_ids.numel() + num_experts * (pad_to - 1) + pad_to - 1) // pad_to * pad_to
    options = {"dtype": torch.int32, "device": topk_ids.device}
    return (
        torch.empty(max_padded, **options),
        torch.empty(max_padded // block_m, **options),
        torch.empty(1, **options),
    )


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
    cpc: int,
    split: bool,
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
    split: bool,
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
    stages: int = 2,
    bpc: int = 1,
    cpc: int = 1,
    split: bool = True,
) -> torch.Tensor:
    """Fused bf16 MoE. Writes into `out` in place and returns it.

    `w` is the gate/up projection as plain contiguous `(E, N, K)`, gate in the first half of
    N and up in the second. The kernel interleaves the two halves through its tensor map, so
    do not pre-interleave them on the host.

    `split` materializes the intermediate activation in a scratch buffer and runs the down
    projection as a second kernel, which removes the split-K reduction over `out` — it zeroes
    `out` itself and requires `stages <= 4`. With `split=False` the whole thing stays on chip
    and accumulates into `out` through global reductions, so `out` must be zeroed by the
    caller.
    """
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
        cpc,
        split,
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
    split: bool = True,
) -> torch.Tensor:
    """Fused mxfp8 MoE. Writes into `out` in place and returns it.

    `w` follows the same plain `(E, N, K)` layout as `fused_moe_bf16`. `w_scales` and
    `w2_scales` must be in the blocked layout `pack_scales_blocked` produces; `x_scales`
    stays row major `(M, K // 32)`.

    `split` means the same as in `fused_moe_bf16`: the split pipeline zeroes `out` itself,
    the single fused kernel needs `out` zeroed by the caller.
    """
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
        split,
    )
    return out
