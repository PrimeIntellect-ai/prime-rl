from dataclasses import dataclass

import prime_kernels
import torch


@dataclass
class FlashMoeScratch:
    sorted_token_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    topk_weights: torch.Tensor
    out: torch.Tensor
    weighted_routed_out: torch.Tensor


def init_flash_moe_scratch(
    capacity: int, n_local_routed: int, hidden_dim: int, device: torch.device
) -> FlashMoeScratch:
    return FlashMoeScratch(
        sorted_token_ids=torch.arange(capacity, dtype=torch.int32, device=device),
        num_tokens_post_padded=torch.tensor([capacity], dtype=torch.int32, device=device),
        topk_weights=torch.ones(capacity, 1, dtype=torch.float32, device=device),
        out=torch.empty(capacity, hidden_dim, dtype=torch.bfloat16, device=device),
        weighted_routed_out=torch.empty(n_local_routed, hidden_dim, dtype=torch.bfloat16, device=device),
    )


def run_flash_moe_expert_compute(
    hidden: torch.Tensor,
    tile_to_local_expert: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    scratch: FlashMoeScratch,
    *,
    block_m: int,
    block_n: int = 64,
    warp_n: int = 4,
    stages: int = 2,
) -> torch.Tensor:
    flash_moe = prime_kernels.load("flash_moe")
    capacity, hidden_dim = hidden.shape
    reason = flash_moe.unsupported_shape_reason(hidden_dim, gate_up_weight.shape[1], mxfp8=False, split=True)
    if reason is not None:
        raise ValueError(f"flash_moe cannot run this MoE layer's shape: {reason}")
    flash_moe.fused_moe_bf16(
        hidden,
        gate_up_weight,
        down_weight,
        scratch.sorted_token_ids,
        tile_to_local_expert.to(torch.int32),
        scratch.num_tokens_post_padded,
        scratch.topk_weights,
        scratch.out,
        1,
        block_m,
        block_n,
        warp_n,
        stages,
        1,
        1,
        True,
    )
    return scratch.out
