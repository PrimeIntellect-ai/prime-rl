from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn as nn
from ring_flash_attn import update_ring_flash_attn_params


def setup_hybrid_cp(model: nn.Module, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
    """Configure DeltaNet modules in Qwen3.5 hybrid models for native fla CP."""
    layers = None
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "language_model"):
            inner = inner.language_model
        if hasattr(inner, "layers"):
            layers = inner.layers

    if layers is None:
        return

    count = 0
    for layer in layers:
        if getattr(layer, "layer_type", None) == "linear_attention":
            attn = getattr(layer, "linear_attn", None)
            if attn is not None:
                attn.cp_group = cp_group
                attn.cp_rank = cp_rank
                attn.cp_world_size = cp_world_size
                count += 1

    if count > 0:
        from prime_rl.utils.logger import get_logger

        get_logger().info(f"Configured hybrid CP on {count} DeltaNet modules (fla native state passing)")


def setup_nemotron_h_cp(model: nn.Module, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
    """Configure NemotronH Mamba layers for context-parallel gather/scatter."""
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers

    if layers is None:
        return

    count = 0
    for layer in layers:
        if hasattr(layer, "mamba") and hasattr(layer, "set_context_parallel_attributes"):
            layer.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)
            count += 1

    if count > 0:
        from prime_rl.utils.logger import get_logger

        get_logger().info(f"Configured NemotronH CP on {count} Mamba layers (all-to-all head partitioning)")


def setup_sparse_mla_cp(model: nn.Module, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
    """Configure GLM-5 sparse MLA modules for context-parallel gather/scatter."""

    count = 0
    if not hasattr(model, "model"):
        return

    if not hasattr(model.model, "layers"):
        return

    for layer in model.model.layers:
        if not hasattr(layer, "set_context_parallel_attributes"):
            continue

        layer.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)
        count += 1

    if count > 0:
        from prime_rl.utils.logger import get_logger

        get_logger().info(f"Configured sparse MLA CP on {count} DSA layers")


def shard_for_cp(t: torch.Tensor, cp_rank: int, cp_world_size: int) -> torch.Tensor:
    """
    Shard a tensor for context parallelism.
    Args:
        t: The tensor to shard.
        cp_rank: The rank of the current process.
        cp_world_size: The number of processes in the context parallel group.
    Returns:
        The shard of the tensor for the current rank.
    """

    assert t.shape[0] == 1, "For CP, tensor must have batch dimension of 1"

    chunked_t = torch.chunk(t, cp_world_size, dim=1)

    return chunked_t[cp_rank]


def gather_for_cp(t: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    gathered_t = dist_nn.all_gather(t, group=cp_group)

    return torch.cat(gathered_t, dim=1)


def gather_for_cp_wo_grad(t: torch.Tensor, cp_world_size: int, cp_group: dist.ProcessGroup) -> torch.Tensor:
    empty_like_t = [torch.empty_like(t) for _ in range(cp_world_size)]
    dist.all_gather(empty_like_t, t, group=cp_group)
    return torch.cat(empty_like_t, dim=1)


def get_padding_logit_from_prev_cp_rank(
    logits: torch.Tensor, cp_rank: int, cp_world_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor | None:
    """
    Get the padding logit from the previous context parallel rank.
    Args:
        logits: The logits tensor.
        cp_rank: The rank of the current process.
        cp_world_size: The number of processes in the context parallel group.
        cp_group: The context parallel group.
    Returns:
        The padding logit from the previous context parallel rank.
    """
    last_logit = logits[:, -1, :].unsqueeze(1)

    all_rank_last_logits = [
        torch.zeros(1, 1, logits.shape[2], dtype=logits.dtype, device=logits.device) for _ in range(cp_world_size)
    ]

    dist.all_gather(all_rank_last_logits, last_logit, group=cp_group)

    prev_cp_rank = cp_rank - 1
    if prev_cp_rank >= 0:
        return all_rank_last_logits[prev_cp_rank]
    else:
        return None


def _get_cu_seqlens_for_cp(position_ids: torch.Tensor) -> torch.Tensor:
    flat_position_ids = position_ids.view(-1)
    seqlens = torch.cat(
        [
            flat_position_ids[0:1],
            flat_position_ids[:-1][(flat_position_ids == 0)[1:]] + 1,
            flat_position_ids[-1:] + 1,
        ]
    )
    cu_seqlens = seqlens.cumsum(dim=0, dtype=torch.int32)
    return cu_seqlens


def setup_cp_params(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cp_rank: int,
    cp_world_size: int,
    cp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare the input for context parallelism and sets required parameters for ring flash attention.
    Args:
        input_ids: The input ids tensor.
        position_ids: The position ids tensor.
        cp_rank: The rank of the current process.
        cp_world_size: The number of processes in the context parallel group.
        cp_group: The context parallel group.
    Returns:
        The sharded input_ids and position_ids for context parallelism.
    """
    input_ids = shard_for_cp(input_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)

    cu_seqlens = _get_cu_seqlens_for_cp(position_ids)
    update_ring_flash_attn_params(cu_seqlens, cp_group)
    position_ids = shard_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
    return (
        input_ids,
        position_ids,
    )
