from __future__ import annotations

# ruff: noqa: I001 — `prime_rl._compat` must run before `ring_flash_attn` imports below.
import prime_rl._compat  # noqa: F401

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from ring_flash_attn import substitute_hf_flash_attn, update_ring_flash_attn_params

from prime_rl.trainer.distributed.collectives import all_gather
from prime_rl.utils.logger import get_logger
from prime_rl.utils.sequence import get_cu_seqlens_from_seq_lens

if TYPE_CHECKING:
    # `prime_rl.trainer.models` imports this module, so importing the model base eagerly would
    # cycle. `from __future__ import annotations` keeps CPStyle out of the runtime path.
    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.trainer.models.base import CPStyle
    from prime_rl.trainer.parallel_dims import ParallelDims


def setup_context_parallel(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims) -> None:
    cp_group = parallel_dims.world_mesh["cp"].get_group()
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank()

    if config.cp_style == "ring":
        # Delayed imports: both modules live under trainer.models, which imports back into
        # prime_rl.utils — a top-level import would deadlock at startup.
        from prime_rl.trainer.models.layers.attn import substitute_ring_attn

        substitute_hf_flash_attn(cp_group, heads_k_stride=1)
        substitute_ring_attn(cp_group, heads_k_stride=1, attn_impl=config.attn)
    elif config.cp_style == "ulysses":
        from prime_rl.trainer.models.layers.ulysses_attn import substitute_hf_ulysses_attn, substitute_ulysses_attn

        substitute_hf_ulysses_attn(cp_group)
        substitute_ulysses_attn(cp_group, attn_impl=config.attn)
    else:
        raise ValueError(f"Unknown cp_style: {config.cp_style}")

    for module in model.modules():
        if hasattr(module, "setup_context_parallel"):
            module.setup_context_parallel(cp_group, cp_rank, parallel_dims.cp, config.cp_style)

    get_logger().info(f"Configured {config.cp_style} context parallelism (cp={parallel_dims.cp})")


def shard_for_cp(t: torch.Tensor, cp_rank: int, cp_world_size: int, seq_dim: int = 1) -> torch.Tensor:
    """
    Shard a tensor for context parallelism.
    Args:
        t: The tensor to shard.
        cp_rank: The rank of the current process.
        cp_world_size: The number of processes in the context parallel group.
    Returns:
        The shard of the tensor for the current rank.
    """

    if seq_dim == 1 and t.shape[0] != 1:
        raise ValueError(f"For CP, tensor must have batch dimension 1, got shape={tuple(t.shape)}")
    if t.shape[seq_dim] % cp_world_size != 0:
        raise ValueError(
            f"CP requires sequence dimension {seq_dim} to be divisible by cp size: "
            f"shape={tuple(t.shape)}, cp_size={cp_world_size}; "
            "uneven shards deadlock CP collectives (e.g. ulysses all-to-all)"
        )

    chunked_t = torch.chunk(t, cp_world_size, dim=seq_dim)

    return chunked_t[cp_rank]


def shard_position_ids_for_cp(position_ids: torch.Tensor, cp_rank: int, cp_world_size: int) -> torch.Tensor:
    if position_ids.ndim == 3:
        return shard_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size, seq_dim=2)
    return shard_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)


def gather_for_cp(t: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    return all_gather(t, 1, cp_group)


def gather_for_cp_wo_grad(t: torch.Tensor, cp_world_size: int, cp_group: dist.ProcessGroup) -> torch.Tensor:
    empty_like_t = [torch.empty_like(t) for _ in range(cp_world_size)]
    dist.all_gather(empty_like_t, t, group=cp_group)
    return torch.cat(empty_like_t, dim=1)


def setup_cp_params(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cp_rank: int,
    cp_world_size: int,
    cp_group: dist.ProcessGroup,
    *,
    seq_lens: torch.Tensor,
    cp_style: CPStyle = "ring",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare the input for context parallelism and set required attention params.

    Both ring and ulysses styles need cu_seqlens computed from the full,
    unsharded sequence lengths, then publish them to the patched attention layer:
      - ring: via ring_flash_attn's DATA_PARAMS (with local_k_slice).
      - ulysses: via ULYSSES_PARAMS (just the full cu_seqlens / max_seqlen).

    Returns the sequence-sharded input_ids and position_ids — the rest of the
    model still runs sequence-sharded; only attention sees the full sequence.
    """
    setup_cp_attention_params(position_ids, cp_group=cp_group, cp_style=cp_style, seq_lens=seq_lens)

    input_ids = shard_for_cp(input_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
    position_ids = shard_position_ids_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
    return input_ids, position_ids


def setup_cp_attention_params(
    position_ids: torch.Tensor,
    cp_group: dist.ProcessGroup,
    *,
    seq_lens: torch.Tensor,
    cp_style: CPStyle = "ring",
) -> None:
    total_tokens = position_ids.shape[-1]
    cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
        seq_lens.to(device=position_ids.device),
        total_tokens=total_tokens,
    )

    if cp_style == "ring":
        update_ring_flash_attn_params(cu_seqlens, cp_group)
    elif cp_style == "ulysses":
        # Delayed import: ulysses_attn lives under trainer.models, which imports
        # back into prime_rl.utils — top-level import would deadlock at startup.
        from prime_rl.trainer.models.layers.ulysses_attn import update_ulysses_params

        update_ulysses_params(cu_seqlens, max_seqlen)
    else:
        raise ValueError(f"Unknown cp_style: {cp_style}")
