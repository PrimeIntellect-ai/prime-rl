from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn

from prime_rl.trainer.distributed.collectives import all_gather
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.configs.trainer import ModelConfig
    from prime_rl.trainer.parallel_dims import ParallelDims


@dataclass(frozen=True)
class CPContext:
    """Context-parallel topology, shared by every module that consumes it."""

    cp_group: dist.ProcessGroup | None = None
    cp_rank: int = 0
    cp_world_size: int = 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp_world_size > 1


def setup_context_parallel(model: nn.Module, config: ModelConfig, parallel_dims: ParallelDims) -> None:
    cp_group = parallel_dims.world_mesh["cp"].get_group()
    cp_rank = parallel_dims.world_mesh["cp"].get_local_rank()

    # Delayed import: ulysses_attn lives under trainer.models, which imports back into
    # prime_rl.utils — a top-level import would deadlock at startup.
    from prime_rl.trainer.models.layers.ulysses_attn import substitute_ulysses_attn

    substitute_ulysses_attn(cp_group, attn_impl=config.attn)

    cp_context = CPContext(cp_group, cp_rank, parallel_dims.cp)
    for module in model.modules():
        if not hasattr(module, "cp_context"):
            continue
        if not isinstance(module.cp_context, CPContext):
            raise TypeError(
                f"{type(module).__name__}.cp_context is {type(module.cp_context).__name__}, not a CPContext; "
                "context-parallel setup claims that attribute name, so rename it"
            )
        module.cp_context = cp_context

    get_logger().info(f"Configured context parallelism (cp={parallel_dims.cp})")


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
    *,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard input_ids and position_ids along the sequence."""
    if int(seq_lens.sum()) != input_ids.shape[-1]:
        raise ValueError(f"seq_lens sum must equal sequence length: {seq_lens.tolist()} vs {input_ids.shape[-1]}")
    input_ids = shard_for_cp(input_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
    position_ids = shard_position_ids_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
    return input_ids, position_ids
