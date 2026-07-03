from __future__ import annotations

# ruff: noqa: I001 — `prime_rl._compat` must run before `ring_flash_attn` imports below.
import prime_rl._compat  # noqa: F401

from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.nn as nn
from ring_flash_attn import update_ring_flash_attn_params

from prime_rl.utils.sequence import get_cu_seqlens_from_position_ids, get_cu_seqlens_from_seq_lens

CPStyle = Literal["ring", "ulysses"]


def _has_linear_attn_layer(model: nn.Module) -> bool:
    """True if the model contains any non-softmax (linear/SSM) attention layer."""
    inner = getattr(model, "model", model)
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    layers = getattr(inner, "layers", None)
    if layers is None:
        return False
    for layer in layers:
        # Qwen3.5 hybrid DeltaNet
        if getattr(layer, "layer_type", None) == "linear_attention":
            return True
        # NemotronH Mamba
        if hasattr(layer, "mamba"):
            return True
    return False


def assert_cp_style_supports_model(cp_style: CPStyle, model: nn.Module) -> None:
    """Refuse `cp_style='ring'` on models that have linear/SSM attention layers.

    Ring CP is a softmax-attention algorithm (sequence ring all-gather of K/V).
    For non-softmax layers (DeltaNet, Mamba) we'd need a fundamentally different
    CP scheme, which is not implemented. Use `cp_style='ulysses'` for those:
    ulysses' all-to-all is purely on Q/K/V tensors, so the linear/SSM kernel
    runs unchanged on a sequence shard.
    """
    if cp_style == "ring" and _has_linear_attn_layer(model):
        raise ValueError(
            "cp_style='ring' is not supported for models with linear-attention "
            "or Mamba/SSM layers (e.g. Qwen3.5 hybrid, NemotronH). Use "
            "cp_style='ulysses' instead — its all-to-all on Q/K/V works "
            "out-of-the-box with non-softmax kernels."
        )


def setup_model_cp(model: nn.Module, cp_group: dist.ProcessGroup, cp_rank: int, cp_world_size: int) -> None:
    """Hand the CP topology to models whose layers need it (DeltaNet, Mamba).

    Such models expose ``set_context_parallel_attributes`` at the top level and
    own their layer wiring; softmax-only models don't and need nothing.
    """
    if hasattr(model, "set_context_parallel_attributes"):
        model.set_context_parallel_attributes(cp_group, cp_rank, cp_world_size)
        from prime_rl.utils.logger import get_logger

        get_logger().info("Configured model CP attributes")
    elif _has_linear_attn_layer(model):
        raise ValueError(
            "Model has linear-attention/Mamba layers but does not implement "
            "set_context_parallel_attributes; CP would silently misconfigure them"
        )


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
    gathered_t = dist_nn.all_gather(t, group=cp_group)

    return torch.cat(gathered_t, dim=1)


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
    cp_style: CPStyle = "ring",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare the input for context parallelism and set required attention params.

    Both ring and ulysses styles need cu_seqlens computed from the *full*
    (un-sharded) position_ids, then publish them to the patched attention layer:
      - ring: via ring_flash_attn's DATA_PARAMS (with local_k_slice).
      - ulysses: via ULYSSES_PARAMS (just the full cu_seqlens / max_seqlen).

    Returns the sequence-sharded input_ids and position_ids — the rest of the
    model still runs sequence-sharded; only attention sees the full sequence.
    """
    setup_cp_attention_params(position_ids, cp_group=cp_group, cp_style=cp_style)

    input_ids = shard_for_cp(input_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
    position_ids = shard_position_ids_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
    return input_ids, position_ids


def setup_cp_attention_params(
    position_ids: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_style: CPStyle = "ring",
    seq_lens: torch.Tensor | None = None,
) -> None:
    if position_ids.ndim == 3:
        total_tokens = position_ids.shape[2]
        if seq_lens is None:
            cu_seqlens = torch.tensor([0, total_tokens], dtype=torch.int32, device=position_ids.device)
            max_seqlen = total_tokens
        else:
            cu_seqlens, max_seqlen = get_cu_seqlens_from_seq_lens(
                seq_lens.to(device=position_ids.device),
                total_tokens=total_tokens,
            )
    else:
        cu_seqlens, max_seqlen = get_cu_seqlens_from_position_ids(position_ids)

    if cp_style == "ring":
        update_ring_flash_attn_params(cu_seqlens, cp_group)
    elif cp_style == "ulysses":
        # Delayed import: ulysses_attn lives under trainer.models, which imports
        # back into prime_rl.utils — top-level import would deadlock at startup.
        from prime_rl.trainer.models.layers.ulysses_attn import update_ulysses_params

        update_ulysses_params(cu_seqlens, max_seqlen)
    else:
        raise ValueError(f"Unknown cp_style: {cp_style}")
