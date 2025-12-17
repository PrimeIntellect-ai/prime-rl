import torch
import torch.distributed as dist
from ring_flash_attn import update_ring_flash_attn_params


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
        *extra_tensors: Extra tensors to prepare for context parallelism.
    Returns:
        The prepared input for context parallelism.
    """
    input_ids = shard_for_cp(input_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)

    cu_seqlens = _get_cu_seqlens_for_cp(position_ids)
    update_ring_flash_attn_params(cu_seqlens, cp_group)
    position_ids = shard_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)
    return (
        input_ids,
        position_ids,
    )
