from typing import Callable

import torch
import torch.distributed as dist
from ring_flash_attn import update_ring_flash_attn_params


def maybe_shard_for_cp(t: torch.Tensor, cp_rank: int, cp_world_size: int) -> torch.Tensor:
    if cp_world_size == 1:
        return t

    assert t.shape[0] == 1, "For CP, args must have batch dimension of 1"

    chunked_t = torch.chunk(t, cp_world_size, dim=1)

    return chunked_t[cp_rank]


def check_first_and_last_sequence_split_across_cp(position_ids: torch.Tensor, cp_rank: int, cp_world_size: int) -> tuple[bool, bool]:
    if cp_world_size == 1 or cp_rank == cp_world_size - 1:
        return False, False

    seq_len = position_ids.shape[1]
    chunked = torch.chunk(position_ids, cp_world_size, dim=1)

    current_chunk_end = sum(c.shape[1] for c in chunked[: cp_rank + 1])
    next_chunk_start = current_chunk_end

    last_sequence_split_across_cp = False

    if next_chunk_start < seq_len:
        next_rank_first_pos_id = position_ids[0, next_chunk_start].item()
        last_sequence_split_across_cp = next_rank_first_pos_id != 0
    
    first_sequence_split_across_cp = False
    prev_chunk_end = sum(c.shape[1] for c in chunked[: cp_rank])
    if prev_chunk_end > 0:
        prev_rank_last_pos_id = position_ids[0, prev_chunk_end - 1].item()
        first_sequence_split_across_cp = prev_rank_last_pos_id != 0

    return first_sequence_split_across_cp, last_sequence_split_across_cp


def maybe_get_padding_logit_from_prev_cp_rank(
    logits: torch.Tensor, cp_rank: int, cp_world_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor | None:
    if cp_world_size == 1:
        return None

    last_logit = logits[:, -1, :].unsqueeze(1)

    all_rank_last_logits = [torch.zeros(1, 1, logits.shape[2], dtype=logits.dtype, device=logits.device)]

    dist.all_gather(all_rank_last_logits, last_logit, cp_group)

    prev_cp_rank = cp_rank - 1
    if prev_cp_rank >= 0:
        return all_rank_last_logits[prev_cp_rank]
    else:
        return None


def maybe_do_stuff(
    position_ids: torch.Tensor, cp_rank: int, cp_world_size: int, cp_group: dist.ProcessGroup
) -> tuple[bool, bool, torch.Tensor]:
    if cp_world_size == 1:
        return False, False, position_ids

    cu_seqlens = _get_cu_seqlens_for_cp(position_ids)
    update_ring_flash_attn_params(cu_seqlens, cp_group)
    is_sharded_first, is_sharded_last = check_first_and_last_sequence_split_across_cp(position_ids, cp_rank, cp_world_size)
    shard_position_ids = maybe_shard_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)

    return is_sharded_first, is_sharded_last, shard_position_ids


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
