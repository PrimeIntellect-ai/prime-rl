import torch
import torch.distributed as dist
from ring_flash_attn import update_ring_flash_attn_params


def maybe_shard_for_cp(t: torch.Tensor, cp_rank: int, cp_world_size: int) -> torch.Tensor:
    if cp_world_size == 1:
        return t

    assert t.shape[0] == 1, "For CP, args must have batch dimension of 1"

    chunked_t = torch.chunk(t, cp_world_size, dim=1)

    return chunked_t[cp_rank]


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


def maybe_update_ring_flash_attn_params_and_shard_for_cp(position_ids: torch.Tensor, cp_rank: int, cp_world_size: int, cp_group: dist.ProcessGroup) -> torch.Tensor:
    if cp_world_size == 1:
        return position_ids

    cu_seqlens = _get_cu_seqlens_for_cp(position_ids)
    update_ring_flash_attn_params(cu_seqlens, cp_group)
    return maybe_shard_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)


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
