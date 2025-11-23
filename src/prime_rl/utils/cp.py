import torch
import torch.distributed as dist
from ring_flash_attn import update_ring_flash_attn_params


def shard_for_cp(t: torch.Tensor, cp_rank: int, cp_world_size: int) -> torch.Tensor:
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

    all_rank_last_logits = [
        torch.zeros(1, 1, logits.shape[2], dtype=logits.dtype, device=logits.device) for _ in range(cp_world_size)
    ]

    dist.all_gather(all_rank_last_logits, last_logit, group=cp_group)

    prev_cp_rank = cp_rank - 1
    if prev_cp_rank >= 0:
        return all_rank_last_logits[prev_cp_rank]
    else:
        return None


def update_ring_flash_attn_params_and_shard(
    position_ids: torch.Tensor, cp_rank: int, cp_world_size: int, cp_group: dist.ProcessGroup
) -> torch.Tensor:
    if cp_world_size == 1:
        return position_ids

    cu_seqlens = _get_cu_seqlens_for_cp(position_ids)
    update_ring_flash_attn_params(cu_seqlens, cp_group)
    return shard_for_cp(position_ids, cp_rank=cp_rank, cp_world_size=cp_world_size)


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


def sync_boundary_stats(
    first_seq_stats: torch.Tensor,
    last_seq_stats: torch.Tensor,
    starts_with_zero: bool,
    is_single_sequence: bool,
    cp_rank: int,
    cp_world_size: int,
    cp_group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Synchronize statistics for boundary sequences across CP ranks.

    Args:
        first_seq_stats: Stats for the first sequence on this rank [num_stats]
        last_seq_stats: Stats for the last sequence on this rank [num_stats]
        starts_with_zero: Whether the first sequence starts with 0 (new document)
        is_single_sequence: Whether this rank has only one sequence
        cp_rank: Current CP rank
        cp_world_size: CP world size
        cp_group: Process group for CP

    Returns:
        Tuple of (aggregated_first_seq_stats, aggregated_last_seq_stats)
    """
    if cp_world_size == 1:
        return first_seq_stats, last_seq_stats

    device = first_seq_stats.device
    num_stats = first_seq_stats.numel()

    # Prepare data for gathering: [starts_with_zero, is_single_sequence, first_stats..., last_stats...]
    # 1 + 1 + num_stats + num_stats = 2 + 2 * num_stats
    local_data = torch.cat(
        [
            torch.tensor([float(starts_with_zero), float(is_single_sequence)], device=device),
            first_seq_stats,
            last_seq_stats,
        ]
    )

    gathered_data = [torch.zeros_like(local_data) for _ in range(cp_world_size)]
    dist.all_gather(gathered_data, local_data, group=cp_group)

    # Parse gathered data
    all_starts_with_zero = [bool(d[0].item()) for d in gathered_data]
    all_is_single = [bool(d[1].item()) for d in gathered_data]
    all_first_stats = [d[2 : 2 + num_stats] for d in gathered_data]
    all_last_stats = [d[2 + num_stats :] for d in gathered_data]

    # Helper to aggregate stats (sum)
    # Note: This assumes all stats are sums. If we have min, we need to handle it differently.
    # But for now we will assume the caller handles the aggregation logic if it's not just sum.
    # Wait, we have min_ratio. We can't just sum everything.
    # We should let the caller handle aggregation? Or pass aggregation mode?
    # Since we have specific stats (sum_log, sum_mask, min_ratio), we should probably handle them specifically here
    # or make this function generic enough.
    # Let's assume the stats are passed such that we can just sum them? No, min is not sum.
    # Let's just return the gathered data and let the caller process it?
    # No, the logic of walking backwards/forwards is complex.
    # Let's implement the walking logic here, but take a `merge_fn`?
    # Or just hardcode for our use case: stats = [sum_log, sum_mask, min_ratio]
    # merge = (a[0]+b[0], a[1]+b[1], min(a[2], b[2]))

    def merge_stats(s1, s2):
        # s1, s2 are tensors of shape [3]
        # 0: sum, 1: sum, 2: min
        return torch.stack([s1[0] + s2[0], s1[1] + s2[1], torch.min(s1[2], s2[2])])

    # Calculate aggregated first seq stats
    agg_first = all_first_stats[cp_rank].clone()
    if not starts_with_zero:
        # Walk backwards
        for j in range(cp_rank - 1, -1, -1):
            agg_first = merge_stats(agg_first, all_last_stats[j])
            if not all_is_single[j]:
                break
            if all_starts_with_zero[j]:
                break

    # Calculate aggregated last seq stats
    agg_last = all_last_stats[cp_rank].clone()

    # If we are a single sequence, we already aggregated the prefix in agg_first.
    # So agg_last should start with agg_first (which includes prefix + local).
    if is_single_sequence:
        agg_last = agg_first.clone()

    # Walk forward
    if cp_rank < cp_world_size - 1:
        # Check if we continue to next
        if not all_starts_with_zero[cp_rank + 1]:
            for j in range(cp_rank + 1, cp_world_size):
                agg_last = merge_stats(agg_last, all_first_stats[j])
                if not all_is_single[j]:
                    break
                if j + 1 < cp_world_size and all_starts_with_zero[j + 1]:
                    break

    # If is_single_sequence, agg_first needs to be updated with the forward aggregation too
    if is_single_sequence:
        agg_first = agg_last.clone()

    return agg_first, agg_last
