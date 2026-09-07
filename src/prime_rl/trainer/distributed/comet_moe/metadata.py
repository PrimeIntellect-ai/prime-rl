from dataclasses import dataclass

import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.token_dispatcher import _local_reorder


@dataclass
class TileList:
    peer_rank: torch.Tensor
    local_row_start: torch.Tensor
    peer_row_start: torch.Tensor
    valid_rows: torch.Tensor
    flag_index: torch.Tensor
    own_tile_ordinal: torch.Tensor


@dataclass
class CometMoESchedule:
    routed_input: torch.Tensor
    routed_scores: torch.Tensor
    token_indices_experts_sorted: torch.Tensor

    dispatch_tiles: TileList
    combine_tiles: TileList

    token_row_map: torch.Tensor

    recv_capacity: int
    recv_tile_to_local_expert: torch.Tensor
    recv_tile_valid: torch.Tensor
    recv_expert_row_offsets: torch.Tensor

    combine_capacity: int
    all_counts: torch.Tensor


def compute_shared_layout(
    all_counts: torch.Tensor,
    *,
    num_local_experts: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Everything derivable from the shared count matrix alone, with no further communication:
    per-(dest, local_expert, source) segment offsets, and per-dest totals.

    Since `all_counts` is identical on every rank, every rank computing this gets the identical
    answer for *any* rank's schedule, not just its own -- which is what lets each rank build its
    own dispatch tiles from `all_counts[my_rank]` and its own receive/combine tiles by replaying
    every other rank's dispatch schedule (see `build_schedule`), with zero extra communication.
    """
    ep_size = all_counts.shape[0]
    counts_sde = all_counts.view(ep_size, ep_size, num_local_experts)
    padded_sde = ((counts_sde + block_m - 1) // block_m) * block_m
    padded_d_e_s = padded_sde.permute(1, 2, 0).contiguous()
    D, E, S = padded_d_e_s.shape
    flat_per_d = padded_d_e_s.view(D, E * S)
    seg_offsets_per_d = (torch.cumsum(flat_per_d, dim=1) - flat_per_d).view(D, E, S)
    expert_total_padded = padded_d_e_s.sum(dim=2)
    dest_capacity = flat_per_d.sum(dim=1)
    return seg_offsets_per_d, expert_total_padded, dest_capacity


def _tiles_for_source(
    counts_row: torch.Tensor,
    seg_offsets_per_d: torch.Tensor,
    *,
    source_rank: int,
    num_local_experts: int,
    block_m: int,
    device: torch.device,
    max_tiles: int,
) -> TileList:
    """This source rank's dispatch tile list: where each of its own routed rows must go.

    Fixed-size (`max_tiles`), entirely GPU-resident -- no `.item()`/`.any()` host sync. Tail
    entries beyond this source's actual tile count are sentinel-padded (`peer_rank = -1`,
    `valid_rows = 0`); if the real count exceeds `max_tiles`, the excess is silently dropped
    (the caller is responsible for sizing `max_tiles` generously enough, exactly like this
    package's symmetric-memory buffer capacities already are -- see `build_schedule`).
    """
    num_experts = counts_row.numel()
    tiles_per_expert = (counts_row + block_m - 1) // block_m
    cum_tiles = torch.cumsum(tiles_per_expert, dim=0)
    total_tiles = cum_tiles[-1:]

    ordinal = torch.arange(max_tiles, device=device)

    tile_expert = torch.searchsorted(cum_tiles, ordinal, right=True).clamp_(max=num_experts - 1)
    group_flat_start = cum_tiles[tile_expert] - tiles_per_expert[tile_expert]
    tile_idx_within_expert = ordinal - group_flat_start

    valid = ordinal < total_tiles

    expert_starts = torch.cumsum(counts_row, dim=0) - counts_row

    dest_rank = (tile_expert // num_local_experts).to(torch.int64)
    local_expert = (tile_expert % num_local_experts).to(torch.int64)

    local_row_start = expert_starts[tile_expert] + tile_idx_within_expert * block_m
    rows_left = counts_row[tile_expert] - tile_idx_within_expert * block_m
    valid_rows = torch.clamp(rows_left, min=0, max=block_m)
    valid_rows = torch.where(valid, valid_rows, torch.zeros_like(valid_rows))

    dest_offset = seg_offsets_per_d[dest_rank, local_expert, source_rank]
    peer_row_start = dest_offset + tile_idx_within_expert * block_m
    flag_index = peer_row_start // block_m

    peer_rank = torch.where(valid, dest_rank, torch.full_like(dest_rank, -1))

    return TileList(
        peer_rank=peer_rank.to(torch.int32),
        local_row_start=local_row_start.to(torch.int32),
        peer_row_start=peer_row_start.to(torch.int32),
        valid_rows=valid_rows.to(torch.int32),
        flag_index=flag_index.to(torch.int32),
        own_tile_ordinal=ordinal.to(torch.int32),
    )


def _tiles_for_all_sources(
    all_counts: torch.Tensor,
    seg_offsets_per_d: torch.Tensor,
    *,
    num_local_experts: int,
    block_m: int,
    device: torch.device,
    max_tiles: int,
) -> TileList:
    ep_size, num_experts = all_counts.shape
    tiles_per_expert = (all_counts + block_m - 1) // block_m
    cum_tiles = torch.cumsum(tiles_per_expert, dim=1)
    total_tiles = cum_tiles[:, -1:]

    ordinal = torch.arange(max_tiles, device=device).repeat(ep_size, 1)

    tile_expert = torch.searchsorted(cum_tiles, ordinal, right=True).clamp_(max=num_experts - 1)
    group_flat_start = torch.gather(cum_tiles, 1, tile_expert) - torch.gather(tiles_per_expert, 1, tile_expert)
    tile_idx_within_expert = ordinal - group_flat_start

    valid = ordinal < total_tiles

    expert_starts = torch.cumsum(all_counts, dim=1) - all_counts

    dest_rank = (tile_expert // num_local_experts).to(torch.int64)
    local_expert = (tile_expert % num_local_experts).to(torch.int64)

    local_row_start = torch.gather(expert_starts, 1, tile_expert) + tile_idx_within_expert * block_m
    rows_left = torch.gather(all_counts, 1, tile_expert) - tile_idx_within_expert * block_m
    valid_rows = torch.clamp(rows_left, min=0, max=block_m)
    valid_rows = torch.where(valid, valid_rows, torch.zeros_like(valid_rows))

    source_ids = torch.arange(ep_size, device=device).unsqueeze(1).expand(ep_size, max_tiles)
    dest_offset = seg_offsets_per_d[dest_rank, local_expert, source_ids]
    peer_row_start = dest_offset + tile_idx_within_expert * block_m
    flag_index = peer_row_start // block_m

    peer_rank = torch.where(valid, dest_rank, torch.full_like(dest_rank, -1))

    return TileList(
        peer_rank=peer_rank.to(torch.int32),
        local_row_start=local_row_start.to(torch.int32),
        peer_row_start=peer_row_start.to(torch.int32),
        valid_rows=valid_rows.to(torch.int32),
        flag_index=flag_index.to(torch.int32),
        own_tile_ordinal=ordinal.to(torch.int32),
    )


def _gather_routing(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    routed_input, token_indices, routed_scores, num_tokens_per_expert_local = _local_reorder(
        x,
        top_scores,
        selected_experts_indices,
        num_experts=num_experts,
        top_k=top_k,
    )
    ep_size = group.size()
    all_counts = torch.empty(ep_size, num_experts, dtype=num_tokens_per_expert_local.dtype, device=x.device)
    dist.all_gather_into_tensor(all_counts, num_tokens_per_expert_local, group=group)
    return routed_input, token_indices, routed_scores, all_counts


def _compute_schedule_from_counts(
    token_indices: torch.Tensor,
    all_counts: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    ep_size: int,
    my_rank: int,
    num_local_experts: int,
    block_m: int,
    max_recv_tiles: int,
    max_dispatch_tiles: int,
    n_local_tokens: int,
) -> tuple[torch.Tensor, ...]:
    device = all_counts.device

    token_row_map = torch.argsort(token_indices, stable=True).to(torch.int32).view(n_local_tokens, top_k)

    seg_offsets_per_d, expert_total_padded, dest_capacity = compute_shared_layout(
        all_counts,
        num_local_experts=num_local_experts,
        block_m=block_m,
    )

    all_source_tiles = _tiles_for_all_sources(
        all_counts,
        seg_offsets_per_d,
        num_local_experts=num_local_experts,
        block_m=block_m,
        device=device,
        max_tiles=max_dispatch_tiles,
    )
    dispatch_peer_rank = all_source_tiles.peer_rank[my_rank]
    dispatch_local_row_start = all_source_tiles.local_row_start[my_rank]
    dispatch_peer_row_start = all_source_tiles.peer_row_start[my_rank]
    dispatch_valid_rows = all_source_tiles.valid_rows[my_rank]
    dispatch_flag_index = all_source_tiles.flag_index[my_rank]
    dispatch_own_tile_ordinal = all_source_tiles.own_tile_ordinal[my_rank]

    mine = all_source_tiles.peer_rank == my_rank
    source_ids = (
        torch.arange(ep_size, device=device, dtype=torch.int32).unsqueeze(1).expand(ep_size, max_dispatch_tiles)
    )
    neg_one = torch.full_like(all_source_tiles.peer_rank, -1)
    combine_peer_rank = torch.where(mine, source_ids, neg_one).reshape(-1)
    combine_local_row_start = all_source_tiles.peer_row_start.reshape(-1)
    combine_peer_row_start = (all_source_tiles.own_tile_ordinal * block_m).reshape(-1)
    combine_valid_rows = torch.where(
        mine, all_source_tiles.valid_rows, torch.zeros_like(all_source_tiles.valid_rows)
    ).reshape(-1)
    combine_flag_index = combine_peer_row_start // block_m
    combine_own_tile_ordinal = combine_local_row_start // block_m

    expert_total_padded_mine = expert_total_padded[my_rank]
    recv_expert_row_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(expert_total_padded_mine, dim=0).to(torch.int32),
        ]
    )
    tiles_per_local_expert = (expert_total_padded_mine // block_m).to(torch.int64)
    cum_tiles_local = torch.cumsum(tiles_per_local_expert, dim=0)

    recv_ordinal = torch.arange(max_recv_tiles, device=device)
    recv_tile_to_local_expert = (
        torch.searchsorted(cum_tiles_local, recv_ordinal, right=True).clamp_(max=num_local_experts - 1).to(torch.int32)
    )
    recv_tile_valid = (recv_ordinal * block_m < dest_capacity[my_rank]).to(torch.int32)

    return (
        dispatch_peer_rank,
        dispatch_local_row_start,
        dispatch_peer_row_start,
        dispatch_valid_rows,
        dispatch_flag_index,
        dispatch_own_tile_ordinal,
        combine_peer_rank,
        combine_local_row_start,
        combine_peer_row_start,
        combine_valid_rows,
        combine_flag_index,
        combine_own_tile_ordinal,
        token_row_map,
        recv_tile_to_local_expert,
        recv_tile_valid,
        recv_expert_row_offsets,
    )


_compiled_compute_schedule_from_counts = torch.compile(_compute_schedule_from_counts, mode="reduce-overhead")


def build_schedule(
    x: torch.Tensor,
    top_scores: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    group: dist.ProcessGroup,
    block_m: int,
    max_recv_tiles: int,
) -> CometMoESchedule:
    ep_size = group.size()
    my_rank = dist.get_rank(group)
    num_local_experts = num_experts // ep_size

    routed_input, token_indices, routed_scores, all_counts = _gather_routing(
        x,
        top_scores,
        selected_experts_indices,
        num_experts=num_experts,
        top_k=top_k,
        group=group,
    )
    n_local_tokens = x.shape[0]

    n_local_routed = routed_input.shape[0]
    max_dispatch_tiles = (n_local_routed + block_m - 1) // block_m + num_experts

    (
        dispatch_peer_rank,
        dispatch_local_row_start,
        dispatch_peer_row_start,
        dispatch_valid_rows,
        dispatch_flag_index,
        dispatch_own_tile_ordinal,
        combine_peer_rank,
        combine_local_row_start,
        combine_peer_row_start,
        combine_valid_rows,
        combine_flag_index,
        combine_own_tile_ordinal,
        token_row_map,
        recv_tile_to_local_expert,
        recv_tile_valid,
        recv_expert_row_offsets,
    ) = _compiled_compute_schedule_from_counts(
        token_indices,
        all_counts,
        num_experts=num_experts,
        top_k=top_k,
        ep_size=ep_size,
        my_rank=my_rank,
        num_local_experts=num_local_experts,
        block_m=block_m,
        max_recv_tiles=max_recv_tiles,
        max_dispatch_tiles=max_dispatch_tiles,
        n_local_tokens=n_local_tokens,
    )

    dispatch_tiles = TileList(
        peer_rank=dispatch_peer_rank,
        local_row_start=dispatch_local_row_start,
        peer_row_start=dispatch_peer_row_start,
        valid_rows=dispatch_valid_rows,
        flag_index=dispatch_flag_index,
        own_tile_ordinal=dispatch_own_tile_ordinal,
    )
    combine_tiles = TileList(
        peer_rank=combine_peer_rank,
        local_row_start=combine_local_row_start,
        peer_row_start=combine_peer_row_start,
        valid_rows=combine_valid_rows,
        flag_index=combine_flag_index,
        own_tile_ordinal=combine_own_tile_ordinal,
    )

    return CometMoESchedule(
        routed_input=routed_input,
        routed_scores=routed_scores,
        token_indices_experts_sorted=token_indices,
        dispatch_tiles=dispatch_tiles,
        combine_tiles=combine_tiles,
        token_row_map=token_row_map,
        recv_capacity=max_recv_tiles * block_m,
        recv_tile_to_local_expert=recv_tile_to_local_expert,
        recv_tile_valid=recv_tile_valid,
        recv_expert_row_offsets=recv_expert_row_offsets,
        combine_capacity=max_dispatch_tiles * block_m,
        all_counts=all_counts,
    )


def padded_expert_offsets(recv_expert_row_offsets: torch.Tensor, recv_capacity: int) -> torch.Tensor:
    offs = recv_expert_row_offsets[1:].clone()
    offs[-1] = recv_capacity
    return offs


def compute_backward_aux(
    dispatch_tiles: TileList,
    *,
    n_local_routed: int,
    block_m: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(block_m, device=device)
    tile_rows = dispatch_tiles.local_row_start.unsqueeze(1).long() + rows.unsqueeze(0)
    combine_rows = dispatch_tiles.own_tile_ordinal.unsqueeze(1).long() * block_m + rows.unsqueeze(0)
    valid_mask = (rows.unsqueeze(0) < dispatch_tiles.valid_rows.unsqueeze(1)) & (
        dispatch_tiles.peer_rank.unsqueeze(1) >= 0
    )

    dispatch_row_to_combine_pos = torch.zeros(n_local_routed, dtype=torch.int64, device=device)
    dispatch_row_to_combine_pos[tile_rows[valid_mask]] = combine_rows[valid_mask]

    combine_tile_valid = (dispatch_tiles.peer_rank >= 0).to(torch.int32)
    return dispatch_row_to_combine_pos, combine_tile_valid
