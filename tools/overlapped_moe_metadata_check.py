import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.overlapped_moe.metadata import (
    _tiles_for_source,
    build_schedule,
    compute_shared_layout,
)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    group = dist.group.WORLD
    num_local_tokens = 200
    num_experts = 4 * world_size
    num_local_experts = num_experts // world_size
    top_k = 2
    hidden_dim = 16
    block_m = 16
    max_dispatch_tiles = (num_local_tokens * top_k + block_m - 1) // block_m + num_experts
    max_recv_tiles = max_dispatch_tiles * world_size  # generous: every source could land on me
    torch.manual_seed(10 + rank)
    x = torch.randn(num_local_tokens, hidden_dim, device=device)
    logits = torch.randn(num_local_tokens, num_experts, device=device)
    top_scores, selected = torch.topk(logits, k=top_k, dim=1)

    schedule = build_schedule(
        x,
        top_scores,
        selected,
        num_experts=num_experts,
        top_k=top_k,
        group=group,
        block_m=block_m,
        max_recv_tiles=max_recv_tiles,
    )
    n_real_dispatch_tiles = int((schedule.dispatch_tiles.peer_rank >= 0).sum().item())
    print(
        f"[rank {rank}] n_routed={schedule.routed_input.shape[0]} "
        f"recv_capacity={schedule.recv_capacity} combine_capacity={schedule.combine_capacity} "
        f"n_dispatch_tiles={n_real_dispatch_tiles} (of {schedule.dispatch_tiles.peer_rank.numel()} fixed slots) "
        f"n_combine_tiles={int((schedule.combine_tiles.peer_rank >= 0).sum().item())} "
        f"(of {schedule.combine_tiles.peer_rank.numel()} fixed slots)",
        flush=True,
    )
    seg_offsets_per_d, _, _ = compute_shared_layout(
        schedule.all_counts, num_local_experts=num_local_experts, block_m=block_m
    )
    all_dispatch_tiles = [
        _tiles_for_source(
            schedule.all_counts[s],
            seg_offsets_per_d,
            source_rank=s,
            num_local_experts=num_local_experts,
            block_m=block_m,
            device=device,
            max_tiles=max_dispatch_tiles,
        )
        for s in range(world_size)
    ]
    assert all_dispatch_tiles[rank].peer_rank.numel() == schedule.dispatch_tiles.peer_rank.numel()
    assert torch.equal(all_dispatch_tiles[rank].peer_rank, schedule.dispatch_tiles.peer_rank), (
        "rebuilt dispatch schedule (via _tiles_for_source directly) disagrees with build_schedule's own"
    )
    all_routed_inputs = [None] * world_size
    for r in range(world_size):
        n = torch.tensor([schedule.routed_input.shape[0]] if r == rank else [0], device=device, dtype=torch.int64)
        dist.broadcast(n, src=r, group=group)
        buf = schedule.routed_input if r == rank else torch.zeros(int(n.item()), hidden_dim, device=device)
        dist.broadcast(buf, src=r, group=group)
        all_routed_inputs[r] = buf
    recv = torch.zeros(schedule.recv_capacity, hidden_dim, device=device)
    filled = torch.zeros(schedule.recv_capacity, dtype=torch.bool, device=device)
    for s in range(world_size):
        tiles = all_dispatch_tiles[s]
        mine = tiles.peer_rank == rank
        if not bool(mine.any()):
            continue
        local_start = tiles.local_row_start[mine].long()
        peer_start = tiles.peer_row_start[mine].long()
        valid = tiles.valid_rows[mine].long()
        for i in range(local_start.numel()):
            n = int(valid[i].item())
            if n == 0:
                continue
            recv[peer_start[i] : peer_start[i] + n] = all_routed_inputs[s][local_start[i] : local_start[i] + n]
            assert not bool(filled[peer_start[i] : peer_start[i] + n].any()), "dispatch tile overlap!"
            filled[peer_start[i] : peer_start[i] + n] = True

    expected_real_rows = sum(int(t.valid_rows[t.peer_rank == rank].sum().item()) for t in all_dispatch_tiles)
    dispatch_ok = int(filled.sum().item()) == expected_real_rows
    print(
        f"[rank {rank}] dispatch: filled={int(filled.sum().item())} expected={expected_real_rows} ok={dispatch_ok}",
        flush=True,
    )
    expert_out = recv * 2.0 + 1.0

    def combine_tiles_for(owner_rank):
        peer_rank, local_start, peer_start, valid = [], [], [], []
        for s in range(world_size):
            t = all_dispatch_tiles[s]
            mine = t.peer_rank == owner_rank
            if not bool(mine.any()):
                continue
            peer_rank.append(torch.full((int(mine.sum().item()),), s, dtype=torch.int32, device=device))
            local_start.append(t.peer_row_start[mine])
            peer_start.append(t.own_tile_ordinal[mine] * block_m)
            valid.append(t.valid_rows[mine])
        if not peer_rank:
            z = torch.empty(0, dtype=torch.int64, device=device)
            return z, z, z, z
        return (
            torch.cat(peer_rank).long(),
            torch.cat(local_start).long(),
            torch.cat(peer_start).long(),
            torch.cat(valid).long(),
        )

    my_combine_peer, my_combine_local, my_combine_peer_row, my_combine_valid = combine_tiles_for(rank)
    real = schedule.combine_tiles.peer_rank >= 0
    assert torch.equal(my_combine_peer.int(), schedule.combine_tiles.peer_rank[real]), (
        "combine schedule mismatch (peer_rank)"
    )
    assert torch.equal(my_combine_local.int(), schedule.combine_tiles.local_row_start[real]), (
        "combine schedule mismatch (local_row_start)"
    )
    assert torch.equal(my_combine_peer_row.int(), schedule.combine_tiles.peer_row_start[real]), (
        "combine schedule mismatch (peer_row_start)"
    )

    combine_recv = torch.zeros(schedule.combine_capacity, hidden_dim, device=device)
    combine_filled = torch.zeros(schedule.combine_capacity, dtype=torch.bool, device=device)

    # Since each rank only holds its own expert_out, reconstruct combine_recv the same way the
    # kernel would: gather every rank's expert_out (small in this test) and apply the schedule.
    all_expert_out = [None] * world_size
    for r in range(world_size):
        n = torch.tensor([schedule.recv_capacity] if r == rank else [0], device=device, dtype=torch.int64)
        dist.broadcast(n, src=r, group=group)
        buf = expert_out if r == rank else torch.zeros(int(n.item()), hidden_dim, device=device)
        dist.broadcast(buf, src=r, group=group)
        all_expert_out[r] = buf

    for s in range(world_size):
        peer_rank, local_start, peer_start, valid = combine_tiles_for(s)
        mine = peer_rank == rank
        if not bool(mine.any()):
            continue
        ls, ps, v = local_start[mine], peer_start[mine], valid[mine]
        for i in range(ls.numel()):
            n = int(v[i].item())
            if n == 0:
                continue
            combine_recv[ps[i] : ps[i] + n] = all_expert_out[s][ls[i] : ls[i] + n]
            assert not bool(combine_filled[ps[i] : ps[i] + n].any()), "combine tile overlap!"
            combine_filled[ps[i] : ps[i] + n] = True
    expected_combined = schedule.routed_input * 2.0 + 1.0
    my_tiles = schedule.dispatch_tiles
    combine_ok = True
    for i in range(my_tiles.peer_rank.numel()):
        n = int(my_tiles.valid_rows[i].item())
        if n == 0:
            continue
        src_start = int(my_tiles.local_row_start[i].item())
        ordinal_start = int(my_tiles.own_tile_ordinal[i].item()) * block_m
        got = combine_recv[ordinal_start : ordinal_start + n]
        expected = expected_combined[src_start : src_start + n]
        if not torch.allclose(got, expected, atol=1e-4, rtol=1e-4):
            combine_ok = False

    print(f"[rank {rank}] combine reconstruction correct: {combine_ok}", flush=True)
    dist.barrier()
    all_ok = dispatch_ok and combine_ok
    ok_tensor = torch.tensor([1 if all_ok else 0], device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        assert bool(ok_tensor.item()), "overlapped_moe metadata check failed on some rank"
        print("PASS: overlapped_moe metadata schedule internally consistent (dispatch + combine)", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
