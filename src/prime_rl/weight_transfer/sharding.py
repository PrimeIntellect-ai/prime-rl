"""Route logical tensor regions onto dim-0 trainer shards."""

from __future__ import annotations

from prime_rl.weight_transfer.wire import TrainerShard


def route_region(
    region_runs: list[tuple[int, int]],
    shards: list[TrainerShard],
    tensor_numel: int,
    itemsize: int,
) -> list[tuple[int, int, int]]:
    """Map full-tensor element runs to ``(agent, address, bytes)`` pieces."""
    ordered = sorted(shards, key=lambda shard: shard.offset)
    bounds: list[tuple[int, int, TrainerShard]] = []
    for index, shard in enumerate(ordered):
        end = ordered[index + 1].offset if index + 1 < len(ordered) else tensor_numel
        if shard.offset < 0 or end <= shard.offset or end > tensor_numel:
            raise ValueError(
                f"invalid trainer shard range [{shard.offset}, {end}) for tensor with {tensor_numel} elements"
            )
        bounds.append((shard.offset, end, shard))

    pieces: list[tuple[int, int, int]] = []
    for elem_offset, num_elems in region_runs:
        pos = elem_offset
        remaining = num_elems
        while remaining:
            owner = next(((start, end, shard) for start, end, shard in bounds if start <= pos < end), None)
            if owner is None:
                raise RuntimeError(f"no trainer shard owns element {pos}")
            owner_start, owner_end, shard = owner
            take = min(remaining, owner_end - pos)
            local_elem = pos - owner_start
            pieces.append((shard.agent, shard.addr + local_elem * itemsize, take * itemsize))
            pos += take
            remaining -= take
    return pieces


def zip_src_dst(
    src_pieces: list[tuple[int, int, int]], dst_runs: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    """Zip equal byte streams into ``(agent, src, dst, bytes)`` units."""
    units: list[tuple[int, int, int, int]] = []
    src_idx = dst_idx = 0
    src_offset = dst_offset = 0
    while src_idx < len(src_pieces) and dst_idx < len(dst_runs):
        agent, src_addr, src_len = src_pieces[src_idx]
        dst_addr, dst_len = dst_runs[dst_idx]
        length = min(src_len - src_offset, dst_len - dst_offset)
        units.append((agent, src_addr + src_offset, dst_addr + dst_offset, length))
        src_offset += length
        dst_offset += length
        if src_offset == src_len:
            src_idx += 1
            src_offset = 0
        if dst_offset == dst_len:
            dst_idx += 1
            dst_offset = 0

    src_total = sum(length for _, _, length in src_pieces)
    dst_total = sum(length for _, length in dst_runs)
    if src_total != dst_total or src_idx != len(src_pieces) or dst_idx != len(dst_runs):
        raise ValueError(f"source/destination byte mismatch: source={src_total}, destination={dst_total}")
    return units
