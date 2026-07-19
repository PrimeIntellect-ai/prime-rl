"""Route logical tensor regions onto trainer shards."""

from __future__ import annotations

from prime_rl.weight_transfer.chains import UnsupportedOpError
from prime_rl.weight_transfer.wire import TrainerShard

_MAX_RUNS_PER_COPY = 1 << 16


def route_region(
    offset: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    shards: list[TrainerShard],
    itemsize: int,
) -> list[tuple[int, int, int]]:
    """Map a strided source view to ``(agent, address, bytes)`` pieces."""
    numel = 1
    for size in shape:
        numel *= size
    if numel == 0:
        return []

    dims = [(size, step) for size, step in zip(shape, stride) if size != 1]
    if any(step < 0 for _, step in dims):
        raise UnsupportedOpError("negative strides are not supported")

    run_elements = 1
    split_at = len(dims)
    while split_at and dims[split_at - 1][1] == run_elements:
        run_elements *= dims[split_at - 1][0]
        split_at -= 1
    outer_dims = dims[:split_at]

    run_count = 1
    for size, _ in outer_dims:
        run_count *= size
    if run_count > _MAX_RUNS_PER_COPY:
        raise UnsupportedOpError(
            f"region shape={shape}, stride={stride} requires {run_count} RDMA runs "
            f"(maximum {_MAX_RUNS_PER_COPY})"
        )

    region_runs: list[tuple[int, int]] = []

    def add_runs(dim: int, run_offset: int) -> None:
        if dim == len(outer_dims):
            region_runs.append((run_offset, run_elements))
            return
        size, step = outer_dims[dim]
        for index in range(size):
            add_runs(dim + 1, run_offset + index * step)

    add_runs(0, offset)

    ordered = sorted(shards, key=lambda shard: shard.offset)
    bounds: list[tuple[int, int, TrainerShard]] = []
    for shard in ordered:
        end = shard.offset + shard.numel
        if shard.offset < 0 or shard.numel <= 0:
            raise ValueError(f"invalid trainer shard range [{shard.offset}, {end})")
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
