"""Route a resolved op-chain region onto the trainer's dim-0 shards.

A worker's lazy bake resolves each destination param slice to a strided
region of the *full logical* source tensor. The trainer serves that tensor
as contiguous dim-0 shards across its ranks (FSDP / expert-parallel, both
dim-0). This module maps the region's bytes onto those shards — splitting at
shard boundaries — so each piece becomes one RDMA READ from the owning rank,
and zips the source pieces against the destination's byte runs.

All sharding is dim-0 only, so a shard owns a contiguous block of dim-0 rows
``[row_start, row_start + num_rows)`` and the trailing dims are whole; a
contiguous run in full-tensor element order therefore crosses shard
boundaries only at multiples of the per-row element count.
"""

from __future__ import annotations

from prime_rl.weight_transfer.wire import TrainerShard


def route_region(
    region_runs: list[tuple[int, int]],
    shards: list[TrainerShard],
    row_numel: int,
    itemsize: int,
) -> list[tuple[int, int, int]]:
    """Map full-tensor element runs onto shards.

    Args:
        region_runs: ``(elem_offset, num_elems)`` runs in full-tensor C order
            (from :func:`~prime_rl.weight_transfer.chains.region_elem_runs`).
        shards: the tensor's dim-0 shards (need not be sorted).
        row_numel: elements per dim-0 row (``prod(full_shape[1:])``).
        itemsize: bytes per element.

    Returns ``(agent, src_addr, num_bytes)`` pieces in the same C order,
    splitting any run that spans multiple shards. Raises if a run touches a
    dim-0 row no shard owns.
    """
    ordered = sorted(shards, key=lambda s: s.row_start)
    bounds = [(s.row_start * row_numel, (s.row_start + s.num_rows) * row_numel, s) for s in ordered]
    pieces: list[tuple[int, int, int]] = []
    for elem_off, num_elems in region_runs:
        pos, remaining = elem_off, num_elems
        while remaining > 0:
            shard = next((s for lo, hi, s in bounds if lo <= pos < hi), None)
            if shard is None:
                raise RuntimeError(f"no trainer shard owns element {pos} (dim-0 row {pos // row_numel})")
            shard_hi = (shard.row_start + shard.num_rows) * row_numel
            take = min(remaining, shard_hi - pos)
            local_elem = pos - shard.row_start * row_numel
            pieces.append((shard.agent, shard.addr + local_elem * itemsize, take * itemsize))
            pos += take
            remaining -= take
    return pieces


def zip_src_dst(
    src_pieces: list[tuple[int, int, int]],
    dst_runs: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Zip source pieces and destination runs of one copy into transfer units.

    Both describe the same byte stream in C order. Returns
    ``(agent, src_addr, dst_addr, num_bytes)`` tuples, splitting at either
    side's boundaries. Totals must match.
    """
    units: list[tuple[int, int, int, int]] = []
    i = j = 0
    src_off = dst_off = 0
    while i < len(src_pieces) and j < len(dst_runs):
        agent, src_addr, src_len = src_pieces[i]
        dst_addr, dst_len = dst_runs[j]
        length = min(src_len - src_off, dst_len - dst_off)
        units.append((agent, src_addr + src_off, dst_addr + dst_off, length))
        src_off += length
        dst_off += length
        if src_off == src_len:
            i += 1
            src_off = 0
        if dst_off == dst_len:
            j += 1
            dst_off = 0
    src_total = sum(n for _, _, n in src_pieces)
    dst_total = sum(n for _, n in dst_runs)
    if src_total != dst_total or i != len(src_pieces) or j != len(dst_runs):
        raise ValueError(f"src/dst byte-length mismatch: src={src_total} dst={dst_total}")
    return units
