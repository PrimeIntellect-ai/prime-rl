"""Map logical tensor regions onto trainer-owned registered memory."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod


@dataclass(frozen=True)
class SourceShard:
    """One rectangular shard of a full logical Prime tensor.

    The wire representation already supports arbitrary source segmentation.
    The first implementation accepts only dim-0 shards because current FSDP
    and expert-parallel parameters use that layout. Keeping offsets and shapes
    as tuples makes extending the planner independent of the wire protocol.
    """

    agent: int
    global_offset: tuple[int, ...]
    shape: tuple[int, ...]
    address: int
    device_id: int


def validate_dim0_shards(full_shape: tuple[int, ...], shards: tuple[SourceShard, ...]) -> None:
    if not full_shape:
        raise ValueError("scalar source tensors are not supported")
    ordered = sorted(shards, key=lambda shard: shard.global_offset[0])
    next_row = 0
    for shard in ordered:
        if len(shard.shape) != len(full_shape) or len(shard.global_offset) != len(full_shape):
            raise ValueError(f"source shard rank does not match full shape {full_shape}: {shard}")
        if shard.global_offset[0] != next_row:
            raise ValueError(f"dim-0 shards must tile without gaps or overlap; expected row {next_row}: {shard}")
        if any(offset != 0 for offset in shard.global_offset[1:]):
            raise ValueError(f"only dim-0 source sharding is supported: {shard}")
        if shard.shape[1:] != full_shape[1:]:
            raise ValueError(f"only dim-0 source sharding is supported: {shard}")
        next_row += shard.shape[0]
    if next_row != full_shape[0]:
        raise ValueError(f"source shards cover {next_row} of {full_shape[0]} dim-0 rows")


def route_source_region(
    region_runs: list[tuple[int, int]],
    full_shape: tuple[int, ...],
    shards: tuple[SourceShard, ...],
    itemsize: int,
) -> list[tuple[int, int, int, int]]:
    """Route full-tensor element runs to ``(agent, address, bytes, device)``."""

    validate_dim0_shards(full_shape, shards)
    row_numel = prod(full_shape[1:])
    bounds = [
        (
            shard.global_offset[0] * row_numel,
            (shard.global_offset[0] + shard.shape[0]) * row_numel,
            shard,
        )
        for shard in sorted(shards, key=lambda shard: shard.global_offset[0])
    ]
    pieces: list[tuple[int, int, int, int]] = []
    for elem_offset, numel in region_runs:
        position = elem_offset
        remaining = numel
        while remaining:
            match = next(((lo, hi, shard) for lo, hi, shard in bounds if lo <= position < hi), None)
            if match is None:
                raise ValueError(f"no source shard owns logical element {position}")
            lower, upper, shard = match
            take = min(remaining, upper - position)
            address = shard.address + (position - lower) * itemsize
            pieces.append((shard.agent, address, take * itemsize, shard.device_id))
            position += take
            remaining -= take
    return pieces


def zip_source_destination(
    source: list[tuple[int, int, int]], destination: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]]:
    """Zip source byte pieces with destination byte runs.

    Returns ``(agent, source_address, destination_address, num_bytes)``.
    """

    units: list[tuple[int, int, int, int]] = []
    source_index = destination_index = 0
    source_offset = destination_offset = 0
    while source_index < len(source) and destination_index < len(destination):
        agent, source_address, source_length = source[source_index]
        destination_address, destination_length = destination[destination_index]
        length = min(source_length - source_offset, destination_length - destination_offset)
        units.append(
            (
                agent,
                source_address + source_offset,
                destination_address + destination_offset,
                length,
            )
        )
        source_offset += length
        destination_offset += length
        if source_offset == source_length:
            source_index += 1
            source_offset = 0
        if destination_offset == destination_length:
            destination_index += 1
            destination_offset = 0

    source_total = sum(length for _, _, length in source)
    destination_total = sum(length for _, length in destination)
    if source_total != destination_total or source_index != len(source) or destination_index != len(destination):
        raise ValueError(f"source/destination byte mismatch: source={source_total}, destination={destination_total}")
    return units
