"""Build HF logical weights directly over sharded Prime trainer storage."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Protocol

import torch

from prime_rl.weight_transfer.chains import region_elem_runs, resolve_chain_region
from prime_rl.weight_transfer.lazy import BakeRecorder, LazyWeight
from prime_rl.weight_transfer.sharding import SourceShard, route_source_region
from prime_rl.weight_transfer.wire import PublishedTensor, TensorSegment


class WeightConverter(Protocol):
    def convert_to_hf(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True)
class SourceTensor:
    """A full logical Prime tensor backed by selected trainer shards."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    shards: tuple[SourceShard, ...]


def _coalesce_segments(segments: list[TensorSegment], itemsize: int) -> tuple[TensorSegment, ...]:
    coalesced: list[TensorSegment] = []
    for segment in segments:
        if coalesced:
            previous = coalesced[-1]
            contiguous = (
                previous.agent == segment.agent
                and previous.device_id == segment.device_id
                and previous.logical_offset + previous.numel == segment.logical_offset
                and previous.address + previous.numel * itemsize == segment.address
            )
            if contiguous:
                coalesced[-1] = TensorSegment(
                    agent=previous.agent,
                    logical_offset=previous.logical_offset,
                    numel=previous.numel + segment.numel,
                    address=previous.address,
                    device_id=previous.device_id,
                )
                continue
        coalesced.append(segment)
    return tuple(coalesced)


def _segments_for_view(source: SourceTensor, view: LazyWeight) -> tuple[TensorSegment, ...]:
    itemsize = torch.empty((), dtype=source.dtype).element_size()
    offset, shape, stride = resolve_chain_region(source.shape, source.dtype, view.op_chain)
    if shape != tuple(view.shape):
        raise ValueError(f"symbolic conversion shape mismatch for {view.source_name}: {shape} != {tuple(view.shape)}")
    pieces = route_source_region(
        region_elem_runs(offset, shape, stride),
        source.shape,
        source.shards,
        itemsize,
    )
    logical_offset = 0
    segments: list[TensorSegment] = []
    for agent, address, num_bytes, device_id in pieces:
        if num_bytes % itemsize:
            raise ValueError(f"unaligned source segment for {view.source_name}: {num_bytes} bytes")
        numel = num_bytes // itemsize
        segments.append(
            TensorSegment(
                agent=agent,
                logical_offset=logical_offset,
                numel=numel,
                address=address,
                device_id=device_id,
            )
        )
        logical_offset += numel
    if logical_offset != prod(view.shape):
        raise ValueError(
            f"published segments for {view.source_name} cover {logical_offset} elements; expected {prod(view.shape)}"
        )
    return _coalesce_segments(segments, itemsize)


def publish_hf_tensors(converter: WeightConverter, sources: tuple[SourceTensor, ...]) -> tuple[PublishedTensor, ...]:
    """Run the real Prime->HF conversion chain symbolically over source shards.

    The result contains HF names and logical shapes, with each logical element
    mapped directly to registered trainer memory. No tensor values are read and
    no full model tensor is allocated.
    """

    recorder = BakeRecorder()
    by_name = {source.name: source for source in sources}
    if len(by_name) != len(sources):
        raise ValueError("source tensor names must be unique")
    state_dict: dict[str, torch.Tensor] = {
        source.name: LazyWeight(
            name=source.name,
            shape=torch.Size(source.shape),
            dtype=source.dtype,
            device=torch.device("meta"),
            recorder=recorder,
        )
        for source in sources
    }
    converted = converter.convert_to_hf(state_dict)

    published: list[PublishedTensor] = []
    for name, value in sorted(converted.items()):
        if not isinstance(value, LazyWeight):
            raise TypeError(
                f"Prime->HF conversion for {name!r} materialized {type(value).__name__}; "
                "NIXL publication only supports symbolic layout operations"
            )
        source = by_name.get(value.source_name)
        if source is None:
            raise KeyError(f"converted tensor {name!r} references unknown source {value.source_name!r}")
        published.append(
            PublishedTensor(
                name=name,
                dtype=str(value.dtype),
                shape=tuple(value.shape),
                segments=_segments_for_view(source, value),
            )
        )
    return tuple(published)


def route_published_region(
    tensor: PublishedTensor,
    region_runs: list[tuple[int, int]],
    itemsize: int,
) -> list[tuple[int, int, int]]:
    """Route logical HF element runs to trainer ``(agent, address, bytes)``."""

    segments = sorted(tensor.segments, key=lambda segment: segment.logical_offset)
    pieces: list[tuple[int, int, int]] = []
    for elem_offset, numel in region_runs:
        position = elem_offset
        remaining = numel
        while remaining:
            segment = next(
                (
                    candidate
                    for candidate in segments
                    if candidate.logical_offset <= position < candidate.logical_offset + candidate.numel
                ),
                None,
            )
            if segment is None:
                raise ValueError(f"published tensor {tensor.name!r} has no segment for logical element {position}")
            within = position - segment.logical_offset
            take = min(remaining, segment.numel - within)
            pieces.append((segment.agent, segment.address + within * itemsize, take * itemsize))
            position += take
            remaining -= take
    return pieces
