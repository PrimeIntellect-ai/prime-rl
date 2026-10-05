"""Route tensor replay sources onto sharded trainer memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prime_rl.transports.weights.nixl.graph import TensorReplayPlan
from prime_rl.transports.weights.nixl.trainer_tensor_table import TrainerTensor


@dataclass(frozen=True)
class TensorRoute:
    agent: int
    source_addr: int
    destination_addr: int
    nbytes: int


def route_sharded_tensor(
    plan: TensorReplayPlan,
    source: TrainerTensor,
    destination: torch.Tensor,
) -> list[TensorRoute]:
    """Route trainer views into destination slices, coalescing runs contiguous on both sides."""
    numel = 1
    for size in plan.source_shape:
        numel *= size
    if numel == 0:
        return []

    dims = [
        (size, source_step, destination_step)
        for size, source_step, destination_step in zip(plan.source_shape, plan.source_stride, destination.stride())
        if size != 1
    ]
    if any(source_step < 0 for _, source_step, _ in dims):
        raise NotImplementedError("negative strides are not supported")

    run_elements = 1
    split_at = len(dims)
    while split_at and dims[split_at - 1][1] == dims[split_at - 1][2] == run_elements:
        run_elements *= dims[split_at - 1][0]
        split_at -= 1
    outer_dims = dims[:split_at]

    routes: list[TensorRoute] = []
    itemsize = destination.element_size()
    destination_addr = destination.data_ptr()

    def route_run(element_offset: int, destination_offset: int, element_count: int) -> None:
        position = element_offset
        remaining = element_count
        while remaining:
            shard = next(
                (shard for shard in source.shards if shard.offset <= position < shard.offset + shard.numel),
                None,
            )
            if shard is None:
                raise RuntimeError(f"no trainer shard owns element {position}")
            take = min(remaining, shard.offset + shard.numel - position)
            nbytes = take * itemsize
            routes.append(
                TensorRoute(
                    agent=shard.agent,
                    source_addr=shard.addr + (position - shard.offset) * itemsize,
                    destination_addr=destination_addr + destination_offset * itemsize,
                    nbytes=nbytes,
                )
            )
            position += take
            remaining -= take
            destination_offset += take

    def route_dimension(dim: int, element_offset: int, destination_offset: int) -> None:
        if dim == len(outer_dims):
            route_run(element_offset, destination_offset, run_elements)
            return
        size, source_step, destination_step = outer_dims[dim]
        for index in range(size):
            route_dimension(
                dim + 1, element_offset + index * source_step, destination_offset + index * destination_step
            )

    route_dimension(0, plan.source_offset, 0)
    return routes
