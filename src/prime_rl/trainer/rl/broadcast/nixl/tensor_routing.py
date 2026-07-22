"""Route tensor replay sources onto sharded trainer memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prime_rl.trainer.rl.broadcast.nixl.graph import TensorReplayPlan
from prime_rl.trainer.rl.broadcast.nixl.trainer_tensor_table import TrainerTensor


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
    """Route a strided trainer view into a contiguous destination tensor."""
    numel = 1
    for size in plan.source_shape:
        numel *= size
    if numel == 0:
        return []

    dims = [(size, step) for size, step in zip(plan.source_shape, plan.source_stride) if size != 1]
    if any(step < 0 for _, step in dims):
        raise NotImplementedError("negative strides are not supported")

    run_elements = 1
    split_at = len(dims)
    while split_at and dims[split_at - 1][1] == run_elements:
        run_elements *= dims[split_at - 1][0]
        split_at -= 1
    outer_dims = dims[:split_at]

    routes: list[TensorRoute] = []
    itemsize = destination.element_size()
    destination_addr = destination.data_ptr()
    destination_offset = 0

    def route_run(element_offset: int, element_count: int) -> None:
        nonlocal destination_offset
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
                    destination_addr=destination_addr + destination_offset,
                    nbytes=nbytes,
                )
            )
            position += take
            remaining -= take
            destination_offset += nbytes

    def route_dimension(dim: int, element_offset: int) -> None:
        if dim == len(outer_dims):
            route_run(element_offset, run_elements)
            return
        size, step = outer_dims[dim]
        for index in range(size):
            route_dimension(dim + 1, element_offset + index * step)

    route_dimension(0, plan.source_offset)
    return routes
