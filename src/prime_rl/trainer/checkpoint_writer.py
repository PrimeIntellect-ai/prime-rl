from collections.abc import Iterator
from dataclasses import replace
from math import prod
from pathlib import Path

from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItem
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future


class CheckpointFileWriter(FileSystemWriter):
    """Write one shard at a time to bound the DCP staging tensors retained per rank."""

    # shortcut: bound staged tensors per file until pytorch/pytorch#195982 is fixed in our torch build.
    _shard_bytes = 1 << 30

    def __init__(self, path: Path) -> None:
        # Blocking GPU-to-CPU copies avoid leaving each shard's pinned buffers in the CUDA host cache.
        super().__init__(path, per_thread_copy_ahead=0)

    def write_data(self, plan: SavePlan, planner: SavePlanner) -> Future[list[WriteResult]]:
        results = []
        for shard, items in enumerate(self._shards(plan.items)):
            storage_data = plan.storage_data
            if shard:
                storage_data = replace(storage_data, prefix=f"{storage_data.prefix}part{shard}_")
            shard_plan = replace(plan, items=items, storage_data=storage_data)
            results.extend(super().write_data(shard_plan, planner).wait())
        future = Future()
        future.set_result(results)
        return future

    def _shards(self, items: list[WriteItem]) -> Iterator[list[WriteItem]]:
        start = 0
        size = 0
        for index, item in enumerate(items):
            tensor = item.tensor_data
            # Use the local chunk's size. A tensor larger than the target gets its own shard.
            item_bytes = prod(tensor.chunk.sizes) * tensor.properties.dtype.itemsize if tensor is not None else 0
            if index > start and size + item_bytes > self._shard_bytes:
                yield items[start:index]
                start = index
                size = 0
            size += item_bytes
        yield items[start:]
