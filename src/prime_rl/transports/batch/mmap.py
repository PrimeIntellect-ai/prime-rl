"""Bounded disk spool with parallel writers and optional resident CPU lookahead."""

import asyncio
import json
import shutil
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import msgspec
import numpy as np
import torch

from prime_rl.configs.shared import MMapTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.tensors import TensorMicroBatch, micro_batch_to_tensor
from prime_rl.transports.batch.types import MicroBatch, MMRefs
from prime_rl.utils.pathing import get_batch_dir, get_step_path, sync_wait_for_path


def compact_routing_ids(ids: torch.Tensor) -> torch.Tensor:
    """Losslessly narrow expert IDs, including negative sentinel values."""
    array = ids.numpy()
    if array.dtype.kind not in "iu":
        raise ValueError("Routing IDs must be integers")
    lo, hi = (int(array.min()), int(array.max())) if array.size else (0, 0)
    choices = (np.uint8, np.uint16, np.int32, np.int64) if lo >= 0 else (np.int16, np.int32, np.int64)
    dtype = next(dtype for dtype in choices if np.iinfo(dtype).min <= lo <= hi <= np.iinfo(dtype).max)
    return torch.from_numpy(array.astype(dtype, copy=True))


def loss_token_counts(batch: TensorMicroBatch) -> tuple[int, int, int]:
    mask, rl = batch["loss_mask"], batch["rl_weights"]
    return (
        int((mask & (rl != 0)).sum()) if rl is not None else int(mask.sum()),
        int((batch["ce_weights"] != 0).sum()) if batch["ce_weights"] is not None else 0,
        int((batch["ref_kl_weights"] != 0).sum()) if batch["ref_kl_weights"] is not None else 0,
    )


class MMapMicroBatches(Sequence[TensorMicroBatch]):
    """Map individual accesses, or prefetch a bounded window into RAM when iterating."""

    def __init__(self, step_dir: Path, data_rank: int, reader_id: int, prefetch_batches: int = 0):
        manifest = json.loads((step_dir / "ready.json").read_text())
        self.step_dir = step_dir
        self.data_rank = data_rank
        self.reader_id = reader_id
        self.rank = manifest["ranks"][data_rank]
        self.loss_counts = tuple(self.rank["loss_counts"])
        self.seq_len = self.rank["seq_len"]
        self.closed = False
        self.prefetch_batches = prefetch_batches
        self._executor: ThreadPoolExecutor | None = None
        self._pending: deque[Future[TensorMicroBatch]] = deque()

    def __len__(self) -> int:
        return self.rank["count"]

    def __getitem__(self, index: int) -> TensorMicroBatch:
        if self.closed:
            raise RuntimeError("Batch has already been released")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._load(index, mmap=True)

    def _load(self, index: int, *, mmap: bool) -> TensorMicroBatch:
        # MAP_PRIVATE tensor storage shares clean file pages across local CP ranks.
        # Training must not modify these CPU tensors in place.
        batch = torch.load(
            self.step_dir / f"rank_{self.data_rank}" / f"{index}.pt",
            mmap=mmap,
            weights_only=True,
            map_location="cpu",
        )
        if batch.get("mm_refs") is not None:
            batch["mm_refs"] = msgspec.msgpack.decode(batch["mm_refs"], type=MMRefs)
        return batch

    def __iter__(self) -> Iterator[TensorMicroBatch]:
        if not self.prefetch_batches:
            for index in range(len(self)):
                yield self[index]
            return
        if self.closed:
            raise RuntimeError("Batch has already been released")
        if self._executor is not None:
            raise RuntimeError("Prefetched batches support one active iterator")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmap-prefetch")
        try:
            for index in range(min(self.prefetch_batches, len(self))):
                self._pending.append(self._executor.submit(self._load, index, mmap=False))
            for index in range(len(self)):
                if self.closed:
                    raise RuntimeError("Batch has already been released")
                yield self._next_prefetched(index)
        finally:
            self._stop_prefetch()

    def _next_prefetched(self, index: int) -> TensorMicroBatch:
        batch = self._pending.popleft().result()
        next_index = index + self.prefetch_batches
        if next_index < len(self):
            # Eager loading faults the tensor data into RAM, unlike mmap alone.
            self._pending.append(self._executor.submit(self._load, next_index, mmap=False))
        return batch

    def _stop_prefetch(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._pending.clear()

    def close(self) -> None:
        if not self.closed:
            # No reader may still be opening files when the spool is reclaimed.
            self._stop_prefetch()
            (self.step_dir / f"rank_{self.data_rank}.reader_{self.reader_id}.done").touch()
            self.closed = True


class MMapBatchSender(BatchSender):
    def __init__(self, output_dir: Path, data_world_size: int, current_step: int, config: MMapTransportConfig):
        super().__init__(output_dir, data_world_size)
        if config.max_pending_bytes < config.max_batch_bytes:
            raise ValueError("mmap max_pending_bytes must be at least max_batch_bytes")
        self.batch_dir = get_batch_dir(output_dir)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.current_step = current_step
        self.config = config

    def pending_bytes(self) -> int:
        pending = 0
        for ready in self.batch_dir.glob("step_*/ready.json"):
            if ready.parent.suffix == ".tmp":
                continue
            manifest = json.loads(ready.read_text())
            if all(
                (ready.parent / f"rank_{rank}.reader_{reader}.done").exists()
                for rank in range(len(manifest["ranks"]))
                for reader in range(manifest["readers_per_rank"])
            ):
                shutil.rmtree(ready.parent)
            else:
                pending += manifest["bytes"]
        return pending

    async def send(self, grid: list[list[MicroBatch]]) -> None:
        if len(grid) != self.data_world_size or not grid or not grid[0]:
            raise ValueError("Expected a nonempty microbatch list for every data rank")
        if any(len(rank) != len(grid[0]) for rank in grid):
            raise ValueError("All data ranks must have the same microbatch count")
        # Reserve a whole bounded write slot before producing files. Waiting yields
        # to the orchestrator's policy updates; a full spool never drops a batch.
        while await asyncio.to_thread(self.pending_bytes) + self.config.max_batch_bytes > self.config.max_pending_bytes:
            await asyncio.sleep(0.2)
        await asyncio.to_thread(self._write, grid)
        self.current_step += 1

    def _write(self, grid: list[list[MicroBatch]]) -> None:
        started = time.perf_counter()
        destination = get_step_path(self.batch_dir, self.current_step)
        staging = destination.with_suffix(".tmp")
        staging.mkdir()
        manifest = {"bytes": 0, "readers_per_rank": self.config.readers_per_rank, "ranks": []}
        byte_lock = Lock()
        failed = Event()
        workers = min(self.config.write_workers, len(grid))

        def write_rank(rank: int):
            microbatches = grid[rank]
            try:
                rank_dir = staging / f"rank_{rank}"
                rank_dir.mkdir()
                counts = [0, 0, 0]
                for index, microbatch in enumerate(microbatches):
                    if failed.is_set():
                        return None
                    batch = micro_batch_to_tensor(microbatch)
                    if batch["mm_refs"] is not None:
                        batch["mm_refs"] = msgspec.msgpack.encode(batch["mm_refs"])
                    if batch["routed_experts"] is not None:
                        batch["routed_experts"] = compact_routing_ids(batch["routed_experts"])
                    counts = [a + b for a, b in zip(counts, loss_token_counts(batch))]
                    path = rank_dir / f"{index}.pt"
                    torch.save(batch, path)
                    size = path.stat().st_size
                    del batch
                    with byte_lock:
                        manifest["bytes"] += size
                        if manifest["bytes"] > self.config.max_batch_bytes:
                            raise ValueError(f"Batch exceeds mmap max_batch_bytes={self.config.max_batch_bytes}")
                return {"count": len(microbatches), "seq_len": len(microbatches[0].input_ids), "loss_counts": counts}
            except Exception:
                failed.set()
                raise

        try:
            # Threads share the source grid rather than copying it into worker processes.
            # Join every writer before publishing or cleaning up a failed batch.
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mmap-write") as executor:
                manifest["ranks"] = list(executor.map(write_rank, range(len(grid))))
            (staging / "ready.json").write_text(json.dumps(manifest))
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        self.logger.info(
            f"Published mmap batch {self.current_step}: {manifest['bytes'] / 1024**3:.2f} GiB, "
            f"{len(grid[0])} microbatches/rank, {self.config.readers_per_rank} readers/rank, "
            f"{workers} writers in {time.perf_counter() - started:.2f}s"
        )

    def close(self) -> None:
        self.pending_bytes()


class MMapBatchReceiver(BatchReceiver):
    def __init__(self, output_dir: Path, data_rank: int, current_step: int, prefetch_batches: int = 0):
        super().__init__(output_dir, data_rank)
        self.batch_dir = get_batch_dir(output_dir)
        self.current_step = current_step
        self.reader_id = 0
        self.prefetch_batches = prefetch_batches

    def _path(self) -> Path:
        return get_step_path(self.batch_dir, self.current_step)

    def wait(self) -> None:
        sync_wait_for_path(self._path() / "ready.json")

    def can_receive(self) -> bool:
        return (self._path() / "ready.json").exists()

    def receive(self) -> MMapMicroBatches:
        batch = MMapMicroBatches(self._path(), self.data_rank, self.reader_id, self.prefetch_batches)
        self.current_step += 1
        return batch
