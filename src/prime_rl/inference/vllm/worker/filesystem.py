import itertools
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import Module
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.inference.vllm.worker.weight_transfer import (
    CheckpointIndexMap,
    load_weights_checkpoint_layerwise,
    update_mla_absorbed_weights,
)
from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    ENGINE_LAYOUT,
    MANIFEST,
    anchor_complete,
    as_bits,
    bits_dtype,
    checkpoint_index,
    decode_gaps,
    pin,
    read_anchor_checksums,
    read_checkpoint_range,
    read_delta,
    read_manifest,
    rows_checksum,
    unpin,
    write_file,
)

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_filesystem")

_prefetch_pool = ThreadPoolExecutor(1)
_reader_pool: ThreadPoolExecutor | None = None


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem.

    A broadcast step dir holds a full HF checkpoint (anchor), a sparse delta
    against an earlier step, or both. The worker tracks which step it holds and
    applies a delta only on top of its base step; otherwise it reloads the
    chain's anchor and replays the deltas since."""

    weight_step: int | None = None
    prefetched: tuple[str, Future] | None = None
    index_map: CheckpointIndexMap | None = None
    host_buffer: torch.Tensor | None = None

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def _model(self) -> Module:
        # When enforce_eager=True, model isn't wrapped by torch.compile so no .runnable attr
        model = self.model_runner.model
        if hasattr(model, "runnable"):
            model = model.runnable
        assert isinstance(model, Module)
        return model

    def prefetch_weights_from_path(self, weight_path: str) -> None:
        """Start decoding the sparse delta in ``weight_path`` into pinned host memory."""
        self.prefetched = (weight_path, _prefetch_pool.submit(self._read_delta, Path(weight_path)))

    def _read_delta(
        self, step_dir: Path
    ) -> tuple[list[tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor]], dict[str, int]]:
        """Read this worker's part of a delta into a pinned host buffer reused across steps, in chunks of up to 2^26
        changes of consecutive frames."""
        global _reader_pool
        if _reader_pool is None:
            _reader_pool = ThreadPoolExecutor(16)
        start = time.perf_counter()
        frames, checksums = read_delta(step_dir, _reader_pool, None if self.index_map is None else self.index_map.names)
        read_time = time.perf_counter() - start
        chunks, size = [[]], 0
        for frame in frames:
            if size >= 1 << 26:
                chunks.append([])
                size = 0
            chunks[-1].append(frame)
            size += len(frame[1])
        layout, num_bytes = [], 0
        for chunk in chunks:
            gaps_bytes = -(-sum(gaps.nbytes for _, gaps, _ in chunk) // 4) * 4
            values_bytes = -(-sum(values.nbytes for _, _, values in chunk) // 4) * 4
            layout.append((num_bytes, num_bytes + gaps_bytes))
            num_bytes += gaps_bytes + values_bytes
        if self.host_buffer is None or self.host_buffer.numel() < num_bytes:
            # Twice the delta, so later deltas fit: pinning can stall for tens of seconds on a host that must
            # reclaim page cache first, and this read is on the update's critical path
            if self.host_buffer is not None:
                unpin(self.host_buffer)
            pin_start = time.perf_counter()
            self.host_buffer = pin(num_bytes * 2)
            logger.info(
                f"Pinned a {self.host_buffer.numel() / 2**30:.2f} GiB host buffer for sparse deltas in "
                f"{time.perf_counter() - pin_start:.2f}s"
            )
        buffer = self.host_buffer.numpy()
        copies, pinned = [], []
        for chunk, (gaps_at, values_at) in zip(chunks, layout):
            gaps_end, values_end = gaps_at, values_at
            for _, gaps, values in chunk:
                copies.append(
                    _reader_pool.submit(np.copyto, buffer[gaps_end : gaps_end + gaps.nbytes].view(gaps.dtype), gaps)
                )
                copies.append(
                    _reader_pool.submit(
                        np.copyto, buffer[values_end : values_end + values.nbytes], values.view(np.uint8)
                    )
                )
                gaps_end += gaps.nbytes
                values_end += values.nbytes
            if chunk:
                pinned.append(
                    (
                        [entry for entries, _, _ in chunk for entry in entries],
                        self.host_buffer[gaps_at:gaps_end].view(torch.int32),
                        self.host_buffer[values_at:values_end],
                    )
                )
        for copy in copies:
            copy.result()
        logger.info(
            f"Read sparse weight delta {step_dir.name}: {num_bytes / 2**30:.2f} GiB, "
            f"read+decompress {read_time:.2f}s, copy {time.perf_counter() - start - read_time:.2f}s"
        )
        return pinned, checksums

    def prefetch_weights_done(self) -> bool:
        if self.prefetched is None:
            return True
        if not self.prefetched[1].done():
            return False
        # Raises if the read failed, e.g. on a trainer rank's failure marker
        self.prefetched[1].result()
        return True

    def update_weights_from_path(self, weight_path: str) -> None:
        """Move to the broadcast step in ``weight_path`` (a full HF checkpoint and/or a sparse delta)."""
        step_dir = Path(weight_path)
        step = int(step_dir.name.removeprefix("step_"))
        has_delta = (step_dir / DELTA_DIR / MANIFEST).exists()
        if has_delta or anchor_complete(step_dir):
            self._build_index_map(step_dir)
        # Unknown until this update succeeds: after a failure, the next update reloads and replays
        held_step, self.weight_step = self.weight_step, -1
        if not has_delta:
            self._load_checkpoint(step_dir)
        else:
            base_step = read_manifest(step_dir)["base_step"]
            if base_step is None:
                # The run's first broadcast: the weights the engine booted with
                if held_step is not None:
                    self._load_checkpoint(self._boot_dir())
            elif base_step == held_step:
                self._apply_delta(step_dir)
            else:
                self._replay(step_dir)
        self.weight_step = step

    def _boot_dir(self) -> Path:
        """The local dir of the checkpoint the engine booted from."""
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        folder, *_ = model_loader._prepare_weights(
            self.model_config.model, None, self.model_config.revision, True, None
        )
        return Path(folder)

    def _apply_chunk(
        self, entries: list[tuple[str, int]], gaps: torch.Tensor, values: torch.Tensor, tensors: dict, device
    ) -> None:
        assert self.index_map is not None
        counts = torch.tensor([count for _, count in entries], device=device)
        indices = decode_gaps(gaps.to(device, non_blocking=True), counts)
        values = values.to(device, non_blocking=True)
        offset = num_bytes = 0
        for bits, group in itertools.groupby(entries, key=lambda entry: bits_dtype(tensors[entry[0]][1])):
            run = list(group)
            num_changes = sum(count for _, count in run)
            itemsize = bits.itemsize
            delta = values[num_bytes : num_bytes + num_changes * itemsize]
            # A run after an odd number of narrower values is misaligned for its dtype view
            delta = delta.view(bits) if num_bytes % itemsize == 0 else delta.clone().view(bits)
            self.index_map.apply_delta(run, indices[offset : offset + num_changes], delta)
            offset += num_changes
            num_bytes += num_changes * itemsize

    def _build_index_map(self, step_dir: Path) -> None:
        """At a sparse run's first update, while the engine still holds its boot weights: map checkpoint tensors to
        the params, and report which ones this worker holds in fp8 (and their scale format), so the trainer sends
        their deltas as fp8."""
        if self.index_map is not None:
            return
        if (step_dir / DELTA_DIR / MANIFEST).exists():
            tensors = read_manifest(step_dir)["tensors"]
        else:
            tensors = {
                name: (shape, str(dtype).removeprefix("torch."))
                for name, (shape, dtype, _) in checkpoint_index(step_dir).items()
            }
        model = self._model()
        index_map = CheckpointIndexMap(model, tensors)
        boot = checkpoint_index(self._boot_dir())

        def read_tensor(name: str) -> torch.Tensor:
            shape, dtype, _ = boot[name]
            data = np.empty(prod(shape) * dtype.itemsize, np.uint8)
            read_checkpoint_range(boot[name], 0, prod(shape), data)
            return torch.from_numpy(data).view(dtype).view(shape)

        # The trainer waits for every worker's layout. The world group spans every DP, PP and TP rank
        group = get_world_group()
        layout = {"world_size": group.world_size, "fp8": index_map.fp8_layout(model, read_tensor)}
        self.index_map = index_map
        path = step_dir.parent / ENGINE_LAYOUT / f"rank_{group.rank_in_group}.json"
        path.parent.mkdir(exist_ok=True)
        write_file(path, json.dumps(layout).encode())
        logger.info(f"Wrote the engine layout: {len(layout['fp8'])} checkpoint tensors held in fp8")

    def _replay(self, step_dir: Path) -> None:
        """Load the latest complete anchor at or before ``step_dir`` (or the boot checkpoint, for a
        chain from the run's first broadcast) and replay the deltas since."""
        chain, source = [], step_dir
        while not anchor_complete(source):
            base_step = read_manifest(source)["base_step"]
            if base_step is None:
                break
            chain.append(source)
            source = source.parent / f"step_{base_step}"
        logger.info(f"Replaying {len(chain)} sparse weight deltas on top of {source.name}")
        self._load_checkpoint(source if anchor_complete(source) else self._boot_dir())
        for delta_dir in reversed(chain):
            self._apply_delta(delta_dir)

    def _apply_delta(self, step_dir: Path) -> None:
        start = time.perf_counter()
        prefetched, self.prefetched = self.prefetched, None
        # Always wait: a running prefetch writes into the same host buffer
        delta = prefetched[1].result() if prefetched is not None else None
        if prefetched is None or prefetched[0] != step_dir.as_posix():
            delta = self._read_delta(step_dir)
        delta, checksums = delta
        read_time = time.perf_counter() - start
        tensors = read_manifest(step_dir)["tensors"]
        model = self._model()
        device = next(model.parameters()).device
        for entries, gaps, values in delta:
            self._apply_chunk(entries, gaps, values, tensors, device)
        update_mla_absorbed_weights(model)
        torch.cuda.synchronize()
        apply_time = time.perf_counter() - start - read_time
        self._verify(step_dir, checksums)
        logger.info(
            f"Applied sparse weight delta {step_dir.name}: {sum(len(gaps) for _, gaps, _ in delta)} changes, "
            f"read {read_time:.2f}s (0 if prefetched), apply {apply_time:.2f}s, "
            f"verify {time.perf_counter() - start - read_time - apply_time:.2f}s"
        )

    def _verify(self, step_dir: Path, checksums: dict[str, int]) -> None:
        """Check every tensor this worker holds against the trainer's checksum of the new weights.
        Tensors split across TP workers are summed over the TP group first; without TP, tensors a worker
        holds only part of are not checked."""
        assert self.index_map is not None
        names, totals, partial = self.index_map.held_checksums()
        tp_group = get_tp_group()
        if tp_group.world_size > 1 and partial.any():
            reduced = totals[partial]
            torch.distributed.all_reduce(reduced, group=tp_group.device_group)
            totals[partial] = reduced
        elif partial.any():
            names = [name if not is_partial else None for name, is_partial in zip(names, partial.tolist())]
        mismatched = [
            name
            for name, value in zip(names, totals.tolist())
            if name is not None and name in checksums and value != checksums[name]
        ]
        if mismatched:
            raise RuntimeError(f"Sparse weight delta {step_dir.name} diverged from the trainer: {mismatched[:8]}")

    def _verify_rows(
        self, step_dir: Path, sums: dict[str, torch.Tensor], partial: dict[str, bool], expected: dict[str, int]
    ) -> None:
        """Check the rows read from an anchor or full checkpoint against the trainer's checksums. Tensors this
        worker read only some rows of are summed over the TP group first (not for fp8 tensors, read in whole
        128-row blocks that TP workers may share); without TP they are not checked."""
        if not sums:
            return
        index_map, tp_group = self.index_map, get_tp_group()
        names = sorted(sums)
        totals = torch.stack([sums[name] for name in names])
        partial = torch.tensor([partial[name] for name in names], device=totals.device)
        if tp_group.world_size > 1 and partial.any():
            reduced = totals[partial]
            torch.distributed.all_reduce(reduced, group=tp_group.device_group)
            totals[partial] = reduced
        skip = (
            partial
            if tp_group.world_size == 1
            else partial & torch.tensor([name in index_map.fp8 for name in names], device=totals.device)
        )
        mismatched = [
            name
            for name, value, skipped in zip(names, totals.tolist(), skip.tolist())
            if not skipped and value != expected[name]
        ]
        if mismatched:
            raise RuntimeError(f"Sparse weight checkpoint {step_dir.name} diverged from the trainer: {mismatched[:8]}")

    def _load_checkpoint(self, step_dir: Path) -> None:
        """Reload every weight from the HF checkpoint (or anchor) in ``step_dir``. In a sparse run, each worker reads
        only the rows it holds, straight into its params."""
        model = self._model()
        if self.index_map is None:
            model_loader = get_model_loader(self.load_config)
            assert isinstance(model_loader, DefaultModelLoader)
            local_source = DefaultModelLoader.Source(
                step_dir.as_posix(),
                revision=None,  # TODO: Check that this is correct or if we should use the default (model_config.revision)
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
            )
            weights_iterator = model_loader._get_weights_iterator(local_source)
            load_weights_checkpoint_layerwise(model, weights_iterator, self.model_runner.model_config, self.vllm_config)
            return
        start = time.perf_counter()
        index_map, index = self.index_map, checkpoint_index(step_dir)
        device = next(model.parameters()).device
        # Checksums of anchors and full checkpoints written by the trainer (none for the boot checkpoint)
        expected = read_anchor_checksums(step_dir)
        sums: dict[str, torch.Tensor] = {}
        partial: dict[str, bool] = {}

        def read(name: str, begin: int, end: int) -> torch.Tensor:
            num_cols, dtype = index_map.num_cols[name], index[name][1]
            data = np.empty((end - begin) * num_cols * dtype.itemsize, np.uint8)
            read_checkpoint_range(index[name], begin * num_cols, end * num_cols, data)
            return torch.from_numpy(data).view(dtype).view(end - begin, num_cols)

        ranges = list(index_map.row_ranges().items())
        num_bytes = 0
        with ThreadPoolExecutor(16) as pool:
            # A bounded window of reads ahead of the GPU writes
            for window in itertools.batched(ranges, 64):
                for (name, (begin, end)), rows in zip(window, pool.map(lambda item: read(item[0], *item[1]), window)):
                    rows = rows.to(device)
                    index_map.load_rows(name, rows, begin)
                    if name in expected:
                        num_cols = index_map.num_cols[name]
                        sums[name] = rows_checksum(as_bits(rows).reshape(-1), begin * num_cols)
                        partial[name] = (end - begin) * num_cols != index_map.numel[name]
                    num_bytes += rows.nbytes
        update_mla_absorbed_weights(model)
        torch.cuda.synchronize()
        self._verify_rows(step_dir, sums, partial, expected)
        logger.info(
            f"Loaded this worker's {num_bytes / 2**30:.2f} GiB of {step_dir.name} in {time.perf_counter() - start:.1f}s"
        )
