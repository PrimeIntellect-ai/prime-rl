import itertools
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, NamedTuple

import numpy as np
import torch
from torch.nn import Module
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.inference.vllm.worker.weight_transfer import (
    CheckpointIndexMap,
    WeightRPCMixin,
    checkpoint_weights,
    get_weight_model,
    load_weights_checkpoint_layerwise,
    update_mla_absorbed_weights,
)
from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    ENGINE_LAYOUT,
    MANIFEST,
    anchor_complete,
    bits_dtype,
    checkpoint_index,
    decode_gaps,
    pin,
    read_anchor_checksums,
    read_checkpoint_range,
    read_delta,
    read_manifest,
    revalidate,
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


class DeltaChunk(NamedTuple):
    """Changes to apply at once, in pinned host memory: ``(checkpoint tensor, count)`` entries, their gaps
    (``decode_gaps``) and value bytes."""

    entries: list[tuple[str, int]]
    gaps: torch.Tensor
    values: torch.Tensor


_prefetch_pool = ThreadPoolExecutor(1)
_reader_pool: ThreadPoolExecutor | None = None


class FileSystemWeightUpdateWorker(WeightRPCMixin, Worker):
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
        return get_weight_model(self.model_runner, "model")

    def prefetch_weights_from_path(self, weight_path: str) -> None:
        """Start decoding the sparse delta in ``weight_path`` into pinned host memory."""
        self.prefetched = (weight_path, _prefetch_pool.submit(self._read_delta, Path(weight_path)))

    def _read_delta(self, step_dir: Path) -> tuple[list[DeltaChunk], dict[str, int]]:
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
                    DeltaChunk(
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
        """Move to the broadcast step in ``weight_path`` (a full HF checkpoint and/or a sparse delta).

        The TP group's workers run collectives while they verify, so they must take the same steps: TP rank 0
        plans the update (which checkpoint to load, which deltas to apply) and the others follow its plan. A
        failure on any worker fails the update on all of them, so they all replay the same way next time."""
        step_dir = Path(weight_path)
        step = int(step_dir.name.removeprefix("step_"))
        tp_group = get_tp_group()
        plan, error = None, None
        if tp_group.rank_in_group == 0:
            try:
                plan = self._plan_update(step_dir)
            except Exception as e:  # re-raised below, after the others got the failure instead of a plan
                error = e
        if tp_group.world_size > 1:
            plan = tp_group.broadcast_object(plan if error is None else {"failed": repr(error)}, src=0)
        if error is not None:
            raise error
        if "failed" in plan:
            raise RuntimeError(f"TP rank 0 could not plan the update to {step_dir.name}: {plan['failed']}")
        # Unknown until this update succeeds: after a failure, the next update reloads and replays
        self.weight_step = -1
        check = None
        try:
            if plan["index_map"] is not None:
                self._build_index_map(Path(plan["index_map"]))
                if plan["write_layout"]:
                    # A full broadcast: the trainer takes the layout at its first one, also after a restart
                    self._write_layout(step_dir.parent)
            check = self._run_plan(plan)
        except Exception as e:  # re-raised below, once every worker of the TP group knows
            error = e
        self._raise_if_any_failed(error, f"update to {step_dir.name}")
        if check is not None:
            self._check(*check)
        self.weight_step = step

    def _plan_update(self, step_dir: Path) -> dict:
        """What every TP worker does to move to ``step_dir``: a dict of plain values, broadcast from TP rank 0."""
        # A stale NFS lookup cache could hide a manifest written before ``.finished``, turning a delta into a full load
        revalidate(step_dir / DELTA_DIR)
        has_delta = (step_dir / DELTA_DIR / MANIFEST).exists()
        base_step = read_manifest(step_dir)["base_step"] if has_delta else None
        plan = {
            "index_map": step_dir.as_posix() if has_delta or anchor_complete(step_dir) else None,
            "write_layout": base_step is None,
            "load": None,
            "deltas": [],
            # Steps that carry an anchor recompute every checksum from the params; others update them with the changes
            "full_verify": any(step_dir.glob("anchor_rank*")),
        }
        if not has_delta:
            plan["load"] = step_dir.as_posix()
        elif base_step is None:
            # The run's first broadcast: the weights the engine booted with
            if self.weight_step is not None:
                plan["load"] = "boot"
        elif base_step == self.weight_step:
            plan["deltas"] = [step_dir.as_posix()]
        else:
            # Load the latest complete anchor at or before ``step_dir`` (or the boot checkpoint, for a chain from the
            # run's first broadcast) and replay the deltas since
            chain, source = [], step_dir
            while not anchor_complete(source):
                source_base = read_manifest(source)["base_step"]
                if source_base is None:
                    break
                chain.append(source)
                source = source.parent / f"step_{source_base}"
            plan["load"] = source.as_posix() if anchor_complete(source) else "boot"
            plan["deltas"] = [delta_dir.as_posix() for delta_dir in reversed(chain)]
            logger.info(f"Replaying {len(chain)} sparse weight deltas on top of {Path(plan['load']).name}")
        return plan

    def _run_plan(self, plan: dict) -> tuple | None:
        """Load and apply what ``plan`` says; returns the check of the final weights for ``_check``."""
        check = None
        if plan["load"] is not None:
            check = self._load_checkpoint(self._boot_dir() if plan["load"] == "boot" else Path(plan["load"]))
        for delta_dir in plan["deltas"]:
            checksums = self._apply_delta(Path(delta_dir))
            # Each delta's checksums cover every tensor, so the last one checks the whole chain
            check = (Path(delta_dir).name, *self.index_map.checksums(full=plan["full_verify"]), checksums)
        return check

    def _raise_if_any_failed(self, error: Exception | None, what: str) -> None:
        """Raise on every worker of the TP group if ``error`` is set on any of them."""
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            failed = torch.tensor([error is not None], dtype=torch.int32)
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX, group=tp_group.cpu_group)
            if failed.item() and error is None:
                raise RuntimeError(f"A TP peer failed the {what}")
        if error is not None:
            raise error

    def _boot_dir(self) -> Path:
        """The local dir of the checkpoint the engine booted from."""
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        folder, *_ = model_loader._prepare_weights(
            self.model_config.model, None, self.model_config.revision, True, None
        )
        return Path(folder)

    def _apply_chunk(
        self,
        entries: list[tuple[str, int]],
        gaps: torch.Tensor,
        values: torch.Tensor,
        tensors: dict,
        device,
        events: list[torch.cuda.Event],
    ) -> None:
        """Apply one chunk of a delta, recording CUDA events after its copies, its index decode and its apply."""
        assert self.index_map is not None
        gaps, values = gaps.to(device, non_blocking=True), values.to(device, non_blocking=True)
        events.append(torch.cuda.Event(enable_timing=True))
        events[-1].record()
        counts = torch.tensor([count for _, count in entries], device=device)
        indices = decode_gaps(gaps, counts)
        events.append(torch.cuda.Event(enable_timing=True))
        events[-1].record()
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
        events.append(torch.cuda.Event(enable_timing=True))
        events[-1].record()

    def _build_index_map(self, step_dir: Path) -> None:
        """At a sparse run's first update, while the engine still holds its boot weights: map checkpoint tensors to
        the params (where the boot weights are, after the engine's processing), and find which ones this worker holds
        in fp8 (and their scale format)."""
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
        boot_dir = self._boot_dir()
        moved = index_map.match_boot(_read_rows(index_map, boot_dir, next(model.parameters()).device))
        if moved:
            logger.info(f"Moved {moved} index map writes to where the engine's kernel layout holds them")
        boot = checkpoint_index(boot_dir)

        def read_tensor(name: str) -> torch.Tensor:
            shape, dtype, _ = boot[name]
            data = np.empty(prod(shape) * dtype.itemsize, np.uint8)
            read_checkpoint_range(boot[name], 0, prod(shape), data)
            return torch.from_numpy(data).view(dtype).view(shape)

        index_map.fp8_layout(model, read_tensor)
        if next(model.parameters()).is_cuda:
            index_map.fused_tables(warm_up=True)
        self.index_map = index_map

    def _write_layout(self, broadcast_dir: Path) -> None:
        """Report which checkpoint tensors this worker holds in fp8, so the trainer sends their deltas as fp8. The
        trainer waits for every worker of the world group (every DP, PP and TP rank)."""
        assert self.index_map is not None
        group = get_world_group()
        layout = {"world_size": group.world_size, "fp8": self.index_map.fp8}
        path = broadcast_dir / ENGINE_LAYOUT / f"rank_{group.rank_in_group}.json"
        path.parent.mkdir(exist_ok=True)
        write_file(path, json.dumps(layout).encode())
        logger.info(f"Wrote the engine layout: {len(layout['fp8'])} checkpoint tensors held in fp8")

    def _apply_delta(self, step_dir: Path) -> dict[str, int]:
        """Patch the params with the sparse delta in ``step_dir``; returns the trainer's checksums of the result."""
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
        events = [torch.cuda.Event(enable_timing=True)]
        events[0].record()
        for entries, gaps, values in delta:
            self._apply_chunk(entries, gaps, values, tensors, device, events)
        host_time = time.perf_counter() - start - read_time
        update_mla_absorbed_weights(model)
        torch.cuda.synchronize()
        # Per chunk: copies, index decode, apply
        phases = [0.0, 0.0, 0.0]
        for index, (begin, end) in enumerate(zip(events, events[1:])):
            phases[index % 3] += begin.elapsed_time(end) / 1000
        logger.info(
            f"Applied sparse weight delta {step_dir.name}: {sum(len(gaps) for _, gaps, _ in delta)} changes in "
            f"{len(delta)} chunks, read {read_time:.2f}s (0 if prefetched), apply {time.perf_counter() - start - read_time:.2f}s "
            f"(GPU: copy {phases[0]:.3f}s, decode {phases[1]:.3f}s, patch {phases[2]:.3f}s; host loop {host_time:.3f}s)"
        )
        return checksums

    def _check(self, label: str, names: list[str], totals: torch.Tensor, held: torch.Tensor, expected: dict) -> None:
        """Check the checksums of the elements this worker holds of each checkpoint tensor (``totals``, over ``held``
        elements) against the trainer's (``expected``), on every worker of the TP group alike.

        Workers sum their parts over the TP group, in one vector over every checkpoint tensor of the run, so each
        runs the same collectives whatever it holds. A tensor is checked when its holders cover it the same number
        of times each (TP slices: once; KV heads replicated over TP workers: once per replica); one held by workers
        that overlap unevenly (fp8 tensors read in 128-row blocks) is checked only where a worker holds it whole."""
        start = time.perf_counter()
        canonical, ids, counts, whole, multiplicity = self._coverage(names, held)
        device = totals.device
        sums = torch.zeros(len(canonical), dtype=torch.int64, device=device).index_put_((ids,), totals)
        has_expected = torch.tensor([name in expected for name in canonical], device=device)
        target = torch.tensor([expected.get(name, 0) for name in canonical], dtype=torch.int64, device=device)
        mismatched = whole & has_expected & (sums != target)
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            group = tp_group.device_group
            torch.distributed.all_reduce(sums, group=group)
            mismatched |= (multiplicity > 0) & has_expected & (sums != multiplicity * target)
            any_mismatch = mismatched.any().to(torch.int32)
            torch.distributed.all_reduce(any_mismatch, op=torch.distributed.ReduceOp.MAX, group=group)
        else:
            any_mismatch = mismatched.any()
        if any_mismatch.item():
            names_off = [canonical[index] for index in mismatched.nonzero().flatten().tolist()[:8]]
            raise RuntimeError(f"Weights at {label} diverged from the trainer: {names_off or 'on a TP peer'}")
        logger.info(f"Verified the weights at {label} in {time.perf_counter() - start:.2f}s")

    def _coverage(self, names: list[str], held: torch.Tensor) -> tuple:
        """What ``_check`` needs of who holds what: every checkpoint tensor of the run (sorted), the positions of
        ``names`` in it, this worker's element counts and whole tensors, and how many times the TP group's holders
        cover each tensor evenly (0: unevenly, or not held). Static for an index map, so kept for the ``names`` and
        ``held`` of its checksum plan; every TP worker follows the same plan, so they compute it (and run its
        collectives) at the same updates."""
        cached = getattr(self, "coverage_cache", None)
        if cached is not None and cached[0] is names and cached[1] is held:
            return cached[2]
        assert self.index_map is not None
        canonical = sorted(self.index_map.numel)
        position = {name: index for index, name in enumerate(canonical)}
        device = held.device
        ids = torch.tensor([position[name] for name in names], dtype=torch.long, device=device)
        counts = torch.zeros(len(canonical), dtype=torch.int64, device=device).index_put_((ids,), held)
        numel = torch.tensor([self.index_map.numel[name] for name in canonical], device=device)
        whole = counts == numel
        multiplicity = torch.zeros_like(counts)
        tp_group = get_tp_group()
        if tp_group.world_size > 1:
            group = tp_group.device_group
            total_counts, most = counts.clone(), counts.clone()
            # Holders' fewest elements; non-holders report the most anyone could hold
            fewest = torch.where(counts > 0, counts, numel)
            torch.distributed.all_reduce(total_counts, group=group)
            torch.distributed.all_reduce(most, op=torch.distributed.ReduceOp.MAX, group=group)
            torch.distributed.all_reduce(fewest, op=torch.distributed.ReduceOp.MIN, group=group)
            even = (total_counts > 0) & (total_counts % numel == 0) & (most == fewest)
            multiplicity = torch.where(even, total_counts // numel, 0)
        coverage = (canonical, ids, counts, whole, multiplicity)
        self.coverage_cache = (names, held, coverage)
        return coverage

    def _load_checkpoint(self, step_dir: Path) -> tuple | None:
        """Reload every weight from the HF checkpoint (or anchor) in ``step_dir``. In a sparse run, each worker reads
        only the rows it holds, straight into its params, and returns their check for ``_check``."""
        model = self._model()
        if self.index_map is None:
            weights = checkpoint_weights(model, step_dir.as_posix(), self.load_config)
            load_weights_checkpoint_layerwise(model, weights, self.model_runner.model_config, self.vllm_config)
            return None
        start = time.perf_counter()
        index_map = self.index_map
        # Checksums of anchors and full checkpoints written by the trainer (none for the boot checkpoint)
        expected = read_anchor_checksums(step_dir)
        num_bytes = 0
        for name, begin, rows in _read_rows(index_map, step_dir, next(model.parameters()).device):
            index_map.load_rows(name, rows, begin)
            num_bytes += rows.nbytes
        update_mla_absorbed_weights(model)
        torch.cuda.synchronize()
        logger.info(
            f"Loaded this worker's {num_bytes / 2**30:.2f} GiB of {step_dir.name} in {time.perf_counter() - start:.1f}s"
        )
        # Checked from the params as loaded, so a wrong mapping or write in ``load_rows`` fails the check
        return step_dir.name, *index_map.held_checksums(), expected


def _read_rows(index_map: CheckpointIndexMap, step_dir: Path, device) -> Iterator[tuple[str, int, torch.Tensor]]:
    """``(name, begin, rows)``: the ``row_ranges`` rows of each checkpoint tensor in ``step_dir``, on ``device``. A
    tied name the checkpoint omits is skipped: its elements come with a tied name it has."""
    index = checkpoint_index(step_dir)
    ranges = []
    for name, rows in index_map.row_ranges().items():
        if name in index:
            ranges.append((name, rows))
        elif not any(tied in index for tied in index_map.tied.get(name, ())):
            raise KeyError(f"{step_dir} has no {name}")

    def read(name: str, begin: int, end: int) -> torch.Tensor:
        num_cols, dtype = index_map.num_cols[name], index[name][1]
        data = np.empty((end - begin) * num_cols * dtype.itemsize, np.uint8)
        read_checkpoint_range(index[name], begin * num_cols, end * num_cols, data)
        return torch.from_numpy(data).view(dtype).view(end - begin, num_cols)

    with ThreadPoolExecutor(16) as pool:
        # A bounded window of reads ahead of the GPU writes
        for window in itertools.batched(ranges, 64):
            for (name, (begin, _)), rows in zip(window, pool.map(lambda item: read(item[0], *item[1]), window)):
                yield name, begin, rows.to(device)
