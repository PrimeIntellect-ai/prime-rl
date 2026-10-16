import json
import threading
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.shared import FileSystemWeightBroadcastConfig
from prime_rl.transports.weights.base import WeightSender
from prime_rl.transports.weights.filesystem import FileSystemWeightSender
from prime_rl.utils.logger import get_logger
from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    FP8_BLOCK,
    MANIFEST,
    anchor_complete,
    as_bits,
    bits_dtype,
    compile_fp8_quantizer,
    encode_gaps,
    page_lock,
    quantize_fp8_blocks,
    rank_part,
    rank_parts_marker,
    read_engine_layout,
    revalidate,
    write_anchor_shards,
    write_file,
    write_rank_file,
    zigzag,
)
from prime_rl.utils.weights import convert_state_dict_to_hf, resolve_fqn, resolve_wire_dtype


class SparseDeltaEncoder:
    """Rank-local deltas of the trainer's weight shards, in HF coordinates.

    Every rank keeps the wire-dtype bits of its own shards from the previous
    broadcast (the snapshot) and diffs against them, so no weights are gathered.
    The prime->HF conversion is replayed once on meta tensors: every HF tensor
    must come out as a contiguous view of one trainer tensor (renames, expert
    unstacking), which maps a changed trainer element to its HF name and index.

    ``encode`` runs on a background thread and a side CUDA stream while the
    next step trains, without collectives; it only reads the weights before
    ``weights_read`` is set, and hands the delta to the writer in parts as it
    goes. The snapshot lives on the GPU when it fits next to the first training
    step's peak memory, else in one page-locked host buffer (locked only between
    training steps: registering it can stall the trainer's CUDA calls). With a
    GPU snapshot, the same host buffer (pageable) holds anchor copies instead, so
    a rank has at most one shard-sized host copy.

    Tensors the engines hold in fp8 (``set_fp8_layout``: HF name -> scale format,
    from the engine layout) are quantized with vLLM's own per-block quantizer,
    previous and current weights alike, and diffed as fp8 bytes. Their 128-row
    blocks must not straddle trainer ranks, else the trainer tensor is gathered.
    The quantizer is compiled ahead of time, so it can run on the encode thread
    (columns a multiple of 128). A step that changes any block scale is
    published as a full checkpoint.
    """

    # Elements per transient chunk of the diff (bf16: 128 MiB per temporary)
    CHUNK = 1 << 26
    # Changes per part handed to the writer
    PART = 1 << 26

    def __init__(self, model: nn.Module, snapshot: str):
        keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
        self.wire_dtypes: dict[str, torch.dtype] = {}
        meta: dict[str, torch.Tensor] = {}
        for key, value in model.state_dict().items():
            dtype = resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16) if isinstance(value, DTensor) else value.dtype
            self.wire_dtypes[key] = dtype
            meta[resolve_fqn(model, key)] = torch.empty(value.shape, dtype=dtype, device="meta")
        owner = {id(tensor): fqn for fqn, tensor in meta.items()}

        segments = defaultdict(list)
        self.hf_tensors: dict[str, tuple[list[int], str]] = {}
        for name, tensor in convert_state_dict_to_hf(model, dict(meta)).items():
            base = tensor if tensor._base is None else tensor._base
            if id(base) not in owner or not tensor.is_contiguous():
                raise NotImplementedError(f"Sparse weight broadcast cannot map {name} to a contiguous trainer slice")
            if tensor.numel() >= 2**32:
                raise NotImplementedError(f"Sparse weight broadcast indexes tensors below 2^32 elements ({name})")
            segments[owner[id(base)]].append((tensor.storage_offset(), name))
            self.hf_tensors[name] = (list(tensor.shape), str(tensor.dtype).removeprefix("torch."))
        self.segments = {fqn: sorted(segs) for fqn, segs in segments.items()}
        self.numel = {fqn: tensor.numel() for fqn, tensor in meta.items()}
        # fqn -> (ue8m0 scales, columns) for trainer tensors held in fp8; quantized as one [rows, columns] matrix
        self.fp8: dict[str, tuple[bool, int]] = {}
        self.fp8_layout: dict[str, str] | None = None
        self.compiled_fp8 = None
        self.scales_changed = False
        self.gather: set[str] = set()
        self.segment_starts: dict[str, torch.Tensor] = {}
        self.shard_segments: dict[str, tuple[list[str], list[int], list[int], list[int]]] = {}
        self.snapshot_placement = snapshot
        # key -> (local numel, global flat offset) of this rank's shard
        self.shards: dict[str, tuple[int, int]] = {}
        # key -> previous wire bits: GPU tensors, or views into the host buffer
        self.snapshot: dict[str, torch.Tensor] = {}
        self.previous: dict[str, torch.Tensor] = {}
        self.on_gpu: bool | None = None  # None until placed (auto: at the first delta)
        self.host_buffer: torch.Tensor | None = None
        self.host: dict[str, torch.Tensor] = {}
        self.pinned = False
        self.initialized = False
        self.device = torch.cuda.current_device()
        self.stream = torch.cuda.Stream()

    def set_fp8_layout(self, model: nn.Module, fp8: dict[str, str], is_master: bool) -> None:
        """Diff the tensors the engines hold in fp8 (HF name -> scale format) as fp8 bytes from now on."""
        self.fp8_layout = fp8
        for fqn, segs in self.segments.items():
            modes = {fp8.get(name) for _, name in segs}
            if modes == {None}:
                continue
            shapes = [self.hf_tensors[name][0] for _, name in segs]
            num_cols = shapes[0][-1]
            num_rows = self.numel[fqn] // num_cols
            boundaries = [start // num_cols for start, _ in segs] + [
                start // num_cols + shape[0] for (start, _), shape in zip(segs, shapes)
            ]
            if (
                len(modes) > 1
                or num_cols % FP8_BLOCK[1]
                or any(len(shape) != 2 or shape[1] != num_cols for shape in shapes)
                or any(row % FP8_BLOCK[0] and row != num_rows for row in boundaries)
            ):
                raise NotImplementedError(f"Sparse weight broadcast cannot quantize {fqn} to fp8 as one matrix")
            self.fp8[fqn] = (modes.pop() == "ue8m0", num_cols)
            for _, name in segs:
                self.hf_tensors[name] = (self.hf_tensors[name][0], "float8_e4m3fn")
        self.gather = self._find_gathered(model, is_master)
        if self.fp8:
            start = time.perf_counter()
            self.compiled_fp8 = compile_fp8_quantizer()
            get_logger().info(f"Compiled the fp8 quantizer ahead of time in {time.perf_counter() - start:.1f}s")

    def _local_shard(self, key: str, value: torch.Tensor, is_master: bool) -> tuple[torch.Tensor, int] | None:
        """This rank's dim-0 slice of ``value`` and its global flat offset, or None when another rank owns the
        elements. Flattened only where it is read (``reshape(-1)``), as that copies a non-contiguous slice."""
        if not isinstance(value, DTensor):
            return (value.detach(), 0) if is_master else None
        if all(placement.is_replicate() for placement in value.placements):
            return (value.to_local().detach(), 0) if is_master else None
        local_shape, global_offset = compute_local_shape_and_global_offset(
            value.shape, value.device_mesh, value.placements
        )
        if tuple(local_shape[1:]) != tuple(value.shape[1:]) or any(global_offset[1:]):
            raise NotImplementedError(f"Sparse weight broadcast needs dim-0 shards ({key}: {value.placements})")
        return value.to_local().detach()[: local_shape[0]], global_offset[0] * prod(value.shape[1:])

    def _find_gathered(self, model: nn.Module, is_master: bool) -> set[str]:
        """fp8 trainer tensors whose 128-row blocks straddle ranks on some rank."""
        keys, straddles = [], []
        for key, value in model.state_dict().items():
            fqn = resolve_fqn(model, key)
            if fqn not in self.fp8:
                continue
            if self.wire_dtypes[key] != torch.bfloat16:
                raise NotImplementedError(f"Sparse weight broadcast quantizes bf16 weights to fp8, not {key}")
            shard = self._local_shard(key, value, is_master)
            num_cols = self.fp8[fqn][1]
            num_rows = value.numel() // num_cols
            first = 0 if shard is None else shard[1] // num_cols
            end = first if shard is None else first + shard[0].numel() // num_cols
            keys.append(key)
            straddles.append(first % FP8_BLOCK[0] != 0 or (end % FP8_BLOCK[0] != 0 and end != num_rows))
        flags = torch.tensor(straddles, dtype=torch.int32, device=next(model.parameters()).device)
        if dist.is_initialized():
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        return {key for key, flag in zip(keys, flags.tolist()) if flag}

    @torch.no_grad()
    def local_shards(self, model: nn.Module, is_master: bool) -> list[tuple]:
        """This rank's ``(key, fqn, shard, global flat offset, gathered)`` for ``encode``, taken on the
        training thread: FSDP swaps module parameters during forward, but the sharded storage stays put.
        ``gathered`` is ``(previous, current)`` bf16 full tensors of fp8 tensors whose blocks straddle ranks
        (the all-gathers run here, as ``encode`` runs no collectives)."""
        shards = []
        for key, value in model.state_dict().items():
            fqn = resolve_fqn(model, key)
            if fqn not in self.segments:
                continue
            shard = self._local_shard(key, value, is_master)
            empty = shard is None or shard[0].numel() == 0
            gathered = None
            if key in self.gather and self.initialized:
                if empty:
                    value.full_tensor()  # every rank joins the all-gathers
                    value.full_tensor()
                else:
                    gathered = (self._gather_previous(key, value), value.full_tensor().to(torch.bfloat16))
            if not empty:
                shards.append((key, fqn, shard[0], shard[1], gathered))
        return shards

    def _gather_previous(self, key: str, value: DTensor) -> torch.Tensor:
        previous = self.snapshot[key].to(f"cuda:{self.device}").view(torch.bfloat16)
        local_shape = (previous.numel() // prod(value.shape[1:]), *value.shape[1:])
        return DTensor.from_local(
            previous.view(local_shape), value.device_mesh, value.placements, shape=value.shape, stride=value.stride()
        ).full_tensor()

    def _bits(self, key: str) -> torch.dtype:
        return bits_dtype(str(self.wire_dtypes[key]).removeprefix("torch."))

    def _allocate_host(self) -> None:
        """One host buffer for all shards (the CPU snapshot, or anchor copies of a GPU snapshot)."""
        sizes = {key: -(-numel * self.wire_dtypes[key].itemsize // 8) * 8 for key, (numel, _) in self.shards.items()}
        self.host_buffer = torch.from_numpy(np.empty(sum(sizes.values()), np.uint8))
        offset = 0
        for key, (numel, _) in self.shards.items():
            self.host[key] = self.host_buffer[offset : offset + numel * self.wire_dtypes[key].itemsize].view(
                self._bits(key)
            )
            offset += sizes[key]

    def _page_lock(self) -> None:
        """Page-lock the host buffer, only while no training step runs: registering a large buffer can
        stall the trainer's CUDA calls."""
        start = time.perf_counter()
        page_lock(self.host_buffer)
        self.pinned = True
        get_logger().info(
            f"Page-locked the {self.host_buffer.numel() / 2**30:.1f} GiB sparse weight host buffer in "
            f"{time.perf_counter() - start:.1f}s"
        )

    def _place(self, peak_reserved: int) -> None:
        """Keep the snapshot on the GPU if it fits next to the first training step's peak, else on the host."""
        num_bytes = sum(numel * self.wire_dtypes[key].itemsize for key, (numel, _) in self.shards.items())
        total = torch.cuda.get_device_properties(self.device).total_memory
        headroom = total - peak_reserved - num_bytes
        self.on_gpu = self.snapshot_placement == "auto" and headroom >= max(4 * 2**30, total // 20)
        get_logger().info(
            f"Sparse weight snapshot ({num_bytes / 2**30:.1f} GiB) on the {'GPU' if self.on_gpu else 'host'} "
            f"(training peak {peak_reserved / 2**30:.1f} GiB of {total / 2**30:.1f} GiB, "
            f"snapshot={self.snapshot_placement})"
        )

    def _fill(self, shards: list[tuple]) -> None:
        """The first broadcast only records the weights, in the host buffer (placed at the first delta,
        against the measured training peak). The host buffer is filled in every mode, so its pages are
        resident before training: faulting them in later stalls the trainer."""
        self.shards = {key: (shard.numel(), offset) for key, _, shard, offset, _ in shards}
        self.staging = torch.empty(1 << 28, dtype=torch.uint8, pin_memory=True)
        self._allocate_host()
        if self.snapshot_placement == "cpu":
            self._page_lock()
        for key, _, shard, _, _ in shards:
            shard = shard.reshape(-1)
            for start in range(0, shard.numel(), self.CHUNK):
                end = start + self.CHUNK
                self.host[key][start:end].copy_(as_bits(shard[start:end].to(self.wire_dtypes[key])), non_blocking=True)
        self.snapshot = dict(self.host)

    def copy_snapshot_to_host(self) -> None:
        """Copy a GPU snapshot (the last broadcast's weights) into the host buffer, for an anchor, through the
        pinned staging buffer. Callers wait for earlier readers of the host buffer first."""
        assert self.on_gpu
        with torch.cuda.stream(self.stream):
            for key, bits in self.snapshot.items():
                self._to_host(bits, self.host[key])

    def anchor_parts(self) -> list[tuple[str, list[int], int, torch.Tensor]]:
        """This rank's slices of the full HF checkpoint of the last broadcast, from host memory (the CPU
        snapshot, or ``copy_snapshot_to_host``)."""
        parts = []
        for key, bits in self.host.items():
            values, (numel, offset) = bits.view(self.wire_dtypes[key]), self.shards[key]
            for start, name in self.segments[self.key_fqn[key]]:
                begin, end = max(start, offset), min(start + prod(self.hf_tensors[name][0]), offset + numel)
                if begin < end:
                    parts.append((name, self.hf_tensors[name][0], begin - start, values[begin - offset : end - offset]))
        return parts

    @torch.no_grad()
    def prepare(self, peak_reserved: int) -> None:
        """The training thread's part of an encode, before ``encode``: place the snapshot at the first delta."""
        self.previous = self.snapshot
        if not self.initialized or self.on_gpu is not None:
            return
        self._place(peak_reserved)
        if self.on_gpu:
            device = f"cuda:{self.device}"
            self.snapshot = {key: torch.empty_like(bits, device=device) for key, bits in self.host.items()}
        elif not self.pinned:
            self._page_lock()

    @torch.no_grad()
    def encode(
        self, shards: list[tuple], ready: torch.cuda.Event, weights_read: threading.Event, emit: Callable
    ) -> None:
        """Diff the shards against the snapshot and advance it, on the side stream once ``ready`` (recorded
        after the optimizer step) has passed; the first call only records the snapshot. Hands the delta to
        ``emit`` in parts ``(entries, gaps, value bytes, new-weight checksum per HF tensor part)`` on the
        host as they are encoded. Sets ``weights_read`` once done reading the weights. ``scales_changed``:
        whether an fp8 block scale changed on this rank."""
        torch.cuda.set_device(self.device)
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                self.key_fqn = {key: fqn for key, fqn, *_ in shards}
                if not self.initialized:
                    self._fill(shards)
                    self.stream.synchronize()
                    self.initialized = True
                    return
                self.scales_changed = self._diff(shards, emit)
                self.stream.synchronize()
        finally:
            weights_read.set()

    def _to_host(self, tensor: torch.Tensor, host: torch.Tensor | None = None) -> torch.Tensor:
        """Copy a GPU tensor to pageable host memory (``host``, or a new tensor) through the pinned staging
        buffer: a copy straight into pageable memory holds the CUDA driver and stalls the training step's
        own CUDA calls."""
        host = torch.empty(tensor.shape, dtype=tensor.dtype) if host is None else host
        data, host_bytes = tensor.view(torch.uint8), host.view(torch.uint8)
        for begin in range(0, data.numel(), self.staging.numel()):
            chunk = self.staging[: min(self.staging.numel(), data.numel() - begin)]
            chunk.copy_(data[begin : begin + chunk.numel()], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            host_bytes[begin : begin + chunk.numel()].copy_(chunk)
        return host

    def _segments(
        self, key: str, fqn: str, offset: int, numel: int
    ) -> tuple[list[str], list[int], list[int], list[int]]:
        """The HF tensor parts in this rank's shard of ``key``: names, begin/end in the shard, and the shift
        from a shard position to the HF flat index."""
        if key not in self.shard_segments:
            names, begins, ends, shifts = [], [], [], []
            for start, name in self.segments[fqn]:
                hf_numel = prod(self.hf_tensors[name][0])
                if start < offset + numel and start + hf_numel > offset:
                    names.append(name)
                    begins.append(max(start - offset, 0))
                    ends.append(min(start - offset + hf_numel, numel))
                    shifts.append(offset - start)
            self.shard_segments[key] = (names, begins, ends, shifts)
        return self.shard_segments[key]

    @staticmethod
    def _segment_sums(bits: torch.Tensor, begins: list[int], ends: list[int], firsts: list[int]) -> torch.Tensor:
        """``bits_checksum`` of each ``bits[begin:end]`` whose first element is at HF flat index ``first``:
        Σ bits[k]·(first + k - begin + 1), wrapping int64. One reduction per run of adjacent equal-length
        segments (e.g. unstacked experts): few kernels, as the encode thread shares the GIL with training."""
        sums, i = [], 0
        while i < len(begins):
            length, j = ends[i] - begins[i], i + 1
            while j < len(begins) and begins[j] == ends[j - 1] and ends[j] - begins[j] == length:
                j += 1
            rows = bits[begins[i] : begins[i] + (j - i) * length].view(j - i, length)
            positions = torch.arange(1, length + 1, device=bits.device)
            sums.append(
                (rows * positions).sum(1)
                + rows.sum(1, dtype=torch.int64) * torch.tensor(firsts[i:j], device=bits.device)
            )
            i = j
        return torch.cat(sums)

    def _diff(self, shards: list[tuple], emit: Callable) -> bool:
        """Diff ``shards`` against ``self.previous`` on the current stream, writing the new bits to the
        snapshot, in chunks of ``CHUNK`` elements (fp8: whole 128-row blocks). Copies the delta to the
        host and hands it to ``emit`` every ``PART`` changes. Returns whether an fp8 block scale changed."""
        device = torch.device(f"cuda:{self.device}")
        pending: list[tuple] = []
        num_pending = 0
        scales_changed = False

        def flush() -> None:
            names = [name for part in pending for name in part[0]]
            counts = torch.cat([part[1] for part in pending]).tolist()
            gaps = self._to_host(torch.cat([part[2] for part in pending]))
            values = self._to_host(torch.cat([part[3] for part in pending]))
            checksum_names = [name for part in pending for name in part[4]]
            checksums = dict(zip(checksum_names, torch.cat([part[5] for part in pending]).tolist()))
            emit(([(name, count) for name, count in zip(names, counts) if count], gaps, values, checksums))
            pending.clear()

        for key, fqn, shard, offset, gathered in shards:
            shard = shard.reshape(-1)
            previous, out, wire = self.previous[key], self.snapshot[key], self.wire_dtypes[key]
            segment_names, begins, ends, shifts = self._segments(key, fqn, offset, shard.numel())
            sums = torch.zeros(len(segment_names), dtype=torch.int64, device=device)
            changed, changed_values = [], []
            if fqn in self.fp8:
                ue8m0, num_cols = self.fp8[fqn]
                if gathered is not None:
                    chunks = [(0, shard.numel())]
                else:
                    rows = max(FP8_BLOCK[0], self.CHUNK // num_cols // FP8_BLOCK[0] * FP8_BLOCK[0])
                    chunks = [(start, start + rows * num_cols) for start in range(0, shard.numel(), rows * num_cols)]
            else:
                chunks = [(start, start + self.CHUNK) for start in range(0, shard.numel(), self.CHUNK)]
            for start, end in chunks:
                end = min(end, shard.numel())
                current = as_bits(shard[start:end].to(device, non_blocking=True).to(wire))
                previous_bits = previous[start:end].to(device, non_blocking=True)
                if fqn in self.fp8:
                    if gathered is not None:
                        previous_weights, current_weights = gathered
                    else:
                        previous_weights, current_weights = (
                            previous_bits.view(torch.bfloat16),
                            current.view(torch.bfloat16),
                        )
                    (previous_fp8, previous_scales), (current_fp8, current_scales) = (
                        quantize_fp8_blocks(weights.reshape(-1, num_cols), ue8m0, self.compiled_fp8)
                        for weights in (previous_weights, current_weights)
                    )
                    scales_changed |= not torch.equal(previous_scales, current_scales)
                    if gathered is not None:
                        previous_fp8 = previous_fp8.view(-1)[offset : offset + shard.numel()]
                        current_fp8 = current_fp8.view(-1)[offset : offset + shard.numel()]
                    delta = current_fp8.view(-1) - previous_fp8.view(-1)
                    new_bits = current_fp8.view(-1)
                else:
                    delta = current - previous_bits
                    new_bits = current
                out[start:end].copy_(current, non_blocking=True)
                index = delta.nonzero().squeeze(1)
                changed.append(index + start)
                changed_values.append(zigzag(delta[index]))
                first, last = bisect_right(ends, start), bisect_left(begins, end)
                if first < last:
                    sums[first:last] += self._segment_sums(
                        new_bits,
                        [max(begin, start) - start for begin in begins[first:last]],
                        [min(stop, end) - start for stop in ends[first:last]],
                        [max(begin, start) + shift for begin, shift in zip(begins[first:last], shifts[first:last])],
                    )
            changed, changed_values = torch.cat(changed), torch.cat(changed_values)
            if fqn not in self.segment_starts:
                self.segment_starts[fqn] = torch.tensor([start for start, _ in self.segments[fqn]], device=device)
            starts = self.segment_starts[fqn]
            position = changed + offset
            segment = torch.searchsorted(starts, position, right=True) - 1
            entry_starts = torch.ones_like(segment, dtype=torch.bool)
            entry_starts[1:] = segment[1:] != segment[:-1]
            gaps = encode_gaps(position - starts[segment], entry_starts)
            counts = torch.bincount(segment, minlength=len(starts))
            pending.append(
                (
                    [name for _, name in self.segments[fqn]],
                    counts,
                    gaps,
                    changed_values.view(torch.uint8),
                    segment_names,
                    sums,
                )
            )
            num_pending += len(gaps)
            if num_pending >= self.PART:
                flush()
                num_pending = 0
        if pending:
            flush()
        return bool(scales_changed)


def _write_json(path: Path, value) -> None:
    write_file(path, json.dumps(value).encode())


class SparseFileSystemWeightSender(FileSystemWeightSender):
    """Filesystem broadcast where steps only carry the delta against the
    previous broadcast (see ``prime_rl.utils.sparse_weights``). The delta is
    encoded and written in the background while the next step trains: the
    trainer only waits, before its next step's final backward (which may run
    the optimizer step), until the encode has read the weights. Full checkpoints (anchors) serve engines that fall off
    the delta chain. Steps are committed by the writer thread once every rank's
    files are on disk."""

    gathers_full_weights = False

    def __init__(self, output_dir: Path, config: FileSystemWeightBroadcastConfig):
        super().__init__(output_dir, config)
        self.config = config
        self.encoder: SparseDeltaEncoder | None = None
        self.last_step: int | None = None
        self.anchor_step: int | None = None
        # Background work, each in order: encodes, delta writes and commits, periodic anchor writes
        self.encodes = ThreadPoolExecutor(1)
        self.writes = ThreadPoolExecutor(1)
        self.anchors = ThreadPoolExecutor(1)
        self.host_readers: list[Future] = []
        self.errors: list[BaseException] = []
        self.weights_read: threading.Event | None = None
        self.part_bytes = 0

    def wait_for_weight_reads(self) -> None:
        if self.weights_read is None:
            return
        start = time.perf_counter()
        self.weights_read.wait()
        self.weights_read = None
        self.logger.debug(f"Waited {time.perf_counter() - start:.2f}s for the sparse weight encode")
        self._raise_errors()

    def close(self) -> None:
        for pool in (self.encodes, self.writes, self.anchors):
            pool.shutdown(wait=True)
        self._raise_errors()

    def _raise_errors(self) -> None:
        if self.errors:
            raise RuntimeError("Sparse weight broadcast failed in the background") from self.errors[0]

    def _submit(self, pool: ThreadPoolExecutor, fn, *args) -> Future:
        def run() -> None:
            try:
                fn(*args)
            except BaseException as error:
                self.logger.exception(f"Sparse weight broadcast failed in {fn.__name__}")
                self.errors.append(error)
                raise

        return pool.submit(run)

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        if self.world.is_master:
            # Engines that load a step with a delta dir report their weight layout next to it.
            (step_dir / DELTA_DIR).mkdir(exist_ok=True)
        assert self.encoder is not None
        first = self.last_step is None
        base_step, self.last_step = self.last_step, step
        if first:
            # The snapshot is recorded (``_prepare``) before the first training step.
            self.wait_for_weight_reads()
            if step == 0:
                # The run's first broadcast is the checkpoint the engines booted from.
                self.anchor_step = step
                if self.world.is_master:
                    manifest = {"step": step, "base_step": None, "ranks": 0, "tensors": self.encoder.hf_tensors}
                    _write_json(step_dir / DELTA_DIR / MANIFEST, manifest)
                    WeightSender._finish(self, step, step_dir)
                return
            # A resumed run's first broadcast is a full checkpoint.
            self._submit(self.encodes, self._publish_full, step, step_dir)
            return
        if self.encoder.fp8_layout is None:
            # Every engine worker reports which tensors it holds in fp8 when it loads the first broadcast.
            layout = [self._wait_for_engine_layout(step_dir.parent) if self.world.is_master else None]
            dist.broadcast_object_list(layout, src=0)
            self.encoder.set_fp8_layout(model, layout[0], self.world.is_master)
        shards = self.encoder.local_shards(model, self.world.is_master)
        self.encoder.prepare(torch.cuda.max_memory_reserved())
        # Periodic anchors only serve engines that fall off the chain: written in the background, each after
        # the previous one.
        anchor = step % self.config.anchor_interval == 0
        ready = torch.cuda.Event()
        ready.record()
        self.weights_read = threading.Event()
        args = (step, base_step, step_dir, shards, ready, self.weights_read, anchor)
        self._submit(self.encodes, self._encode, *args)

    def _wait_for_engine_layout(self, broadcast_dir: Path) -> dict[str, str]:
        start = time.monotonic()
        while (layout := read_engine_layout(broadcast_dir)) is None:
            if time.monotonic() - start > self.timeout:
                raise TimeoutError(f"Not every engine worker reported its weight layout within {self.timeout}s")
            time.sleep(0.1)
        self.logger.debug(f"Waited {time.monotonic() - start:.2f}s for the engine weight layout")
        return layout

    def _prepare(self, model: nn.Module, step: int) -> None:
        self.wait_for_weight_reads()
        self._raise_errors()
        if self.encoder is None:
            # The first broadcast only records the snapshot, while the master waits for the receivers.
            self.encoder = SparseDeltaEncoder(model, self.config.snapshot)
            shards = self.encoder.local_shards(model, self.world.is_master)
            ready = torch.cuda.Event()
            ready.record()
            self.weights_read = threading.Event()
            self._submit(self.encodes, self._record, shards, ready, self.weights_read)

    def _record(self, shards, ready, weights_read: threading.Event) -> None:
        encoder = self.encoder
        assert encoder is not None
        start = time.perf_counter()
        encoder.encode(shards, ready, weights_read, emit=None)
        num_bytes = sum(numel * encoder.wire_dtypes[key].itemsize for key, (numel, _) in encoder.shards.items())
        self.logger.info(
            f"Recorded the {num_bytes / 2**30:.1f} GiB sparse weight snapshot in {time.perf_counter() - start:.1f}s"
        )

    def _wait_for_host_readers(self) -> None:
        for reader in self.host_readers:
            reader.exception()
        self.host_readers = []

    def _anchor_parts(self) -> list:
        """This rank's slices of the last broadcast's full checkpoint, in host memory."""
        assert self.encoder is not None
        if self.encoder.on_gpu:
            self._wait_for_host_readers()
            self.encoder.copy_snapshot_to_host()
        return self.encoder.anchor_parts()

    def _publish_full(self, step: int, step_dir: Path) -> None:
        parts = self._anchor_parts()  # before reading self.host_readers, which this resets
        self.host_readers.append(self._submit(self.writes, self._write_full, step, step_dir, parts))

    def _encode(
        self,
        step: int,
        base_step: int,
        step_dir: Path,
        shards,
        ready,
        weights_read: threading.Event,
        anchor: bool,
    ) -> None:
        """Encode thread: diff v{step}, handing the delta to the writer in parts as it is encoded. With fp8
        deltas, the parts are held until the ranks agree that no block scale changed, else the step is a
        full checkpoint."""
        encoder = self.encoder
        assert encoder is not None
        if not encoder.on_gpu:
            self._wait_for_host_readers()  # anchors written from the host snapshot, which the encode advances
        start = time.perf_counter()
        held: list[tuple] = []
        num_parts = 0

        def emit(part: tuple) -> None:
            nonlocal num_parts
            if num_parts == 0 and self.world.is_master:
                self._submit(self.writes, self._start_delta, step, base_step, step_dir)
            self._submit(self.writes, self._write_part, step_dir, num_parts, part)
            num_parts += 1

        encoder.encode(shards, ready, weights_read, held.append if encoder.fp8 else emit)
        # Free the gathered fp8 tensors now, not after the anchor copy below, which may overlap the next step's
        # memory peak.
        shards.clear()
        if encoder.fp8:
            if self._scales_changed_anywhere(step_dir, encoder.scales_changed):
                self.logger.info(f"fp8 block scales changed at v{step}: publishing a full checkpoint")
                self._publish_full(step, step_dir)
                return
            for part in held:
                emit(part)
        if num_parts == 0 and self.world.is_master:
            self._submit(self.writes, self._start_delta, step, base_step, step_dir)
        self._submit(self.writes, self._finish_delta, step, base_step, step_dir, num_parts, time.perf_counter() - start)
        if anchor:
            parts = self._anchor_parts()  # before reading self.host_readers, which this resets
            self.host_readers.append(self._submit(self.anchors, self._write_anchor, step, step_dir, parts))

    def _scales_changed_anywhere(self, step_dir: Path, changed: bool) -> bool:
        """Agree through the step dir whether any rank's fp8 block scales changed: the encode thread runs no
        collectives, and the training thread may be waiting for a batch that needs this step."""
        delta_dir = step_dir / DELTA_DIR
        delta_dir.mkdir(exist_ok=True)
        _write_json(delta_dir / f"rank_{self.world.rank}.scales", int(changed))
        flags = [delta_dir / f"rank_{rank}.scales" for rank in range(self.world.world_size)]
        while not all(flag.exists() for flag in flags):
            time.sleep(0.05)
            revalidate(delta_dir)
        return any(flag.read_text() == "1" for flag in flags)

    def _start_delta(self, step: int, base_step: int, step_dir: Path) -> None:
        """The manifest comes first, so engines can read part files as they appear."""
        assert self.encoder is not None
        manifest = {
            "step": step,
            "base_step": base_step,
            "ranks": self.world.world_size,
            "tensors": self.encoder.hf_tensors,
        }
        _write_json(step_dir / DELTA_DIR / MANIFEST, manifest)

    def _write_part(self, step_dir: Path, index: int, part: tuple) -> None:
        assert self.encoder is not None
        entries, gaps, values, checksums = part
        path = step_dir / DELTA_DIR / rank_part(self.world.rank, index)
        self.part_bytes += write_rank_file(
            path, entries, gaps.numpy(), values.numpy(), self.encoder.hf_tensors, checksums
        )

    def _finish_delta(self, step: int, base_step: int, step_dir: Path, num_parts: int, encode_time: float) -> None:
        """After this rank's last part: its parts marker. The master then waits for every rank's marker and
        commits the step. Runs on the writer thread, without collectives."""
        delta_dir = step_dir / DELTA_DIR
        _write_json(delta_dir / rank_parts_marker(self.world.rank), {"parts": num_parts, "bytes": self.part_bytes})
        self.part_bytes = 0
        if not self.world.is_master:
            return
        markers = [delta_dir / rank_parts_marker(rank) for rank in range(self.world.world_size)]
        while not all(marker.exists() for marker in markers):
            time.sleep(0.05)
            revalidate(delta_dir)
        WeightSender._finish(self, step, step_dir)
        num_bytes = sum(json.loads(marker.read_text())["bytes"] for marker in markers)
        self.logger.info(
            f"Sparse weight delta v{step} (base v{base_step}): {num_bytes / 2**20:.1f} MiB, "
            f"encode {encode_time:.2f}s (parts written as encoded)"
        )

    def _write_full(self, step: int, step_dir: Path, parts: list) -> None:
        """Publish v{step} as a full checkpoint: every rank writes its slices, the master commits."""
        self._write_anchor(step, step_dir, parts)
        if self.world.is_master:
            WeightSender._finish(self, step, step_dir)

    def _write_anchor(self, step: int, step_dir: Path, parts: list) -> None:
        start = time.perf_counter()
        write_anchor_shards(parts, step_dir, self.world.rank, self.world.world_size)
        self.logger.debug(f"Wrote the {step_dir.name} anchor in {time.perf_counter() - start:.1f}s")
        if self.world.is_master:
            while not anchor_complete(step_dir):
                time.sleep(0.1)
            self.anchor_step = max(self.anchor_step or 0, step)

    def _finish(self, step: int, step_dir: Path) -> None:
        """Committed in ``_broadcast`` at v0, else in the background (``_finish_delta``, ``_write_full``)."""

    def _clean(self, step: int) -> None:
        # Sparse chains replay from their anchor, so keep everything since it.
        super()._clean(step if self.anchor_step is None else min(step, self.anchor_step + 1))
