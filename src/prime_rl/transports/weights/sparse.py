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
from huggingface_hub import snapshot_download
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
    checkpoint_index,
    compile_fp8_quantizer,
    encode_gaps,
    page_lock,
    quantize_fp8_blocks,
    rank_part,
    rank_parts_marker,
    read_checkpoint_range,
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
    goes. The snapshot is placed at the first broadcast after a training step: on
    the GPU when it fits next to that step's peak memory, else in one page-locked
    host buffer (locked only between training steps: registering it can stall the
    trainer's CUDA calls). It starts as the checkpoint the engines booted from,
    read back from storage. With a GPU snapshot, a pageable host buffer holds
    anchor copies, allocated at the first periodic anchor.

    Tensors the engines hold in fp8 (``set_fp8_layout``: HF name -> scale format,
    from the engine layout) are quantized with vLLM's own per-block quantizer,
    previous and current weights alike, and diffed as fp8 bytes. Their 128-row
    blocks must not straddle trainer ranks, else the trainer tensor is gathered.
    The quantizer is compiled ahead of time, so it can run on the encode thread
    (columns a multiple of 128). Their block scales are diffed too, as tensors
    ``"{hf_name}_scale_inv"`` (fp32, one element per 128x128 block) in the same
    delta, so engines patch scales in place.
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
            if tensor.numel() >= 2**31:
                raise NotImplementedError(f"Sparse weight broadcast indexes tensors below 2^31 elements ({name})")
            segments[owner[id(base)]].append((tensor.storage_offset(), name))
            self.hf_tensors[name] = (list(tensor.shape), str(tensor.dtype).removeprefix("torch."))
        self.segments = {fqn: sorted(segs) for fqn, segs in segments.items()}
        self.numel = {fqn: tensor.numel() for fqn, tensor in meta.items()}
        # fqn -> (ue8m0 scales, columns) for trainer tensors held in fp8; quantized as one [rows, columns] matrix
        self.fp8: dict[str, tuple[bool, int]] = {}
        self.fp8_layout: dict[str, str] | None = None
        self.compiled_fp8 = None
        # fqn -> segments of its fp8 block scales, like ``segments``
        self.scale_segments: dict[str, list[tuple[int, str]]] = {}
        self.scale_changes = 0
        self.gather: set[str] = set()
        self.segment_starts: dict[str, torch.Tensor] = {}
        self.shard_segments: dict[str, tuple[list[str], list[int], list[int], list[int]]] = {}
        self.snapshot_placement = snapshot
        # key -> (local numel, global flat offset) of this rank's shard
        self.shards: dict[str, tuple[int, int]] = {}
        # key -> previous wire bits: GPU tensors, or views into the host buffer
        self.snapshot: dict[str, torch.Tensor] = {}
        self.on_gpu: bool | None = None  # None until placed, at the first broadcast after a training step
        self.host_buffer: torch.Tensor | None = None
        self.write_staging: torch.Tensor | None = None
        # The boot checkpoint's ``checkpoint_index``, and keys whose v0 is still to read from it
        self.boot_files: dict[str, tuple] = {}
        self.boot_pending: set[str] = set()
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
            scale_cols = num_cols // FP8_BLOCK[1]
            self.scale_segments[fqn] = []
            for (start, name), shape in zip(segs, shapes):
                self.hf_tensors[name] = (shape, "float8_e4m3fn")
                self.hf_tensors[f"{name}_scale_inv"] = ([-(-shape[0] // FP8_BLOCK[0]), scale_cols], "float32")
                self.scale_segments[fqn].append((start // num_cols // FP8_BLOCK[0] * scale_cols, f"{name}_scale_inv"))
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
        """Keep the snapshot on the GPU if it fits next to the training step's peak, else on the host."""
        num_bytes = sum(numel * self.wire_dtypes[key].itemsize for key, (numel, _) in self.shards.items())
        total = torch.cuda.get_device_properties(self.device).total_memory
        headroom = total - peak_reserved - num_bytes
        self.on_gpu = self.snapshot_placement == "auto" and headroom >= max(4 * 2**30, total // 20)
        get_logger().info(
            f"Sparse weight snapshot ({num_bytes / 2**30:.1f} GiB) on the {'GPU' if self.on_gpu else 'host'} "
            f"(training peak {peak_reserved / 2**30:.1f} GiB of {total / 2**30:.1f} GiB, "
            f"snapshot={self.snapshot_placement})"
        )

    def track(self, shards: list[tuple]) -> None:
        """Record which elements of each tensor this rank's shards hold."""
        self.shards = {key: (shard.numel(), offset) for key, _, shard, offset, _ in shards}
        self.key_fqn = {key: fqn for key, fqn, *_ in shards}

    def copy_snapshot_to_host(self) -> None:
        """Copy a GPU snapshot (the last broadcast's weights) into the host buffer (allocated at the first
        anchor), for an anchor, through the pinned staging buffer. Callers wait for earlier readers of the
        host buffer first."""
        assert self.on_gpu
        if self.host_buffer is None:
            self._allocate_host()
        with torch.cuda.stream(self.stream):
            for key, bits in self.snapshot.items():
                self._to_host(bits, self.host[key])

    def anchor_parts(self, source: dict[str, torch.Tensor]) -> list[tuple[str, list[int], int, torch.Tensor]]:
        """This rank's slices of a full HF checkpoint, from ``source`` (key -> flat wire bits of its shard)."""
        parts = []
        for key, bits in source.items():
            values, (numel, offset) = bits.view(self.wire_dtypes[key]), self.shards[key]
            for start, name in self.segments[self.key_fqn[key]]:
                begin, end = max(start, offset), min(start + prod(self.hf_tensors[name][0]), offset + numel)
                if begin < end:
                    parts.append((name, self.hf_tensors[name][0], begin - start, values[begin - offset : end - offset]))
        return parts

    @torch.no_grad()
    def initialize(self, model: nn.Module, is_master: bool, peak_reserved: int, boot_dir: Path) -> None:
        """At the first delta (v1), on the training thread: place the snapshot against the measured training
        peak and allocate it (the host buffer only for a CPU snapshot). The snapshot starts as v0, the
        checkpoint in ``boot_dir`` the engines booted from, read as the encode goes (tensors gathered for fp8
        are read here, before ``local_shards`` gathers them)."""
        self.track(self.local_shards(model, is_master))
        self._place(peak_reserved)
        self.staging = torch.empty(1 << 28, dtype=torch.uint8, pin_memory=True)
        if self.on_gpu:
            device = f"cuda:{self.device}"
            self.snapshot = {
                key: torch.empty(numel, dtype=self._bits(key), device=device) for key, (numel, _) in self.shards.items()
            }
        else:
            self._allocate_host()
            self._page_lock()
            self.snapshot = dict(self.host)
        self.boot_files = checkpoint_index(boot_dir)
        self.boot_pending = set(self.shards) - self.gather
        # The snapshot was just allocated on this stream, whose earlier work may still use that memory
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for key in self.gather & set(self.shards):
                self._read_boot(key)
            self.stream.synchronize()
        self.initialized = True

    def _read_boot(self, key: str) -> None:
        """Fill the snapshot of ``key`` with this rank's slice of the boot checkpoint, through the pinned
        staging buffer, on the current stream."""
        out = self.snapshot[key].view(self.wire_dtypes[key])
        numel, offset = self.shards[key]
        staging = self.staging.numpy()
        for start, name in self.segments[self.key_fqn[key]]:
            begin, end = max(start, offset), min(start + prod(self.hf_tensors[name][0]), offset + numel)
            if begin >= end:
                continue
            dtype = self.boot_files[name][1]
            step = self.staging.numel() // dtype.itemsize
            for chunk in range(begin, end, step):
                count = min(step, end - chunk)
                read_checkpoint_range(self.boot_files[name], chunk - start, chunk - start + count, staging)
                out[chunk - offset : chunk - offset + count].copy_(
                    self.staging[: count * dtype.itemsize].view(dtype), non_blocking=True
                )
                torch.cuda.current_stream().synchronize()

    @torch.no_grad()
    def encode(
        self, shards: list[tuple], ready: torch.cuda.Event, weights_read: threading.Event, emit: Callable
    ) -> None:
        """Diff the shards against the snapshot and advance it, on the side stream once ``ready`` (recorded
        after the optimizer step) has passed. Hands the delta to ``emit`` in parts ``(entries, gaps, value
        bytes, new-weight checksum per HF tensor part)`` on the host as they are encoded. Sets ``weights_read``
        once done reading the weights."""
        torch.cuda.set_device(self.device)
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                self._diff(shards, emit)
                self.stream.synchronize()
        finally:
            weights_read.set()

    def stream_to_host(self, tensor: torch.Tensor):
        """A tensor's bytes on the host; for a GPU tensor, in chunks through a pinned staging buffer, each to be
        written before the next is taken."""
        data = tensor.reshape(-1).view(torch.uint8)
        if not data.is_cuda:
            yield data.numpy()
            return
        if self.write_staging is None:
            self.write_staging = torch.empty(1 << 28, dtype=torch.uint8, pin_memory=True)
            self.write_stream = torch.cuda.Stream(self.device)
        with torch.cuda.stream(self.write_stream):
            for begin in range(0, data.numel(), self.write_staging.numel()):
                chunk = self.write_staging[: min(self.write_staging.numel(), data.numel() - begin)]
                chunk.copy_(data[begin : begin + chunk.numel()], non_blocking=True)
                self.write_stream.synchronize()
                yield chunk.numpy()

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
        self, cache_key, segments: list[tuple[int, str]], offset: int, numel: int
    ) -> tuple[list[str], list[int], list[int], list[int]]:
        """The HF tensor parts of ``segments`` in this rank's ``numel`` elements from ``offset``: names,
        begin/end in the shard, and the shift from a shard position to the HF flat index."""
        if cache_key not in self.shard_segments:
            names, begins, ends, shifts = [], [], [], []
            for start, name in segments:
                hf_numel = prod(self.hf_tensors[name][0])
                if start < offset + numel and start + hf_numel > offset:
                    names.append(name)
                    begins.append(max(start - offset, 0))
                    ends.append(min(start - offset + hf_numel, numel))
                    shifts.append(offset - start)
            self.shard_segments[cache_key] = (names, begins, ends, shifts)
        return self.shard_segments[cache_key]

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

    def _part(self, cache_key, segments: list[tuple[int, str]], position, values, checksums) -> tuple:
        """A part entry for the changes at global flat ``position`` of a trainer tensor made of ``segments``:
        ``(names, counts, gaps, value bytes, checksum names, checksums)``."""
        if cache_key not in self.segment_starts:
            self.segment_starts[cache_key] = torch.tensor([start for start, _ in segments], device=position.device)
        starts = self.segment_starts[cache_key]
        segment = torch.searchsorted(starts, position, right=True) - 1
        entry_starts = torch.ones_like(segment, dtype=torch.bool)
        entry_starts[1:] = segment[1:] != segment[:-1]
        gaps = encode_gaps(position - starts[segment], entry_starts)
        counts = torch.bincount(segment, minlength=len(starts))
        return [name for _, name in segments], counts, gaps, values.view(torch.uint8), *checksums

    def _scale_part(self, fqn: str, first_block: int, previous: torch.Tensor, current: torch.Tensor) -> tuple:
        """The part entry of an fp8 tensor's changed block scales: block rows from ``first_block`` on."""
        scale_cols = current.shape[1]
        delta = as_bits(current).reshape(-1) - as_bits(previous).reshape(-1)
        index = delta.nonzero().squeeze(1)
        self.scale_changes += len(index)
        offset = first_block * scale_cols
        names, begins, ends, shifts = self._segments((fqn, "scales"), self.scale_segments[fqn], offset, current.numel())
        sums = self._segment_sums(
            as_bits(current).reshape(-1), begins, ends, [begin + shift for begin, shift in zip(begins, shifts)]
        )
        return self._part(
            (fqn, "scales"), self.scale_segments[fqn], index + offset, zigzag(delta[index]), (names, sums)
        )

    def _diff(self, shards: list[tuple], emit: Callable) -> None:
        """Diff ``shards`` against the snapshot on the current stream, writing the new bits to the
        snapshot, in chunks of ``CHUNK`` elements (fp8: whole 128-row blocks, and their block scales).
        Copies the delta to the host and hands it to ``emit`` every ``PART`` changes."""
        device = torch.device(f"cuda:{self.device}")
        pending: list[tuple] = []
        num_pending = 0
        self.scale_changes = 0

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
            if key in self.boot_pending:
                self._read_boot(key)
                self.boot_pending.discard(key)
            previous = out = self.snapshot[key]
            wire = self.wire_dtypes[key]
            segment_names, begins, ends, shifts = self._segments(key, self.segments[fqn], offset, shard.numel())
            sums = torch.zeros(len(segment_names), dtype=torch.int64, device=device)
            changed, changed_values, scales = [], [], []
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
                    scales.append((previous_scales, current_scales))
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
            pending.append(
                self._part(
                    fqn,
                    self.segments[fqn],
                    torch.cat(changed) + offset,
                    torch.cat(changed_values),
                    (segment_names, sums),
                )
            )
            num_pending += len(pending[-1][2])
            if scales:
                # This rank sends the scales of the blocks that start in its rows
                first_row = offset // num_cols
                first_block, end_block = (
                    -(-row // FP8_BLOCK[0]) for row in (first_row, first_row + len(shard) // num_cols)
                )
                previous_scales, current_scales = (torch.cat(side) for side in zip(*scales))
                if gathered is None:
                    first_block = first_row // FP8_BLOCK[0]
                else:
                    previous_scales, current_scales = (
                        s[first_block:end_block] for s in (previous_scales, current_scales)
                    )
                if current_scales.numel():
                    pending.append(self._scale_part(fqn, first_block, previous_scales, current_scales))
            if num_pending >= self.PART:
                flush()
                num_pending = 0
        if pending:
            flush()


def _model_dir(model: nn.Module) -> Path:
    """The local dir of the checkpoint the model was loaded from, as the trainer resolves it."""
    name = model.config.name_or_path
    return Path(name) if Path(name).exists() else Path(snapshot_download(repo_id=name, repo_type="model"))


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
        # Background writes that read the snapshot (the next encode waits for them), and anchor writes from the
        # host buffer of a GPU snapshot (the next anchor copy waits for them)
        self.snapshot_writes: list[Future] = []
        self.anchor_writes: list[Future] = []
        self.errors: list[BaseException] = []
        self.weights_read: threading.Event | None = None
        self.part_bytes = 0
        # The checkpoint the engines hold before the first delta: the model's (None), or a resumed run's first
        # broadcast
        self.boot_dir: Path | None = None

    def wait_for_weight_reads(self) -> None:
        if self.weights_read is not None:
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
        self.wait_for_weight_reads()
        if self.world.is_master:
            (step_dir / DELTA_DIR).mkdir(exist_ok=True)
        if self.encoder is None:
            self.encoder = SparseDeltaEncoder(model, self.config.snapshot)
        encoder = self.encoder
        base_step, self.last_step = self.last_step, step
        if base_step is None and step == 0:
            # The run's first broadcast is the checkpoint the engines booted from.
            self.anchor_step = step
            if self.world.is_master:
                manifest = {"step": step, "base_step": None, "ranks": 0, "tensors": encoder.hf_tensors}
                _write_json(step_dir / DELTA_DIR / MANIFEST, manifest)
                WeightSender._finish(self, step, step_dir)
            return
        if base_step is None:
            # A resumed run's startup broadcast: a full checkpoint, written from the weights before training. The
            # next broadcast's delta is against it.
            shards = encoder.local_shards(model, self.world.is_master)
            encoder.track(shards)
            source = {key: as_bits(shard.reshape(-1).to(encoder.wire_dtypes[key])) for key, _, shard, _, _ in shards}
            torch.cuda.synchronize()
            self._write_full(step, step_dir, encoder.anchor_parts(source))
            self.boot_dir = step_dir
            return
        if encoder.fp8_layout is None:
            # Every engine worker reports which tensors it holds in fp8 when it loads the first broadcast.
            layout = [self._wait_for_engine_layout(step_dir.parent) if self.world.is_master else None]
            dist.broadcast_object_list(layout, src=0)
            encoder.set_fp8_layout(model, layout[0], self.world.is_master)
        if not encoder.initialized:
            encoder.initialize(
                model, self.world.is_master, torch.cuda.max_memory_reserved(), self.boot_dir or _model_dir(model)
            )
        shards = encoder.local_shards(model, self.world.is_master)
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

    @staticmethod
    def _wait(writes: list[Future]) -> None:
        for write in writes:
            write.exception()
        writes.clear()

    def _write_periodic_anchor(self, step: int, step_dir: Path) -> None:
        """Write this rank's slices of the last broadcast's full checkpoint in the background. A GPU snapshot
        goes through the host buffer, so the next encode does not wait for the write; a CPU snapshot is
        written from itself, and the next encode waits for that."""
        encoder = self.encoder
        assert encoder is not None
        if encoder.on_gpu:
            self._wait(self.anchor_writes)
            encoder.copy_snapshot_to_host()
            parts, writes = encoder.anchor_parts(encoder.host), self.anchor_writes
        else:
            parts, writes = encoder.anchor_parts(encoder.snapshot), self.snapshot_writes
        writes.append(self._submit(self.anchors, self._write_anchor, step, step_dir, parts))

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
        """Encode thread: diff v{step}, handing the delta to the writer in parts as it is encoded."""
        encoder = self.encoder
        assert encoder is not None
        self._wait(self.snapshot_writes)
        start = time.perf_counter()
        num_parts = 0

        def emit(part: tuple) -> None:
            nonlocal num_parts
            if num_parts == 0 and self.world.is_master:
                self._submit(self.writes, self._start_delta, step, base_step, step_dir)
            self._submit(self.writes, self._write_part, step_dir, num_parts, part)
            num_parts += 1

        encoder.encode(shards, ready, weights_read, emit)
        # Free the gathered fp8 tensors now, not after the anchor copy below, which may overlap the next step's
        # memory peak.
        shards.clear()
        if num_parts == 0 and self.world.is_master:
            self._submit(self.writes, self._start_delta, step, base_step, step_dir)
        encode_time = time.perf_counter() - start
        args = (step, base_step, step_dir, num_parts, encode_time, encoder.scale_changes)
        self._submit(self.writes, self._finish_delta, *args)
        if anchor:
            self._write_periodic_anchor(step, step_dir)

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

    def _finish_delta(
        self, step: int, base_step: int, step_dir: Path, num_parts: int, encode_time: float, scale_changes: int
    ) -> None:
        """After this rank's last part: its parts marker. The master then waits for every rank's marker and
        commits the step. Runs on the writer thread, without collectives."""
        delta_dir = step_dir / DELTA_DIR
        _write_json(
            delta_dir / rank_parts_marker(self.world.rank),
            {"parts": num_parts, "bytes": self.part_bytes, "scale_changes": scale_changes},
        )
        self.part_bytes = 0
        if not self.world.is_master:
            return
        markers = [delta_dir / rank_parts_marker(rank) for rank in range(self.world.world_size)]
        while not all(marker.exists() for marker in markers):
            time.sleep(0.05)
            revalidate(delta_dir)
        WeightSender._finish(self, step, step_dir)
        written = [json.loads(marker.read_text()) for marker in markers]
        num_bytes = sum(marker["bytes"] for marker in written)
        scales = (
            f", {sum(marker['scale_changes'] for marker in written)} fp8 block scales changed"
            if self.encoder.fp8
            else ""
        )
        self.logger.info(
            f"Sparse weight delta v{step} (base v{base_step}): {num_bytes / 2**20:.1f} MiB, "
            f"encode {encode_time:.2f}s (parts written as encoded){scales}"
        )

    def _write_full(self, step: int, step_dir: Path, parts: list) -> None:
        """Publish v{step} as a full checkpoint: every rank writes its slices, the master commits."""
        self._write_anchor(step, step_dir, parts)
        if self.world.is_master:
            WeightSender._finish(self, step, step_dir)

    def _write_anchor(self, step: int, step_dir: Path, parts: list) -> None:
        start = time.perf_counter()
        write_anchor_shards(parts, step_dir, self.world.rank, self.world.world_size, self.encoder.stream_to_host)
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
