import json
import mmap
import queue
import threading
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

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
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path
from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    FP8_BLOCK,
    MANIFEST,
    AnchorWriter,
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
from prime_rl.utils.weight_store import S3Path, StoragePath, remove, storage_path
from prime_rl.utils.weights import convert_state_dict_to_hf, resolve_fqn, resolve_wire_dtype

if TYPE_CHECKING:
    from prime_rl.trainer.optim.offload import FullCPUOffloadOptimizer


class SparseDeltaEncoder:
    """Rank-local deltas of the trainer's weight shards, in HF coordinates.

    Each rank diffs its own shards against the wire-dtype bits of the previous broadcast (the snapshot), so no
    weights are gathered. The prime->HF conversion is replayed once on meta tensors: every HF tensor must be a
    contiguous view of one trainer tensor, which maps a changed trainer element to its HF name and flat index.

    ``encode`` runs on a background thread and a side CUDA stream while the next step trains, without collectives.
    The snapshot starts as the boot checkpoint, read back from storage, and lives on the GPU if it fits next to the
    first step's peak memory, else in the host buffer.

    Tensors the engines hold in fp8 are quantized with vLLM's own quantizer (old and new weights) and diffed as fp8
    bytes; their changed block scales travel as ``"{hf_name}_scale_inv"`` tensors. fp8 tensors whose 128-row blocks
    straddle ranks are gathered.

    With a fully offloaded optimizer (``bind_optimizer``), the tensors stored in its parameters are diffed in its
    step (``optimizer_hook``): each shard still holds the previous broadcast's weights when the step writes the new
    ones, so they need no snapshot. ``encode`` then only checks that they still hold what the step wrote.
    """

    # Elements per diff chunk and changes per written part. Smaller in the optimizer step, which overlaps the final
    # backward.
    CHUNK = 1 << 26
    PART = 1 << 26
    HOOK_CHUNK = 1 << 24
    HOOK_PART = 1 << 24
    # GPU slots of the optimizer step's diff thread, each twice the largest parameter shard
    HOOK_SLOTS = 2

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
        # fqn -> (ue8m0 scales, columns) of trainer tensors held in fp8, quantized as one [rows, columns] matrix
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
        self.key_fqn: dict[str, str] = {}
        # key -> previous wire bits: GPU tensors, or views into the host buffer
        self.snapshot: dict[str, torch.Tensor] = {}
        self.on_gpu: bool | None = None
        self.host_buffer: torch.Tensor | None = None
        self.write_staging: torch.Tensor | None = None
        # The boot checkpoint's ``checkpoint_index``, and keys whose v0 is still to read from it
        self.boot_files: dict[str, tuple] = {}
        self.boot_pending: set[str] = set()
        self.host: dict[str, torch.Tensor] = {}
        self.initialized = False
        # Optimizer-step deltas: parameter storage -> (key, fqn, numel, global flat offset, (shape, stride, storage
        # offset) of the view); the step's parts, diff futures and checksums of the written wire bits
        self.param_keys: dict[int, list[tuple[str, str, int, int, tuple]]] = defaultdict(list)
        self.optimizer_keys: set[str] = set()
        self.step_parts: list[tuple] = []
        self.diffs: list[Future] = []
        self.expected: dict[str, torch.Tensor] = {}
        self.anchor: AnchorWriter | None = None
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
        elements. Not flattened here: ``reshape(-1)`` copies a non-contiguous slice."""
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
        """This rank's ``(key, fqn, shard, global flat offset, gathered)``, taken on the training thread (FSDP swaps
        module parameters during forward, but the sharded storage stays put). ``gathered`` is ``(previous,
        current)`` full bf16 tensors of fp8 tensors whose blocks straddle ranks, all-gathered here because
        ``encode`` runs no collectives."""
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

    def allocate_host(self) -> None:
        """One host buffer for all shards (the CPU snapshot, or anchor copies of a GPU snapshot). Faulted in before
        training: on a node short of free memory, faulting it in during an anchor copy waits minutes for the kernel
        to reclaim page cache while the trainer waits for that copy."""
        start = time.perf_counter()
        sizes = {key: -(-numel * self.wire_dtypes[key].itemsize // 8) * 8 for key, (numel, _) in self.shards.items()}
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE
        # mmap needs a non-empty mapping: a rank may hold no shards
        self.host_buffer = torch.from_numpy(np.frombuffer(mmap.mmap(-1, max(sum(sizes.values()), 8), flags), np.uint8))
        offset = 0
        for key, (numel, _) in self.shards.items():
            self.host[key] = self.host_buffer[offset : offset + numel * self.wire_dtypes[key].itemsize].view(
                self._bits(key)
            )
            offset += sizes[key]
        get_logger().info(
            f"Allocated the {self.host_buffer.numel() / 2**30:.1f} GiB sparse weight host buffer in "
            f"{time.perf_counter() - start:.1f}s"
        )

    def _page_lock(self) -> None:
        """Only while no training step runs: registering a large buffer stalls the trainer's CUDA calls."""
        start = time.perf_counter()
        page_lock(self.host_buffer)
        get_logger().info(
            f"Page-locked the {self.host_buffer.numel() / 2**30:.1f} GiB sparse weight host buffer in "
            f"{time.perf_counter() - start:.1f}s"
        )

    def _snapshot_bytes(self) -> int:
        return sum(numel * self.wire_dtypes[key].itemsize for key, (numel, _) in self.shards.items())

    def _place(self, peak_reserved: int) -> None:
        """Keep the snapshot on the GPU if it fits next to the training step's peak, else on the host."""
        num_bytes = self._snapshot_bytes()
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
        self.key_fqn.update({key: fqn for key, fqn, *_ in shards})

    def copy_snapshot_to_host(self, give_up: Callable[[], bool]) -> bool:
        """Copy a GPU snapshot into the host buffer for an anchor; False if ``give_up()`` turned true first. Callers
        wait for earlier readers of the host buffer."""
        with torch.cuda.stream(self.stream):
            for key, bits in self.snapshot.items():
                if give_up():
                    return False
                self._to_host(bits, self.host[key])
        return True

    def _slices(self, key: str, numel: int, offset: int) -> list[tuple[str, list[int], int, int, int]]:
        """``(hf_name, shape, flat start in the HF tensor, begin, end in the shard)`` of the HF tensors in a
        shard of ``numel`` elements at global flat ``offset``."""
        slices = []
        for start, name in self.segments[self.key_fqn[key]]:
            shape = self.hf_tensors[name][0]
            begin, end = max(start, offset), min(start + prod(shape), offset + numel)
            if begin < end:
                slices.append((name, shape, begin - start, begin - offset, end - offset))
        return slices

    def anchor_parts(self, source: dict[str, torch.Tensor]) -> list[tuple[str, list[int], int, torch.Tensor]]:
        """This rank's slices of a full HF checkpoint, from ``source`` (key -> flat wire bits of its shard)."""
        parts = []
        for key, bits in source.items():
            values = bits.view(self.wire_dtypes[key])
            for name, shape, start, begin, end in self._slices(key, *self.shards[key]):
                parts.append((name, shape, start, values[begin:end]))
        return parts

    @torch.no_grad()
    def initialize(self, model: nn.Module, is_master: bool, peak_reserved: int, boot_dir: Path) -> None:
        """At the first delta (v1), on the training thread: place and allocate the snapshot. It starts as v0, the
        checkpoint in ``boot_dir``, read as the encode goes; tensors gathered for fp8 are read here, before
        ``local_shards`` gathers them."""
        self.track([shard for shard in self.local_shards(model, is_master) if shard[0] not in self.optimizer_keys])
        self._place(peak_reserved)
        self.staging = torch.empty(1 << 28, dtype=torch.uint8, pin_memory=True)
        if self.on_gpu:
            device = f"cuda:{self.device}"
            self.snapshot = {
                key: torch.empty(numel, dtype=self._bits(key), device=device) for key, (numel, _) in self.shards.items()
            }
        else:
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
        """Fill the snapshot of ``key`` with this rank's slice of the boot checkpoint, on the current stream."""
        out = self.snapshot[key].view(self.wire_dtypes[key])
        staging = self.staging.numpy()
        for name, _, hf_start, begin, end in self._slices(key, *self.shards[key]):
            dtype = self.boot_files[name][1]
            step = self.staging.numel() // dtype.itemsize
            for chunk in range(begin, end, step):
                count = min(step, end - chunk)
                first = hf_start + chunk - begin
                read_checkpoint_range(self.boot_files[name], first, first + count, staging)
                out[chunk : chunk + count].copy_(self.staging[: count * dtype.itemsize].view(dtype), non_blocking=True)
                torch.cuda.current_stream().synchronize()

    @torch.no_grad()
    def encode(
        self, shards: list[tuple], ready: torch.cuda.Event, weights_read: threading.Event, emit: Callable
    ) -> int:
        """Diff the shards against the snapshot and advance it, on the side stream after ``ready``. Hands host parts
        ``(entries, gaps, value bytes, checksums)`` to ``emit`` as they are encoded and sets ``weights_read`` once
        done reading the weights. Returns the number of changed fp8 block scales."""
        torch.cuda.set_device(self.device)
        if not self.optimizer_keys:
            self.scale_changes = 0
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                self._diff(shards, emit)
                self.stream.synchronize()
            return self.scale_changes
        finally:
            weights_read.set()

    def stream_to_host(self, tensor: torch.Tensor):
        """A tensor's bytes on the host; for a GPU tensor, in pinned staging chunks, each to be consumed before the
        next is taken."""
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

    def _to_host(
        self, tensor: torch.Tensor, host: torch.Tensor | None = None, staging: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Copy a GPU tensor to pageable host memory (``host``, or a new tensor) through pinned staging: a copy
        straight into pageable memory holds the CUDA driver and stalls the training step's CUDA calls."""
        host = torch.empty(tensor.shape, dtype=tensor.dtype) if host is None else host
        staging = self.staging if staging is None else staging
        data, host_bytes = tensor.view(torch.uint8), host.view(torch.uint8)
        for begin in range(0, data.numel(), staging.numel()):
            chunk = staging[: min(staging.numel(), data.numel() - begin)]
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
        """``bits_checksum`` of each ``bits[begin:end]`` whose first element is at HF flat index ``first``. One
        reduction per run of adjacent equal-length segments (e.g. unstacked experts): few kernel launches, as the
        encode thread shares the GIL with training."""
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
        """Diff ``shards`` against the snapshot on the current stream and advance it, handing a host part to ``emit``
        every ``PART`` changes. Shards diffed in the optimizer step are only checked against what it wrote; its
        remaining parts go first."""
        pending, self.step_parts = self.step_parts, []
        num_pending = sum(len(part[2]) for part in pending)
        checks: list[tuple[str, torch.Tensor, torch.Tensor]] = []

        def flush() -> None:
            emit(self._host_part(pending))
            pending.clear()

        for key, fqn, shard, offset, gathered in shards:
            shard = shard.reshape(-1)
            if key in self.optimizer_keys:
                if key not in self.expected:
                    raise RuntimeError(f"{key} was not written by the optimizer step before this broadcast")
                checks.append((key, self.expected.pop(key), self._checksum(as_bits(shard.to(self.wire_dtypes[key])))))
                continue
            if key in self.boot_pending:
                self._read_boot(key)
                self.boot_pending.discard(key)
            parts = self._diff_shard(key, fqn, shard, offset, gathered, self.snapshot[key], self.snapshot[key])
            pending += parts
            num_pending += sum(len(part[2]) for part in parts)
            if num_pending >= self.PART:
                flush()
                num_pending = 0
        if pending:
            flush()
        if checks:
            keys, expected, live = zip(*checks)
            changed = [key for key, same in zip(keys, torch.cat(expected).eq(torch.cat(live)).tolist()) if not same]
            if changed:
                raise RuntimeError(f"Trainer weights changed outside the optimizer step, not sent: {changed}")

    def wait_for_diffs(self) -> None:
        for diff in self.diffs:
            diff.result()
        self.diffs.clear()

    def _host_part(self, pending: list[tuple], staging: torch.Tensor | None = None) -> tuple:
        """One part ``(entries, gaps, value bytes, checksums)`` on the host from GPU part entries."""
        names = [name for part in pending for name in part[0]]
        counts = torch.cat([part[1] for part in pending]).tolist()
        gaps = self._to_host(torch.cat([part[2] for part in pending]), staging=staging)
        values = self._to_host(torch.cat([part[3] for part in pending]), staging=staging)
        checksum_names = [name for part in pending for name in part[4]]
        checksums = dict(zip(checksum_names, torch.cat([part[5] for part in pending]).tolist()))
        return [(name, count) for name, count in zip(names, counts) if count], gaps, values, checksums

    def _diff_shard(
        self,
        key: str,
        fqn: str,
        shard: torch.Tensor,
        offset: int,
        gathered: tuple | None,
        previous: torch.Tensor,
        out: torch.Tensor | None,
        chunk: int = CHUNK,
    ) -> list[tuple]:
        """The part entries of a flat shard's changes against ``previous`` (its wire bits), on the current stream, in
        chunks of ``chunk`` elements (fp8: whole 128-row blocks, plus their block scales). Writes the new bits to
        ``out`` unless None."""
        device = torch.device(f"cuda:{self.device}")
        wire = self.wire_dtypes[key]
        segment_names, begins, ends, shifts = self._segments(key, self.segments[fqn], offset, shard.numel())
        sums = torch.zeros(len(segment_names), dtype=torch.int64, device=device)
        changed, changed_values, scales = [], [], []
        if fqn in self.fp8:
            ue8m0, num_cols = self.fp8[fqn]
            if gathered is not None:
                chunks = [(0, shard.numel())]
            else:
                rows = max(FP8_BLOCK[0], chunk // num_cols // FP8_BLOCK[0] * FP8_BLOCK[0])
                chunks = [(start, start + rows * num_cols) for start in range(0, shard.numel(), rows * num_cols)]
        else:
            chunks = [(start, start + chunk) for start in range(0, shard.numel(), chunk)]
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
            if out is not None:
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
        parts = [
            self._part(
                fqn,
                self.segments[fqn],
                torch.cat(changed) + offset,
                torch.cat(changed_values),
                (segment_names, sums),
            )
        ]
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
                previous_scales, current_scales = (s[first_block:end_block] for s in (previous_scales, current_scales))
            if current_scales.numel():
                parts.append(self._scale_part(fqn, first_block, previous_scales, current_scales))
        return parts

    def _checksum(self, bits: torch.Tensor) -> torch.Tensor:
        """``bits_checksum`` of a flat shard, in chunks of ``HOOK_CHUNK`` elements."""
        n = bits.numel()
        sums = [
            self._segment_sums(bits, [start], [min(start + self.HOOK_CHUNK, n)], [start])
            for start in range(0, n, self.HOOK_CHUNK)
        ]
        return torch.cat(sums).sum(0, keepdim=True) if sums else torch.zeros(1, dtype=torch.int64, device=bits.device)

    def bind_optimizer(self, model: nn.Module, is_master: bool, params: list[torch.Tensor]) -> None:
        """Diff the trainer tensors stored in ``params`` (the offloaded optimizer's contiguous local shards) in the
        optimizer step (``optimizer_hook``); the snapshot keeps only the other tensors and gathered fp8 tensors. A
        trainer tensor may be a strided view of a parameter (e.g. one half of a packed gate/up weight)."""
        assert all(param.is_contiguous() for param in params)
        stored = {param.untyped_storage().data_ptr() for param in params}
        residual = []
        for shard in self.local_shards(model, is_master):
            key, fqn, tensor, offset, _ = shard
            storage = tensor.untyped_storage().data_ptr()
            if key not in self.gather and storage in stored:
                view = (tuple(tensor.shape), tensor.stride(), tensor.storage_offset())
                self.param_keys[storage].append((key, fqn, tensor.numel(), offset, view))
                self.optimizer_keys.add(key)
            else:
                residual.append(shard)
        self.track(residual)
        slot_bytes = 2 * -(-max(param.nbytes for param in params) // 256) * 256
        self.hook_slots = queue.SimpleQueue()
        for _ in range(self.HOOK_SLOTS):
            self.hook_slots.put(torch.empty(slot_bytes, dtype=torch.uint8, device=f"cuda:{self.device}"))
        self.differ = ThreadPoolExecutor(1)
        self.diff_stream = torch.cuda.Stream(self.device)
        self.diff_staging = torch.empty(1 << 28, dtype=torch.uint8, pin_memory=True)
        num_bytes = self._snapshot_bytes()
        get_logger().info(
            f"Sparse weight deltas of {len(self.optimizer_keys)} tensors come from the optimizer step, "
            f"{len(self.shards)} tensors ({num_bytes / 2**30:.2f} GiB) are diffed against a snapshot"
        )

    def arm(self, anchor: AnchorWriter | None, emit: Callable) -> None:
        """Collect the next optimizer step's delta, handing host parts to ``emit`` as the step goes, and write its
        anchor slices if ``anchor``."""
        self.step_parts, self.expected, self.scale_changes, self.anchor, self.emit = [], {}, 0, anchor, emit

    def anchor_slices(self) -> list[tuple[str, list[int], int, int, torch.dtype]]:
        """Every ``(hf_name, shape, flat start, numel, wire dtype)`` slice of this rank's share of an anchor."""
        shards = {key: (numel, offset) for entries in self.param_keys.values() for key, _, numel, offset, _ in entries}
        shards.update(self.shards)
        return [
            (name, shape, start, end - begin, self.wire_dtypes[key])
            for key, (numel, offset) in shards.items()
            for name, shape, start, begin, end in self._slices(key, numel, offset)
        ]

    @torch.no_grad()
    def optimizer_hook(self, local_param: torch.Tensor, source: torch.Tensor) -> None:
        """``FullCPUOffloadOptimizer.weight_hook``, on its H2D stream: copy the shard (still the previous
        broadcast's weights) and its new values ``source`` (pinned host) into a GPU slot, write the new values and
        diff the slot on the diff thread. At anchor steps, also writes the new values to the anchor from the
        host."""
        start = local_param.storage_offset()
        entries = [
            (key, fqn, numel, offset, (size, stride, at - start))
            for key, fqn, numel, offset, (size, stride, at) in self.param_keys.get(
                local_param.untyped_storage().data_ptr(), []
            )
            if start <= at < start + local_param.numel()
        ]
        if not entries:
            local_param.copy_(source, non_blocking=True)
            return
        slot = self.hook_slots.get()
        half = slot.numel() // 2
        previous = slot[: source.nbytes].view(source.dtype)
        new = slot[half : half + source.nbytes].view(source.dtype)
        previous.copy_(local_param.reshape(-1))
        new.copy_(source.reshape(-1), non_blocking=True)
        local_param.copy_(new.view(local_param.shape))
        ready = torch.cuda.Event()
        ready.record()
        if self.anchor is not None:
            for key, _, numel, offset, (size, stride, at) in entries:
                host = source.as_strided(size, stride, source.storage_offset() + at).reshape(-1)
                host = host.to(self.wire_dtypes[key])
                self.anchor.write_all(
                    [
                        (name, hf_start, host[begin:end].view(torch.uint8).numpy())
                        for name, _, hf_start, begin, end in self._slices(key, numel, offset)
                    ]
                )
        self.diffs.append(self.differ.submit(self._diff_slot, entries, slot, previous, new, ready))

    @torch.no_grad()
    def _diff_slot(self, entries: list[tuple], slot: torch.Tensor, previous, new, ready: torch.cuda.Event) -> None:
        """Diff thread: the parts of one parameter's tensors from its slot, then free the slot. Hands the parts so far
        to ``emit`` every ``HOOK_PART`` changes."""
        torch.cuda.set_device(self.device)
        try:
            with torch.cuda.stream(self.diff_stream):
                self.diff_stream.wait_event(ready)
                for key, fqn, _, offset, (size, stride, at) in entries:
                    wire = self.wire_dtypes[key]
                    current, old = (
                        values.as_strided(size, stride, values.storage_offset() + at).reshape(-1).to(wire)
                        for values in (new, previous)
                    )
                    self.step_parts += self._diff_shard(
                        key, fqn, current, offset, None, as_bits(old), None, self.HOOK_CHUNK
                    )
                    self.expected[key] = self._checksum(as_bits(current))
                self.diff_stream.synchronize()
        finally:
            self.hook_slots.put(slot)
        if sum(len(part[2]) for part in self.step_parts) >= self.HOOK_PART:
            with torch.cuda.stream(self.diff_stream):
                self.emit(self._host_part(self.step_parts, self.diff_staging))
            self.step_parts = []


def _model_dir(model: nn.Module) -> Path:
    """The local dir of the checkpoint the model was loaded from, as the trainer resolves it."""
    name = model.config.name_or_path
    return Path(name) if Path(name).exists() else Path(snapshot_download(repo_id=name, repo_type="model"))


def _write_json(path: StoragePath, value) -> None:
    write_file(path, json.dumps(value).encode())


class SparseFileSystemWeightSender(FileSystemWeightSender):
    """Filesystem broadcast where steps carry only the delta against the previous broadcast (format in
    ``prime_rl.utils.sparse_weights``). The delta is encoded and written in the background while the next step
    trains; the trainer waits only before its next final backward, until the encode has read the weights. Anchors
    (full checkpoints) serve engines that fall off the delta chain. The writer thread commits a step once every
    rank's files are on disk."""

    gathers_full_weights = False
    # Seconds an anchor copy may hold back the next encode before that anchor is skipped: the trainer waits for the
    # encode, and other ranks would hit the offload pipeline's timeout
    ANCHOR_COPY_WAIT = 30.0

    def __init__(self, output_dir: Path, config: FileSystemWeightBroadcastConfig):
        super().__init__(output_dir, config)
        self.config = config
        self.encoder: SparseDeltaEncoder | None = None
        self.last_step: int | None = None
        self.anchor_step: int | None = None
        # Each in order: encodes, delta writes and commits, anchor writes
        self.encodes = ThreadPoolExecutor(1)
        self.writes = ThreadPoolExecutor(1)
        self.anchors = ThreadPoolExecutor(1)
        # Anchor writes that read the snapshot (the next encode waits for them), and anchor writes from the host
        # buffer of a GPU snapshot (the next anchor copy waits for them)
        self.snapshot_writes: list[Future] = []
        self.anchor_writes: list[Future] = []
        self.errors: list[BaseException] = []
        self.weights_read: threading.Event | None = None
        self.encode_queued_at: float | None = None
        self.part_bytes = 0
        # The checkpoint the engines hold before the first delta: the model's (None), or a resumed run's first
        # broadcast
        self.boot_dir: StoragePath | None = None
        self.storage = storage_path(config.storage) if config.storage else get_broadcast_dir(output_dir)
        if config.storage and self.world.is_master:
            # Leftovers of an earlier run would look like anchors and deltas of this one
            for path in self.storage.glob("step_*"):
                remove(path)
        self.optimizer: "FullCPUOffloadOptimizer | None" = None
        self.model: nn.Module | None = None
        # The step whose delta the next optimizer step collects
        self.armed_step: int | None = None
        self.streamed_dir: Path | None = None
        self.num_streamed = 0

    def diff_in_optimizer(self, optimizer: "FullCPUOffloadOptimizer") -> None:
        """Take the deltas of the optimizer's parameters from its step (``SparseDeltaEncoder.optimizer_hook``); needs
        a broadcast after every optimizer step. Not on an object store: it writes local files in place."""
        if isinstance(self.storage, S3Path):
            self.logger.info("Sparse weight deltas on an object store are diffed against a snapshot")
            return
        self.optimizer = optimizer

    def wait_for_weight_reads(self) -> None:
        self._wait_for_encode()
        if self.optimizer is not None and self.last_step is not None and self.armed_step is None:
            self._arm(self.last_step + 1)

    def _wait_for_encode(self) -> None:
        if self.weights_read is not None:
            start = time.perf_counter()
            self.weights_read.wait()
            self.weights_read = None
            self.logger.debug(f"Waited {time.perf_counter() - start:.2f}s for the sparse weight encode")
        self._raise_errors()

    def _arm(self, step: int) -> None:
        """Collect v{step}'s delta, and its anchor slices at anchor steps, in the coming optimizer step."""
        encoder = self.encoder
        self._set_fp8_layout()
        if not encoder.optimizer_keys:
            params = [master.model_param.to_local() for master in self.optimizer._master_weights.values()]
            encoder.bind_optimizer(self.model, self.world.is_master, params)
            encoder.allocate_host()
        anchor = None
        if step % self.config.anchor_interval == 0:
            # Written next to the step dir, which the broadcast handshake recreates, and moved in once complete
            anchor = AnchorWriter(encoder.anchor_slices(), get_step_path(self.storage, step), self.world.rank)
        # Parts written during the step go next to the step dir too, and are moved in when it is published
        step_dir = get_step_path(self.storage, step)
        self.streamed_dir = step_dir.with_name(f".{step_dir.name}.delta{self.world.rank}")
        self.streamed_dir.mkdir(parents=True, exist_ok=True)
        self.num_streamed = 0
        encoder.arm(anchor, self._stream_part)
        self.optimizer.weight_hook = encoder.optimizer_hook
        self.armed_step = step

    def _stream_part(self, part: tuple) -> None:
        path = self.streamed_dir / rank_part(self.world.rank, self.num_streamed)
        self._submit(self.writes, self._write_part, path, part)
        self.num_streamed += 1

    def _set_fp8_layout(self) -> None:
        if self.encoder.fp8_layout is None:
            # Every engine worker reports which tensors it holds in fp8 at its first update
            layout = [self._wait_for_engine_layout(self.storage) if self.world.is_master else None]
            dist.broadcast_object_list(layout, src=0)
            self.encoder.set_fp8_layout(self.model, layout[0], self.world.is_master)

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
        self._wait_for_encode()
        step_dir = get_step_path(self.storage, step)
        # Every rank: on a separate storage dir, no handshake creates the step dir before the others write to it
        (step_dir / DELTA_DIR).mkdir(parents=True, exist_ok=True)
        if self.encoder is None:
            self.encoder = SparseDeltaEncoder(model, self.config.snapshot)
            self.model = model
        encoder = self.encoder
        base_step, self.last_step = self.last_step, step
        if self.optimizer is not None and base_step is not None:
            if self.armed_step != step:
                raise RuntimeError(f"Sparse weight delta v{step} was not collected in the optimizer step")
            self.optimizer.weight_hook = None
            self.armed_step = None
        if base_step is None:
            shards = encoder.local_shards(model, self.world.is_master)
            encoder.track(shards)
            if self.optimizer is None:
                encoder.allocate_host()
        if base_step is None and step == 0:
            # The run's first broadcast is the checkpoint the engines booted from
            self.anchor_step = step
            if self.world.is_master:
                self._write_manifest(step_dir, step, None, 0)
                WeightSender._finish(self, step, self.step_dir(step))
        elif base_step is None:
            # A resumed run's first broadcast: a full checkpoint, which the next delta is against
            source = {key: as_bits(shard.reshape(-1).to(encoder.wire_dtypes[key])) for key, _, shard, _, _ in shards}
            torch.cuda.synchronize()
            self._write_full(step, step_dir, encoder.anchor_parts(source))
            self.boot_dir = step_dir
        if base_step is None:
            if self.optimizer is not None:
                # Engines report their layout as they load this broadcast. Take it (and compile the fp8 quantizer)
                # now, not in the first optimizer step, next to its activations.
                self._set_fp8_layout()
            return
        self._set_fp8_layout()
        if not encoder.initialized:
            encoder.initialize(
                model, self.world.is_master, torch.cuda.max_memory_reserved(), self.boot_dir or _model_dir(model)
            )
        shards = encoder.local_shards(model, self.world.is_master)
        anchor = encoder.anchor if self.optimizer is not None else step % self.config.anchor_interval == 0
        ready = torch.cuda.Event()
        ready.record()
        self.weights_read = threading.Event()
        self.encode_queued_at = time.monotonic()
        args = (step, base_step, step_dir, shards, ready, self.weights_read, anchor)
        self._submit(self.encodes, self._encode, *args)

    def _wait_for_engine_layout(self, broadcast_dir: StoragePath) -> dict[str, str]:
        start = time.monotonic()
        while (layout := read_engine_layout(broadcast_dir)) is None:
            if time.monotonic() - start > self.timeout:
                raise TimeoutError(f"Not every engine worker reported its weight layout within {self.timeout}s")
            time.sleep(0.1)
        return layout

    @staticmethod
    def _wait(writes: list[Future]) -> None:
        for write in writes:
            write.exception()
        writes.clear()

    def _write_periodic_anchor(self, step: int, step_dir: StoragePath) -> None:
        """Write this rank's slices of the last broadcast in the background. A GPU snapshot is first copied to the
        host buffer, so the next encode waits only for that copy; a CPU snapshot is written from itself, and the
        next encode waits for the write."""
        encoder = self.encoder
        if encoder.on_gpu:

            def held_too_long() -> bool:
                queued_at = self.encode_queued_at
                return queued_at is not None and time.monotonic() - queued_at > self.ANCHOR_COPY_WAIT

            while not all(write.done() for write in self.anchor_writes) and not held_too_long():
                time.sleep(0.1)
            if held_too_long() or not encoder.copy_snapshot_to_host(held_too_long):
                self.logger.warning(
                    f"Skipped the {step_dir.name} anchor: its copy to host memory held the next encode for over "
                    f"{self.ANCHOR_COPY_WAIT:.0f}s"
                )
                write_file(step_dir / f"anchor_rank{self.world.rank}.skipped", b"")
                return
            self._wait(self.anchor_writes)
            parts, writes = encoder.anchor_parts(encoder.host), self.anchor_writes
        else:
            parts, writes = encoder.anchor_parts(encoder.snapshot), self.snapshot_writes
        writes.append(self._submit(self.anchors, self._write_anchor, step, step_dir, parts))

    def _encode(
        self,
        step: int,
        base_step: int,
        step_dir: StoragePath,
        shards,
        ready,
        weights_read: threading.Event,
        anchor: bool | AnchorWriter | None,
    ) -> None:
        """Encode thread: diff v{step}, handing the delta to the writer in parts. ``anchor``: write a periodic
        anchor (True), or finish the one the optimizer step wrote."""
        encoder = self.encoder
        self.encode_queued_at = None
        self._wait(self.snapshot_writes)
        start = time.perf_counter()
        num_parts = 0

        def emit(part: tuple) -> None:
            nonlocal num_parts
            path = step_dir / DELTA_DIR / rank_part(self.world.rank, num_parts)
            self._submit(self.writes, self._write_part, path, part)
            num_parts += 1

        try:
            encoder.wait_for_diffs()
            if self.optimizer is not None:
                num_parts = self.num_streamed
            if self.world.is_master:
                self._submit(self.writes, self._write_manifest, step_dir, step, base_step, self.world.world_size)
            if self.optimizer is not None:
                self._submit(self.writes, self._move_streamed, self.streamed_dir, step_dir / DELTA_DIR)
            scale_changes = encoder.encode(shards, ready, weights_read, emit)
        except BaseException as error:
            # Recorded before the trainer wakes up, so it fails at its next wait instead of training on
            self.errors.append(error)
            weights_read.set()
            raise
        # Free the gathered fp8 tensors before the anchor copy below, which may overlap the next step's memory peak
        shards.clear()
        encode_time = time.perf_counter() - start
        args = (step, base_step, step_dir, num_parts, start, encode_time, scale_changes)
        self._submit(self.writes, self._finish_delta, *args)
        if isinstance(anchor, AnchorWriter):
            # The snapshot tensors hold v{step} until the next encode, which runs on this thread
            anchor.write_parts(encoder.anchor_parts(encoder.snapshot), encoder.stream_to_host)
            self._submit(self.anchors, self._close_anchor, step, step_dir, anchor)
        elif anchor:
            self._write_periodic_anchor(step, step_dir)

    def _write_manifest(self, step_dir: StoragePath, step: int, base_step: int | None, ranks: int) -> None:
        """Written before the parts, so engines can read part files as they appear."""
        manifest = {"step": step, "base_step": base_step, "ranks": ranks, "tensors": self.encoder.hf_tensors}
        _write_json(step_dir / DELTA_DIR / MANIFEST, manifest)

    def _write_part(self, path: StoragePath, part: tuple) -> None:
        entries, gaps, values, checksums = part
        self.part_bytes += write_rank_file(
            path, entries, gaps.numpy(), values.numpy(), self.encoder.hf_tensors, checksums
        )

    @staticmethod
    def _move_streamed(streamed_dir: Path, delta_dir: Path) -> None:
        delta_dir.mkdir(parents=True, exist_ok=True)
        for path in streamed_dir.iterdir():
            path.rename(delta_dir / path.name)
        streamed_dir.rmdir()

    def _finish_delta(
        self,
        step: int,
        base_step: int,
        step_dir: StoragePath,
        num_parts: int,
        start: float,
        encode_time: float,
        scale_changes: int,
    ) -> None:
        """After this rank's last part, its parts marker. The master then waits for every rank's marker and commits
        the step (on the writer thread, without collectives)."""
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
        WeightSender._finish(self, step, self.step_dir(step))
        written = [json.loads(marker.read_text()) for marker in markers]
        num_bytes = sum(marker["bytes"] for marker in written)
        scales = (
            f", {sum(marker['scale_changes'] for marker in written)} fp8 block scales changed"
            if self.encoder.fp8
            else ""
        )
        self.logger.info(
            f"Sparse weight delta v{step} (base v{base_step}): {num_bytes / 2**20:.1f} MiB, "
            f"encode {encode_time:.2f}s, committed {time.perf_counter() - start:.2f}s after the encode started{scales}"
        )

    def _write_full(self, step: int, step_dir: StoragePath, parts: list) -> None:
        """Publish v{step} as a full checkpoint: every rank writes its slices, the master commits."""
        self._write_anchor(step, step_dir, parts)
        if self.world.is_master:
            WeightSender._finish(self, step, self.step_dir(step))

    def _write_anchor(self, step: int, step_dir: StoragePath, parts: list) -> None:
        write_anchor_shards(parts, step_dir, self.world.rank, self.world.world_size, self.encoder.stream_to_host)
        self._wait_for_anchor(step, step_dir)

    def _close_anchor(self, step: int, step_dir: Path, anchor: AnchorWriter) -> None:
        anchor.close(self.world.world_size)
        self._wait_for_anchor(step, step_dir)

    def _wait_for_anchor(self, step: int, step_dir: Path) -> None:
        if self.world.is_master:
            while not anchor_complete(step_dir):
                if any(step_dir.glob("anchor_rank*.skipped")):
                    self.logger.warning(f"The {step_dir.name} anchor is incomplete: a rank skipped it")
                    return
                time.sleep(0.1)
            self.anchor_step = max(self.anchor_step or 0, step)

    def _finish(self, step: int, step_dir: Path) -> None:
        """Committed in ``_broadcast`` at v0, else in the background (``_finish_delta``, ``_write_full``)."""

    def _clean(self, step: int) -> None:
        # Engines replay from the last anchor, so keep everything since it
        step = step if self.anchor_step is None else min(step, self.anchor_step + 1)
        super()._clean(step)
        if self.config.storage:  # the base class cleans the broadcast dir
            for path in self.storage.glob("step_*"):
                if int(path.name.removeprefix("step_")) < step - 1:
                    remove(path)
