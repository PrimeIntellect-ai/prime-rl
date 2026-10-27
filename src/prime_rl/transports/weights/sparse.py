import json
import queue
import shutil
import threading
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.shared import FileSystemWeightBroadcastConfig
from prime_rl.trainer.models.fusions import get_model_packed_parameters, join_module_path
from prime_rl.transports.weights.base import WeightSender
from prime_rl.transports.weights.filesystem import FileSystemWeightSender
from prime_rl.transports.weights.sparse_kernels import Pieces, build_pieces, count_changes, write_changes
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path
from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    ENGINE_LAYOUT,
    FP8_BLOCK,
    MANIFEST,
    AnchorWriter,
    anchor_complete,
    as_bits,
    compile_fp8_quantizer,
    encode_gaps,
    quantize_fp8_blocks,
    rank_failed_marker,
    rank_part,
    rank_parts_marker,
    read_engine_layout,
    revalidate,
    write_file,
    write_rank_file,
    zigzag,
)
from prime_rl.utils.weight_store import S3Path, StoragePath, remove, storage_path
from prime_rl.utils.weights import convert_state_dict_to_hf, resolve_fqn, resolve_wire_dtype

if TYPE_CHECKING:
    from prime_rl.trainer.optim import OptimizerLike


def trainer_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """The model's state-dict entries without running its state-dict hooks (no collectives). A packed parameter
    (``fusions``) whose packing dimension is sharded stays packed: its checkpoint entries would be gathered copies."""
    tensors = dict(model.named_parameters(remove_duplicate=False))
    for name, module in model.named_modules(remove_duplicate=False):
        for buffer_name, buffer in module.named_buffers(recurse=False):
            if buffer_name not in module._non_persistent_buffers_set:
                tensors[join_module_path(name, buffer_name)] = buffer
    keys = {id(tensor): key for key, tensor in tensors.items()}
    for packed in get_model_packed_parameters(model):
        param, spec = packed.parameter, packed.spec
        if isinstance(param, DTensor) and any(placement.is_shard(spec.dim) for placement in param.placements):
            continue
        prefix = keys[id(param)].removesuffix(spec.name)
        del tensors[keys[id(param)]]
        tensors.update(zip([prefix + name for name in spec.logical_names], spec.split_into_logical_views(param)))
    return tensors


class SparseDeltaEncoder:
    """Rank-local deltas of the trainer's weight shards, in HF coordinates.

    The deltas of the optimizer's parameters come from its step (``bind``): right before the step updates a chunk of
    parameters, their shards still hold the previous broadcast's weights. ``update_hook`` (optimizers on the GPU) and
    ``optimizer_hook`` (``FullCPUOffloadOptimizer``) copy the chunk's old wire bits; right after the update,
    ``sparse_kernels`` compare them with the new bits in place and write the changes to pinned memory without a host
    sync (bf16; fp8 and fp32 tensors are diffed on the diff thread), and the diff thread streams them out as the step
    goes. No copy of the weights is kept. The other tensors (the residual: buffers, frozen parameters, fp8 tensors
    whose 128-row blocks straddle ranks) are diffed in ``encode`` against a copy of their previous wire bits, which
    also checks that the optimizer's tensors still hold what its step wrote.

    The prime->HF conversion is replayed once on meta tensors: every HF tensor must be a contiguous view of one trainer
    tensor, which maps a changed trainer element to its HF name and flat index. Tensors the engines hold in fp8 are
    quantized with vLLM's own quantizer (old and new weights) and diffed as fp8 bytes; their changed block scales
    travel as ``"{hf_name}_scale_inv"`` tensors.
    """

    # Elements per diff chunk and changes per written part: in ``encode``, and in the optimizer step (smaller, as it
    # overlaps training)
    CHUNK = 1 << 26
    PART = 1 << 26
    HOOK_CHUNK = 1 << 24
    HOOK_PART = 1 << 24
    # Updates in flight: GPU slots of their old bf16 bits (full offload: each the size of the largest parameter
    # shard) and pinned regions of their changes; bytes of other old values that ``update_hook`` holds for the diff
    # thread (at least one update's)
    HOOK_SLOTS = 2
    UPDATE_BYTES = 1 << 32
    # Changes per pinned region of the kernels (more take further rounds)
    REGION = 1 << 24

    def __init__(self, model: nn.Module):
        keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
        self.wire_dtypes: dict[str, torch.dtype] = {}
        self.key_fqn: dict[str, str] = {}
        meta: dict[str, torch.Tensor] = {}
        for key, value in trainer_tensors(model).items():
            dtype = resolve_wire_dtype(keep_in_fp32, key, torch.bfloat16) if isinstance(value, DTensor) else value.dtype
            self.wire_dtypes[key] = dtype
            self.key_fqn[key] = resolve_fqn(model, key)
            meta[self.key_fqn[key]] = torch.empty(value.shape, dtype=dtype, device="meta")
        owner = {id(tensor): fqn for fqn, tensor in meta.items()}
        # The checkpoint names of packed parameters are views of them
        canonical = dict(meta)
        for packed in get_model_packed_parameters(model):
            if packed.fqn not in meta:
                continue
            views = packed.spec.split_into_logical_views(canonical.pop(packed.fqn))
            canonical.update(zip(packed.logical_fqns, views))

        segments = defaultdict(list)
        self.hf_tensors: dict[str, tuple[list[int], str]] = {}
        for name, tensor in convert_state_dict_to_hf(model, canonical).items():
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
        # Residual keys -> (local numel, global flat offset) of this rank's shard, and their previous wire bits
        self.shards: dict[str, tuple[int, int]] = {}
        self.previous: dict[str, torch.Tensor] = {}
        # Optimizer parameter storage -> (key, fqn, numel, global flat offset, (shape, stride, storage offset) of the
        # view) of the trainer tensors stored in it, and the checksum of each one's last written wire bits
        self.param_keys: dict[int, list[tuple[str, str, int, int, tuple]]] = defaultdict(list)
        self.checksums: dict[str, torch.Tensor] = {}
        self.residual: set[str] = set()
        self.differ: ThreadPoolExecutor | None = None
        # The armed step's parts and diffs, and its anchor
        self.armed = False
        self.step_parts: list[tuple] = []
        self.diffs: list[Future] = []
        self.anchor: AnchorWriter | None = None
        self.stepped: set[str] = set()
        self.write_staging: torch.Tensor | None = None
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
        for key, value in trainer_tensors(model).items():
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
    def local_shards(self, model: nn.Module, is_master: bool, gather: bool = False) -> list[tuple]:
        """This rank's ``(key, fqn, shard, global flat offset, gathered)``, taken on the training thread (FSDP swaps
        module parameters during forward, but the sharded storage stays put). With ``gather``, ``gathered`` is
        ``(previous, current)`` full bf16 tensors of fp8 tensors whose blocks straddle ranks, all-gathered here
        because ``encode`` runs no collectives."""
        shards = []
        for key, value in trainer_tensors(model).items():
            fqn = self.key_fqn[key]
            if fqn not in self.segments:
                continue
            shard = self._local_shard(key, value, is_master)
            empty = shard is None or shard[0].numel() == 0
            gathered = None
            if gather and key in self.gather:
                if empty:
                    value.full_tensor()  # every rank joins the all-gathers
                    value.full_tensor()
                else:
                    gathered = (self._gather_previous(key, value), value.full_tensor().to(torch.bfloat16))
            if not empty:
                shards.append((key, fqn, shard[0], shard[1], gathered))
        return shards

    def _gather_previous(self, key: str, value: DTensor) -> torch.Tensor:
        previous = self.previous[key].view(torch.bfloat16)
        local_shape = (previous.numel() // prod(value.shape[1:]), *value.shape[1:])
        return DTensor.from_local(
            previous.view(local_shape), value.device_mesh, value.placements, shape=value.shape, stride=value.stride()
        ).full_tensor()

    def _bits(self, key: str, shard: torch.Tensor) -> torch.Tensor:
        """A shard's flat wire bits (a copy)."""
        return as_bits(shard.reshape(-1).to(self.wire_dtypes[key], copy=True))

    def _slices(self, fqn: str, numel: int, offset: int) -> list[tuple[str, list[int], int, int, int]]:
        """``(hf_name, shape, flat start in the HF tensor, begin, end in the shard)`` of the HF tensors in a
        shard of ``numel`` elements at global flat ``offset``."""
        slices = []
        for start, name in self.segments[fqn]:
            shape = self.hf_tensors[name][0]
            begin, end = max(start, offset), min(start + prod(shape), offset + numel)
            if begin < end:
                slices.append((name, shape, begin - start, begin - offset, end - offset))
        return slices

    def anchor_parts(self, source: dict[str, torch.Tensor]) -> list[tuple[str, list[int], int, torch.Tensor]]:
        """This rank's ``(hf_name, shape, flat start, flat values)`` slices of a full HF checkpoint, from ``source``
        (key -> (global flat offset, flat wire bits of its shard))."""
        parts = []
        for key, (offset, bits) in source.items():
            values = bits.view(self.wire_dtypes[key])
            for name, shape, start, begin, end in self._slices(self.key_fqn[key], len(bits), offset):
                parts.append((name, shape, start, values[begin:end]))
        return parts

    @torch.no_grad()
    def write_full_checkpoint(
        self, model: nn.Module, is_master: bool, open_writer: Callable[[list], AnchorWriter], world_size: int
    ) -> None:
        """Write this rank's slices of a full HF checkpoint of the live weights, one tensor at a time, with the
        writer ``open_writer(slices)``."""
        shards = [
            (key, shard, offset, self._slices(fqn, shard.numel(), offset))
            for key, fqn, shard, offset, _ in self.local_shards(model, is_master)
        ]
        writer = open_writer(
            [
                (name, shape, start, end - begin, self.wire_dtypes[key])
                for key, _, _, slices in shards
                for name, shape, start, begin, end in slices
            ]
        )
        for key, shard, offset, slices in shards:
            values = shard.reshape(-1).to(self.wire_dtypes[key])
            writer.add_sums(*self._value_sums(key, offset, as_bits(values)))
            parts = [(name, shape, start, values[begin:end]) for name, shape, start, begin, end in slices]
            writer.write_parts(parts, self.stream_to_host)
        writer.close(world_size)

    @torch.no_grad()
    def encode(
        self,
        shards: list[tuple],
        ready: torch.cuda.Event,
        weights_read: threading.Event,
        emit: Callable,
        diffed: Callable[[int], None],
    ) -> None:
        """After ``wait_for_diffs``, on the side stream after ``ready``: diff the residual shards against their
        previous bits and advance them, and check the optimizer's tensors. Hands host parts ``(entries, gaps, value
        bytes, checksums)`` to ``emit`` as they are encoded, then calls ``diffed`` with the number of changed fp8
        block scales. At anchor steps, then writes the anchor slices the optimizer step did not write from the
        live weights. Sets ``weights_read`` once done reading the weights."""
        torch.cuda.set_device(self.device)
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                self._diff(shards, emit)
                self.stream.synchronize()
                diffed(self.scale_changes)
                if self.anchor is not None:
                    self._write_optimizer_anchor(shards)
                    self.stream.synchronize()
        finally:
            weights_read.set()

    def _write_optimizer_anchor(self, shards: list[tuple]) -> None:
        """The anchor slices of the optimizer's tensors that the step did not write (with full offload, all of
        them: writing them in the step stalls its gradient offload, and the gradients pile up on the GPU)."""
        for key, fqn, shard, offset, _ in shards:
            if key in self.residual or (key in self.stepped and not self.full_offload):
                continue
            values = shard.reshape(-1).to(self.wire_dtypes[key])
            self._write_anchor_slices(key, fqn, len(values), offset, values)

    def wait_for_diffs(self) -> None:
        for diff in self.diffs:
            diff.result()
        self.diffs.clear()
        get_logger().debug(
            f"Sparse weight delta in the optimizer step: waited {self.hook_wait:.2f}s for earlier diffs, diff thread "
            f"busy {self.diff_busy:.2f}s"
        )

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
        self.write_stream.wait_stream(torch.cuda.current_stream())
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
            # Through pinned memory: a copy from pageable memory waits for the stream, holding the GIL
            first = torch.tensor(firsts[i:j]).pin_memory().to(bits.device, non_blocking=True)
            sums.append((rows * positions).sum(1) + rows.sum(1, dtype=torch.int64) * first)
            i = j
        return torch.cat(sums)

    def _part(self, cache_key, names, begins, shifts, position, values, checksums) -> tuple:
        """A part entry for changes at flat ``position``s in segments that start at ``begins`` (sorted), where the HF
        flat index is the position plus the segment's shift: ``(names, counts, gaps, value bytes, checksum names,
        checksums)``."""
        if cache_key not in self.segment_starts:
            device = position.device
            self.segment_starts[cache_key] = (torch.tensor(begins, device=device), torch.tensor(shifts, device=device))
        starts, shift = self.segment_starts[cache_key]
        segment = torch.searchsorted(starts, position, right=True) - 1
        entry_starts = torch.ones_like(segment, dtype=torch.bool)
        entry_starts[1:] = segment[1:] != segment[:-1]
        gaps = encode_gaps(position + shift[segment], entry_starts)
        # Not bincount, which syncs to find the largest value
        counts = torch.zeros_like(starts).index_add_(0, segment, torch.ones_like(segment))
        return names, counts, gaps, values.view(torch.uint8), *checksums

    def _scale_part(self, fqn: str, first_block: int, previous: torch.Tensor, current: torch.Tensor) -> tuple:
        """The part entry of an fp8 tensor's changed block scales: block rows from ``first_block`` on."""
        scale_cols = current.shape[1]
        delta = as_bits(current).reshape(-1) - as_bits(previous).reshape(-1)
        torch.cuda.current_stream().synchronize()
        index = delta.nonzero().squeeze(1)
        self.scale_changes += len(index)
        offset = first_block * scale_cols
        names, begins, ends, shifts = self._segments((fqn, "scales"), self.scale_segments[fqn], offset, current.numel())
        sums = self._segment_sums(
            as_bits(current).reshape(-1), begins, ends, [begin + shift for begin, shift in zip(begins, shifts)]
        )
        segments = self.scale_segments[fqn]
        return self._part(
            (fqn, "scales"),
            [name for _, name in segments],
            [start for start, _ in segments],
            [-start for start, _ in segments],
            index + offset,
            zigzag(delta[index]),
            (names, sums),
        )

    def _diff(self, shards: list[tuple], emit: Callable) -> None:
        """Hand the optimizer step's remaining parts to ``emit``, then diff the residual ``shards`` on the current
        stream, handing a host part to ``emit`` every ``PART`` changes, and check that the optimizer's tensors hold
        what its step wrote."""
        pending, self.step_parts = self.step_parts, []
        num_pending = sum(len(part[2]) for part in pending)
        checks: list[tuple[str, torch.Tensor]] = []

        def flush() -> None:
            emit(self._host_part(pending))
            pending.clear()

        for key, fqn, shard, offset, gathered in shards:
            shard = shard.reshape(-1)
            if key in self.checksums:
                bits = as_bits(shard.to(self.wire_dtypes[key]))
                checks.append((key, self._checksum(key, offset, bits)))
                if key not in self.stepped:
                    # Not updated in this step (e.g. no gradient): an empty part carries its checksums (the bf16
                    # tensors' come with the last part)
                    pending += self._diff_shard(key, fqn, shard, offset, None, bits, None)
                continue
            if key in self.fast:
                continue
            parts = self._diff_shard(key, fqn, shard, offset, gathered, self.previous[key], self.previous[key])
            pending += parts
            num_pending += sum(len(part[2]) for part in parts)
            if num_pending >= self.PART:
                flush()
                num_pending = 0
        if pending:
            flush()
        names = [name for pieces in self.all_pieces for name in pieces.names]
        ids = [self.name_ids[name] for name in names]
        sums, live = self.sums[ids].tolist(), self._live_sums()[ids].tolist()
        self._emit_host_parts(emit, dict(zip(names, sums)))
        changed = {name for name, expected, value in zip(names, sums, live) if expected != value}
        if checks:
            keys, live = zip(*checks)
            expected = torch.cat([self.checksums[key] for key in keys])
            changed |= {key for key, same in zip(keys, expected.eq(torch.cat(live)).tolist()) if not same}
        if changed:
            raise RuntimeError(f"Trainer weights changed outside the optimizer step, not sent: {sorted(changed)}")

    def _host_part(self, pending: list[tuple], staging: torch.Tensor | None = None) -> tuple:
        """One part ``(entries, gaps, value bytes, checksums)`` on the host from GPU part entries."""
        names = [name for part in pending for name in part[0]]
        gaps = self._to_host(torch.cat([part[2] for part in pending]), staging=staging)
        values = self._to_host(torch.cat([part[3] for part in pending]), staging=staging)
        counts = torch.cat([part[1] for part in pending]).tolist()
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
        wire = self.wire_dtypes[key]
        segment_names, begins, ends, shifts = self._segments(key, self.segments[fqn], offset, shard.numel())
        if fqn not in self.fp8:
            current = as_bits(shard.to(wire))
            part = self._diff_flat(key, segment_names, begins, ends, shifts, current, previous, chunk)
            if out is not None:
                out.copy_(current)
            return [part]
        sums = torch.zeros(len(segment_names), dtype=torch.int64, device=shard.device)
        changed, changed_values, scales = [], [], []
        ue8m0, num_cols = self.fp8[fqn]
        if gathered is not None:
            chunks = [(0, shard.numel())]
        else:
            rows = max(FP8_BLOCK[0], chunk // num_cols // FP8_BLOCK[0] * FP8_BLOCK[0])
            chunks = [(start, start + rows * num_cols) for start in range(0, shard.numel(), rows * num_cols)]
        for start, end in chunks:
            end = min(end, shard.numel())
            current = as_bits(shard[start:end].to(wire))
            previous_bits = previous[start:end]
            if gathered is not None:
                previous_weights, current_weights = gathered
            else:
                previous_weights, current_weights = previous_bits.view(torch.bfloat16), current.view(torch.bfloat16)
            (previous_fp8, previous_scales), (current_fp8, current_scales) = (
                quantize_fp8_blocks(weights.reshape(-1, num_cols), ue8m0, self.compiled_fp8)
                for weights in (previous_weights, current_weights)
            )
            scales.append((previous_scales, current_scales))
            if gathered is not None:
                previous_fp8 = previous_fp8.view(-1)[offset : offset + shard.numel()]
                current_fp8 = current_fp8.view(-1)[offset : offset + shard.numel()]
            delta = current_fp8.view(-1) - previous_fp8.view(-1)
            if out is not None:
                out[start:end].copy_(current, non_blocking=True)
            # nonzero syncs to size its output, holding the GIL: wait for the stream first, without it
            torch.cuda.current_stream().synchronize()
            index = delta.nonzero().squeeze(1)
            changed.append(index + start)
            changed_values.append(zigzag(delta[index]))
            sums += self._window_sums(current_fp8.view(-1), start, end, begins, ends, shifts)
        parts = [
            self._part(
                key, segment_names, begins, shifts, torch.cat(changed), torch.cat(changed_values), (segment_names, sums)
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

    def _diff_flat(self, cache_key, names, begins, ends, shifts, current, previous, chunk: int) -> tuple:
        """The part entry of the changes between the flat bits ``previous`` and ``current``, made of segments
        ``[begin, end)`` whose HF flat index is the position plus the shift, in chunks of ``chunk`` elements."""
        sums = torch.zeros(len(names), dtype=torch.int64, device=current.device)
        changed, changed_values = [], []
        for start in range(0, current.numel(), chunk):
            end = min(start + chunk, current.numel())
            delta = current[start:end] - previous[start:end]
            # nonzero syncs to size its output, holding the GIL: wait for the stream first, without it
            torch.cuda.current_stream().synchronize()
            index = delta.nonzero().squeeze(1)
            changed.append(index + start)
            changed_values.append(zigzag(delta[index]))
            sums += self._window_sums(current[start:end], start, end, begins, ends, shifts)
        return self._part(
            cache_key, names, begins, shifts, torch.cat(changed), torch.cat(changed_values), (names, sums)
        )

    def _window_sums(self, bits, start: int, end: int, begins, ends, shifts) -> torch.Tensor:
        """Every segment's ``bits_checksum`` contribution from the window ``[start, end)``, whose flat bits are
        ``bits``."""
        sums = torch.zeros(len(begins), dtype=torch.int64, device=bits.device)
        first, last = bisect_right(ends, start), bisect_left(begins, end)
        if first < last:
            sums[first:last] = self._segment_sums(
                bits,
                [max(begin, start) - start for begin in begins[first:last]],
                [min(stop, end) - start for stop in ends[first:last]],
                [max(begin, start) + shift for begin, shift in zip(begins[first:last], shifts[first:last])],
            )
        return sums

    def _value_sums(self, key: str, offset: int, bits: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """The ``bits_checksum`` of each HF tensor in a key's flat shard bits, in chunks of ``HOOK_CHUNK``
        elements."""
        names, begins, ends, shifts = self._segments(key, self.segments[self.key_fqn[key]], offset, bits.numel())
        sums = torch.zeros(len(names), dtype=torch.int64, device=bits.device)
        for start in range(0, bits.numel(), self.HOOK_CHUNK):
            end = min(start + self.HOOK_CHUNK, bits.numel())
            sums += self._window_sums(bits[start:end], start, end, begins, ends, shifts)
        return names, sums

    def _checksum(self, key: str, offset: int, bits: torch.Tensor) -> torch.Tensor:
        """The sum of the ``bits_checksum``s of the HF tensors in a key's flat shard bits: what the optimizer step
        records for the tensors it writes."""
        return self._value_sums(key, offset, bits)[1].sum(0, keepdim=True)

    def bind(self, model: nn.Module, is_master: bool, params: list[torch.Tensor], full_offload: bool) -> None:
        """Diff the trainer tensors stored in ``params`` (the optimizer's local parameter shards, contiguous) in its
        step; the others are the residual. A trainer tensor may be a strided view of a parameter (e.g. one half of
        a packed gate/up weight). fp8 tensors whose blocks straddle ranks need collectives and stay residual."""
        assert all(param.is_contiguous() for param in params)
        stored = {param.untyped_storage().data_ptr() for param in params}
        residual = []
        for key, fqn, tensor, offset, _ in self.local_shards(model, is_master):
            storage = tensor.untyped_storage().data_ptr()
            if key not in self.gather and storage in stored:
                view = (tuple(tensor.shape), tensor.stride(), tensor.storage_offset())
                self.param_keys[storage].append((key, fqn, tensor.numel(), offset, view))
            else:
                residual.append(key)
        self.residual = set(residual)
        # Diffed by ``sparse_kernels``: the tensors of parameters that are all bf16 on the wire. The others (fp8,
        # fp32) by ``_diff_slot``
        self.fast = {
            key
            for entries in self.param_keys.values()
            if all(self.wire_dtypes[key] == torch.bfloat16 and fqn not in self.fp8 for key, fqn, *_ in entries)
            for key, *_ in entries
        }
        self.params, self.full_offload = params, full_offload
        self.pieces: dict[tuple, Pieces] = {}
        self.name_ids = {name: index for index, name in enumerate(self.hf_tensors)}
        device = f"cuda:{self.device}"
        # The bf16 tensors' checksums of their last written bits, by ``name_ids``
        self.sums = torch.zeros(len(self.name_ids), dtype=torch.int64, device=device)
        # Old values of the updates in flight (full offload: one parameter each; grown as needed otherwise), and
        # the pinned regions the kernels write the changes to
        slot_bytes = -(-max(param.nbytes for param in params) // 256) * 256 if full_offload else 0
        self.slots, self.regions = queue.SimpleQueue(), queue.SimpleQueue()
        for _ in range(self.HOOK_SLOTS):
            self.slots.put(torch.empty(slot_bytes, dtype=torch.uint8, device=device))
            self.regions.put(
                (
                    torch.empty(self.REGION, dtype=torch.int32, pin_memory=True),
                    torch.empty(self.REGION, dtype=torch.int16, pin_memory=True),
                    torch.empty(0, dtype=torch.int32, pin_memory=True),
                )
            )
        self.update_bytes = threading.Condition()
        self.held_bytes = 0
        self.differ = ThreadPoolExecutor(1)
        self.diff_stream = torch.cuda.Stream(self.device)
        self.diff_staging = torch.empty(1 << 28, dtype=torch.uint8, pin_memory=True)
        self.staging = torch.empty(1 << 28, dtype=torch.uint8, pin_memory=True)

    @torch.no_grad()
    def reset(self, model: nn.Module, is_master: bool) -> None:
        """After a full checkpoint: take the residual's previous bits and the optimizer tensors' checksums from the
        live weights, on the training thread."""
        self.previous, self.checksums, self.shards = {}, {}, {}
        for key, _, shard, offset, _ in self.local_shards(model, is_master):
            if key in self.residual:
                self.previous[key] = self._bits(key, shard)
                self.shards[key] = (shard.numel(), offset)
            elif key not in self.fast:
                self.checksums[key] = self._checksum(key, offset, as_bits(shard.reshape(-1).to(self.wire_dtypes[key])))
        self.all_pieces = self._group_pieces(
            [(param, [entry for entry in self._entries(param) if entry[0] in self.fast], 0) for param in self.params]
        )
        self.sums = self._live_sums()
        num_bytes = sum(bits.nbytes for bits in self.previous.values())
        get_logger().info(
            f"Sparse weight deltas of {len(self.checksums) + len(self.fast)} tensors come from the optimizer step, "
            f"{len(self.previous)} tensors ({num_bytes / 2**30:.2f} GiB) are diffed against a copy"
        )

    def _live_sums(self) -> torch.Tensor:
        """The checksums of the bf16 optimizer tensors' live bits, by ``name_ids``."""
        sums = torch.zeros_like(self.sums)
        for pieces in self.all_pieces:
            count_changes(pieces, None, torch.empty_like(pieces.tables[5]), sums)
        return sums

    def arm(self, anchor: AnchorWriter | None, emit: Callable) -> None:
        """Collect the next optimizer step's delta, handing host parts to ``emit`` as the step goes, and write its
        anchor slices if ``anchor``."""
        self.step_parts, self.scale_changes, self.anchor, self.emit = [], 0, anchor, emit
        self.host_parts: list[tuple] = []
        # Optimizer tensors the step updated (and wrote the anchor slices of)
        self.stepped: set[str] = set()
        self.hook_wait = self.diff_busy = 0.0
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def anchor_slices(self) -> list[tuple[str, list[int], int, int, torch.dtype]]:
        """Every ``(hf_name, shape, flat start, numel, wire dtype)`` slice of this rank's share of an anchor."""
        shards = {
            key: (fqn, numel, offset) for entries in self.param_keys.values() for key, fqn, numel, offset, _ in entries
        }
        shards.update({key: (self.key_fqn[key], numel, offset) for key, (numel, offset) in self.shards.items()})
        return [
            (name, shape, start, end - begin, self.wire_dtypes[key])
            for key, (fqn, numel, offset) in shards.items()
            for name, shape, start, begin, end in self._slices(fqn, numel, offset)
        ]

    def _entries(self, local_param: torch.Tensor) -> list[tuple]:
        """The ``param_keys`` entries of the trainer tensors in ``local_param``, views relative to it."""
        start = local_param.storage_offset()
        return [
            (key, fqn, numel, offset, (size, stride, at - start))
            for key, fqn, numel, offset, (size, stride, at) in self.param_keys.get(
                local_param.untyped_storage().data_ptr(), []
            )
            if start <= at < start + local_param.numel()
        ]

    def _group_pieces(self, items: list[tuple]) -> list[Pieces]:
        """The ``Pieces`` of ``items`` (``(local parameter, entries, position of its old bits)``), one per
        parameter dtype; cached."""
        cache_key = tuple((local.data_ptr(), position) for local, _, position in items)
        if cache_key not in self.pieces:
            rows = defaultdict(list)
            for local, entries, position in items:
                for key, fqn, numel, offset, (size, stride, at) in entries:
                    names, begins, ends, shifts = self._segments(key, self.segments[fqn], offset, numel)
                    for name, begin, end, shift in zip(names, begins, ends, shifts):
                        # A segment is contiguous in the parameter, also in a strided view
                        first, last = (
                            at + sum(i * s for i, s in zip(np.unravel_index(j, size), stride)) for j in (begin, end - 1)
                        )
                        assert last - first == end - 1 - begin
                        rows[local.dtype].append(
                            (
                                name,
                                position + first,
                                local.data_ptr() + first * local.element_size(),
                                end - begin,
                                begin + shift,
                                self.name_ids[name],
                            )
                        )
            self.pieces[cache_key] = [
                build_pieces(group, dtype == torch.float32, local.device) for dtype, group in rows.items()
            ]
        return self.pieces[cache_key]

    def _launch(self, items: list[tuple], old: torch.Tensor, release: Callable, written: list[tuple]) -> None:
        """Right after the update of ``items`` (``(local parameter, entries, position of its old bits in old``),
        on the current stream: count and write its changes to pinned regions and record the new bits' checksums;
        the diff thread collects them, writes the anchor slices of ``written`` and then calls ``release``."""
        launches = []
        for pieces in self._group_pieces(items):
            indices, values, counts_host = self.regions.get()
            num_blocks = len(pieces.tables[5])
            if len(counts_host) < num_blocks:
                counts_host = torch.empty(num_blocks, dtype=torch.int32, pin_memory=True)
            counts = torch.empty(num_blocks, dtype=torch.int32, device=old.device)
            self.sums.index_fill_(0, pieces.segments, 0)
            count_changes(pieces, old, counts, self.sums)
            ends = counts.cumsum(0)
            write_changes(pieces, old, ends, indices, values, 0)
            counts_host[:num_blocks].copy_(counts, non_blocking=True)
            launches.append((pieces, ends, (indices, values, counts_host)))
        ready = torch.cuda.Event()
        ready.record()
        keys = [key for _, entries, _ in items for key, *_ in entries]
        self.diffs.append(self.differ.submit(self._collect, launches, keys, old, ready, release, written))

    @torch.no_grad()
    def _collect(self, launches: list[tuple], keys: list[str], old: torch.Tensor, ready, release, written) -> None:
        """Diff thread: the host parts of ``_launch``'s changes to ``keys`` (writing the changes past a region's
        capacity in further rounds); then ``release`` the old values and the regions, and write the anchor slices
        of ``written``."""
        torch.cuda.set_device(self.device)
        ready.synchronize()
        start = time.perf_counter()
        try:
            with torch.cuda.stream(self.diff_stream):
                for pieces, ends, (indices, values, counts) in launches:
                    block_counts = counts[: len(pieces.tables[5])].numpy()
                    total = int(block_counts.sum())
                    changed, changed_values = [], []
                    for skip in range(0, total, len(indices)):
                        if skip:
                            write_changes(pieces, old, ends, indices, values, skip)
                            self.diff_stream.synchronize()
                        num = min(len(indices), total - skip)
                        changed.append(indices[:num].numpy().copy())
                        changed_values.append(values[:num].numpy().copy())
                    if not total:
                        continue
                    piece_counts = np.add.reduceat(block_counts, pieces.first_blocks)
                    index = np.concatenate(changed)
                    previous = np.empty_like(index)
                    previous[1:] = index[:-1]
                    previous[(np.cumsum(piece_counts) - piece_counts)[piece_counts > 0]] = -1
                    entries = [(name, int(count)) for name, count in zip(pieces.names, piece_counts) if count]
                    self.host_parts.append((entries, index - previous - 1, np.concatenate(changed_values)))
        finally:
            # Before the anchor writes, which read the live weights: the next updates wait for these
            for _, _, region in launches:
                self.regions.put(region)
            release()
        try:
            if self.anchor is not None:
                with torch.cuda.stream(self.diff_stream):
                    for key, fqn, numel, offset, view in written:
                        values = view.reshape(-1).to(self.wire_dtypes[key])
                        self._write_anchor_slices(key, fqn, numel, offset, values)
                    self.diff_stream.synchronize()
            self.stepped.update(keys)
        finally:
            self.diff_busy += time.perf_counter() - start
        if sum(len(part[1]) for part in self.host_parts) >= self.HOOK_PART:
            self._emit_host_parts(self.emit)

    def _emit_host_parts(self, emit: Callable, checksums: dict[str, int] | None = None) -> None:
        parts, self.host_parts = self.host_parts, []
        gaps = np.concatenate([part[1] for part in parts]) if parts else np.empty(0, np.int32)
        values = np.concatenate([part[2] for part in parts]) if parts else np.empty(0, np.int16)
        entries = [entry for part in parts for entry in part[0]]
        emit((entries, torch.from_numpy(gaps), torch.from_numpy(values.view(np.uint8)), checksums or {}))

    @torch.no_grad()
    def update_hook(self, params: list[torch.Tensor]) -> Callable[[], None]:
        """``update_hook`` of the optimizers on the GPU, right before they update ``params`` on the current stream:
        copy the old wire bits of their shards, and diff them against the updated shards (bf16: right after the
        update, with ``sparse_kernels``; others on the diff thread)."""
        if not self.armed:
            return lambda: None
        fast, other = [], []
        for param in params:
            local = param.to_local() if isinstance(param, DTensor) else param
            entries = self._entries(local)
            if entries:
                (fast if entries[0][0] in self.fast else other).append((local, entries))
        start = time.perf_counter()
        num_bytes = sum(local.numel() for local, _ in other) * 2
        if other:
            with self.update_bytes:
                self.update_bytes.wait_for(
                    lambda: self.held_bytes == 0 or self.held_bytes + num_bytes <= self.UPDATE_BYTES
                )
                self.held_bytes += num_bytes
        if fast:
            slot = self.slots.get()
            numel = sum(local.numel() for local, _ in fast)
            if slot.numel() < 2 * numel:
                slot = torch.empty(2 * numel, dtype=torch.uint8, device=slot.device)
            old, items, written, position = slot.view(torch.bfloat16), [], [], 0
            for local, entries in fast:
                old[position : position + local.numel()].view(local.shape).copy_(local)
                items.append((local, entries, position))
                written += [
                    (key, fqn, numel, offset, local.as_strided(size, stride, local.storage_offset() + at))
                    for key, fqn, numel, offset, (size, stride, at) in entries
                ]
                position += local.numel()
        self.hook_wait += time.perf_counter() - start
        items_other = []
        for local, entries in other:
            dtypes = {self.wire_dtypes[key] for key, *_ in entries}
            dtype = dtypes.pop() if len(dtypes) == 1 else local.dtype
            items_other.append((entries, local.to(dtype, copy=True), local))

        def release() -> None:
            with self.update_bytes:
                self.held_bytes -= num_bytes
                self.update_bytes.notify_all()

        def finish() -> None:
            if fast:
                self._launch(items, slot.view(torch.int16), partial(self.slots.put, slot), written)
            if other:
                ready = torch.cuda.Event()
                ready.record()
                self.diffs.append(self.differ.submit(self._diff_slot, items_other, ready, release, True))

        return finish

    @torch.no_grad()
    def optimizer_hook(self, local_param: torch.Tensor, source: torch.Tensor) -> None:
        """``FullCPUOffloadOptimizer.weight_hook``, on its H2D stream: copy the shard (still the previous
        broadcast's weights) into a GPU slot, write the new values ``source`` (pinned host) and diff the shard
        against the slot (bf16: right away, with ``sparse_kernels``; others on the diff thread)."""
        entries = self._entries(local_param) if self.armed else []
        if not entries:
            local_param.copy_(source, non_blocking=True)
            return
        slot = self.slots.get()
        fast = entries[0][0] in self.fast
        previous = slot[: local_param.numel() * (2 if fast else local_param.element_size())]
        previous = previous.view(torch.bfloat16 if fast else local_param.dtype)
        previous.view(local_param.shape).copy_(local_param)
        local_param.copy_(source, non_blocking=True)
        release = partial(self.slots.put, slot)
        if fast:
            self._launch([(local_param, entries, 0)], previous.view(torch.int16), release, [])
            return
        ready = torch.cuda.Event()
        ready.record()
        # The shard holds the new values until the next step, which waits for the diffs
        items = [(entries, previous, local_param.reshape(-1))]
        self.diffs.append(self.differ.submit(self._diff_slot, items, ready, release, False))

    @torch.no_grad()
    def _diff_slot(self, items: list[tuple], ready: torch.cuda.Event, release: Callable, on_gpu: bool) -> None:
        """Diff thread: the parts of the trainer tensors in ``items`` (``(entries, old values, new values)`` of
        parameter shards that are not all bf16), and their anchor slices if the optimizer is ``on_gpu``; then
        ``release`` the old values. Hands the parts so far to ``emit`` every ``HOOK_PART`` changes."""
        torch.cuda.set_device(self.device)
        ready.synchronize()
        start = time.perf_counter()
        chunk = self.CHUNK if on_gpu else self.HOOK_CHUNK
        try:
            with torch.cuda.stream(self.diff_stream):
                for entries, previous, new in items:
                    for key, fqn, numel, offset, (size, stride, at) in entries:
                        current, old = (
                            values.as_strided(size, stride, values.storage_offset() + at) for values in (new, previous)
                        )
                        current = current.reshape(-1).to(self.wire_dtypes[key])
                        old = as_bits(old.reshape(-1).to(self.wire_dtypes[key]))
                        self.step_parts += self._diff_shard(key, fqn, current, offset, None, old, None, chunk)
                        self.checksums[key] = self._checksum(key, offset, as_bits(current))
                        if self.anchor is not None and on_gpu:
                            self._write_anchor_slices(key, fqn, numel, offset, current)
                        self.stepped.add(key)
                self.diff_stream.synchronize()
        finally:
            release()
            self.diff_busy += time.perf_counter() - start
        if sum(len(part[2]) for part in self.step_parts) >= self.HOOK_PART:
            with torch.cuda.stream(self.diff_stream):
                self.emit(self._host_part(self.step_parts, self.diff_staging))
            self.step_parts = []

    def _write_anchor_slices(self, key: str, fqn: str, numel: int, offset: int, values: torch.Tensor) -> None:
        """Write a GPU shard's anchor slices, and their checksums, through the pinned staging buffer: the slices in
        each staged window in parallel, in pieces of up to 32 MiB."""
        self.anchor.add_sums(*self._value_sums(key, offset, as_bits(values)))
        data, itemsize = values.view(torch.uint8), values.element_size()
        slices = [
            (name, start, begin * itemsize, end * itemsize)
            for name, _, start, begin, end in self._slices(fqn, numel, offset)
        ]
        window, piece = self.diff_staging.numel(), 1 << 25
        for at in range(0, data.numel(), window):
            count = min(window, data.numel() - at)
            self.diff_staging[:count].copy_(data[at : at + count], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            staged = self.diff_staging[:count].numpy()
            items = []
            for name, start, begin, end in slices:
                for first in range(max(begin, at), min(end, at + count), piece):
                    last = min(first + piece, end, at + count)
                    items.append((name, start, staged[first - at : last - at], first - begin))
            self.anchor.write_all(items)


def _write_json(path: StoragePath, value) -> None:
    write_file(path, json.dumps(value).encode())


class SparseFileSystemWeightSender(FileSystemWeightSender):
    """Filesystem broadcast where steps carry only the delta against the previous broadcast (format in
    ``prime_rl.utils.sparse_weights``). The delta comes from the optimizer step (``diff_in_optimizer``) and is
    written as the step goes; the rest is finished in the background while the next step trains, and the trainer
    waits only before its next final backward, until that work has read the weights. A broadcast that does not
    follow an optimizer step collected this way (the first one, or any broadcast without ``diff_in_optimizer``)
    publishes a full checkpoint. Anchors (full checkpoints every ``anchor_interval`` steps) serve engines that fall
    off the delta chain. The writer thread commits a step once every rank's files are on disk."""

    gathers_full_weights = False

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
        self.errors: list[BaseException] = []
        self.weights_read: threading.Event | None = None
        self.part_bytes = 0
        self.storage = storage_path(config.storage) if config.storage else get_broadcast_dir(output_dir)
        if config.storage and self.world.is_master:
            # Leftovers of an earlier run would look like anchors and deltas of this one. A new run's engines report
            # their layout again; a resumed trainer (its broadcast dir has steps) keeps the running engines' one.
            stale = list(self.storage.glob("step_*"))
            if not any(get_broadcast_dir(output_dir).glob("step_*")):
                stale.append(self.storage / ENGINE_LAYOUT)
            for path in stale:
                remove(path)
        self.optimizer: "OptimizerLike | None" = None
        self.model: nn.Module | None = None
        # The step whose delta the next optimizer step collects
        self.armed_step: int | None = None
        self.streamed_dir: Path | None = None
        self.num_streamed = 0
        # The delta dir of the step being encoded, where a failed encode or write leaves this rank's failure marker
        self.delta_dir: StoragePath | None = None

    def diff_in_optimizer(self, optimizer: "OptimizerLike") -> None:
        """Take the deltas of the optimizer's parameters from its steps, which must each be followed by a broadcast."""
        self.optimizer = optimizer

    def _bind(self) -> None:
        """Hook the encoder into the optimizer's parameter updates."""
        from prime_rl.trainer.optim import FullCPUOffloadOptimizer, OffloadOptimizer

        optimizer, encoder = self.optimizer, self.encoder
        if isinstance(optimizer, FullCPUOffloadOptimizer):
            params = [master.model_param.to_local() for master in optimizer._master_weights.values()]
            encoder.bind(self.model, self.world.is_master, params, full_offload=True)
            optimizer.weight_hook = encoder.optimizer_hook
            return
        base = optimizer.base_optimizer if isinstance(optimizer, OffloadOptimizer) else optimizer
        if not hasattr(base, "update_hook"):
            raise NotImplementedError(f"Sparse weight broadcast cannot hook into {type(base).__name__} updates")
        params = [p.to_local() if isinstance(p, DTensor) else p for group in base.param_groups for p in group["params"]]
        encoder.bind(self.model, self.world.is_master, params, full_offload=False)
        base.update_hook = encoder.update_hook

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
        anchor = self._anchor_writer(step, encoder.anchor_slices()) if step % self.config.anchor_interval == 0 else None
        step_dir = get_step_path(self.storage, step)
        # Parts written during the step go next to the broadcast step dir, which the broadcast handshake recreates,
        # and are moved in when it is published; on separate storage straight into the step dir
        if self.config.storage:
            self.streamed_dir = step_dir / DELTA_DIR
        else:
            self.streamed_dir = step_dir.with_name(f".{step_dir.name}.delta{self.world.rank}")
        self.streamed_dir.mkdir(parents=True, exist_ok=True)
        self.num_streamed = 0
        encoder.arm(anchor, self._stream_part)
        self.armed_step = step

    def _anchor_writer(self, step: int, slices: list) -> AnchorWriter:
        """Written next to the step dir and moved in once complete; for S3, next to the broadcast step dir and
        uploaded."""
        local_dir = None
        if isinstance(self.storage, S3Path):
            local_dir = self.step_dir(step).with_name(f".{self.step_dir(step).name}.anchor{self.world.rank}")
        return AnchorWriter(slices, get_step_path(self.storage, step), self.world.rank, local_dir)

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
                if pool is not self.anchors and self.delta_dir is not None:
                    # Engines reading the delta fail instead of waiting for parts that never come
                    _write_json(self.delta_dir / rank_failed_marker(self.world.rank), {"error": repr(error)})
                raise

        return pool.submit(run)

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        self._wait_for_encode()
        step_dir = get_step_path(self.storage, step)
        # Every rank: on a separate storage dir, no handshake creates the step dir before the others write to it
        (step_dir / DELTA_DIR).mkdir(parents=True, exist_ok=True)
        if self.encoder is None:
            self.encoder = SparseDeltaEncoder(model)
            self.model = model
        if self.armed_step is not None and self.armed_step != step:
            raise RuntimeError(f"Sparse weight delta v{self.armed_step} was collected, but v{step} is broadcast")
        if self.armed_step is None:
            self._broadcast_full(model, step, step_dir)
            return
        encoder = self.encoder
        encoder.disarm()
        self.armed_step = None
        base_step, self.last_step = self.last_step, step
        shards = encoder.local_shards(model, self.world.is_master, gather=True)
        ready = torch.cuda.Event()
        ready.record()
        self.weights_read = threading.Event()
        args = (step, base_step, step_dir, shards, ready, self.weights_read, encoder.anchor)
        self._submit(self.encodes, self._encode, *args)

    def _broadcast_full(self, model: nn.Module, step: int, step_dir: StoragePath) -> None:
        """Publish v{step} as a full checkpoint of the live weights; the run's first broadcast at step 0 is the
        checkpoint the engines booted from. The next delta is against it."""
        if self.last_step is None and self.world.is_master:
            # Engines report their layout again as they load it; the last run's may differ
            shutil.rmtree(step_dir.parent / ENGINE_LAYOUT, ignore_errors=True)
        if self.last_step is None and step == 0:
            self.anchor_step = step
            if self.world.is_master:
                self._write_manifest(step_dir, step, None, 0)
                WeightSender._finish(self, step, self.step_dir(step))
        else:
            self.encoder.write_full_checkpoint(
                model, self.world.is_master, partial(self._anchor_writer, step), self.world.world_size
            )
            self._wait_for_anchor(step, step_dir)
            if self.world.is_master:
                WeightSender._finish(self, step, self.step_dir(step))
        self.last_step = step
        if self.optimizer is not None:
            # Engines report their layout as they load this broadcast. Take it (and compile the fp8 quantizer) now,
            # not in the first optimizer step, next to its activations.
            self._set_fp8_layout()
            if self.encoder.differ is None:
                self._bind()
            self.encoder.reset(model, self.world.is_master)

    def _wait_for_engine_layout(self, broadcast_dir: StoragePath) -> dict[str, str]:
        start = time.monotonic()
        while (layout := read_engine_layout(broadcast_dir)) is None:
            if time.monotonic() - start > self.timeout:
                raise TimeoutError(f"Not every engine worker reported its weight layout within {self.timeout}s")
            time.sleep(0.1)
        return layout

    def _encode(
        self,
        step: int,
        base_step: int,
        step_dir: StoragePath,
        shards,
        ready,
        weights_read: threading.Event,
        anchor: AnchorWriter | None,
    ) -> None:
        """Encode thread: finish v{step}'s delta, and the anchor the optimizer step wrote."""
        encoder = self.encoder
        start = time.perf_counter()
        num_parts = 0
        self.delta_dir = step_dir / DELTA_DIR

        def emit(part: tuple) -> None:
            nonlocal num_parts
            path = step_dir / DELTA_DIR / rank_part(self.world.rank, num_parts)
            self._submit(self.writes, self._write_part, path, part)
            num_parts += 1

        try:
            encoder.wait_for_diffs()
            num_parts = self.num_streamed
            if self.world.is_master:
                self._submit(self.writes, self._write_manifest, step_dir, step, base_step, self.world.world_size)
            if not self.config.storage:
                self._submit(self.writes, self._move_streamed, self.streamed_dir, step_dir / DELTA_DIR)

            def diffed(scale_changes: int) -> None:
                args = (step, base_step, step_dir, num_parts, start, time.perf_counter() - start, scale_changes)
                self._submit(self.writes, self._finish_delta, *args)

            encoder.encode(shards, ready, weights_read, emit, diffed)
        except BaseException as error:
            # Recorded before the trainer wakes up, so it fails at its next wait instead of training on
            self.errors.append(error)
            weights_read.set()
            raise
        # Free the gathered fp8 tensors
        shards.clear()
        if anchor is not None:
            # The residual's previous bits hold v{step} until the next encode, which runs on this thread
            source = {key: (self.encoder.shards[key][1], bits) for key, bits in encoder.previous.items()}
            for key, (offset, bits) in source.items():
                anchor.add_sums(*encoder._value_sums(key, offset, bits))
            anchor.write_parts(encoder.anchor_parts(source), encoder.stream_to_host)
            self._submit(self.anchors, self._close_anchor, step, step_dir, anchor)

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

    def _close_anchor(self, step: int, step_dir: StoragePath, anchor: AnchorWriter) -> None:
        anchor.close(self.world.world_size)
        self._wait_for_anchor(step, step_dir)

    def _wait_for_anchor(self, step: int, step_dir: StoragePath) -> None:
        if self.world.is_master:
            while not anchor_complete(step_dir):
                time.sleep(0.1)
            self.anchor_step = max(self.anchor_step or 0, step)

    def _finish(self, step: int, step_dir: Path) -> None:
        """Committed in ``_broadcast_full`` or in the background (``_finish_delta``)."""

    def _clean(self, step: int) -> None:
        # Engines replay from the last anchor, so keep everything since it
        step = step if self.anchor_step is None else min(step, self.anchor_step + 1)
        super()._clean(step)
        if self.config.storage:  # the base class cleans the broadcast dir
            for path in self.storage.glob("step_*"):
                if int(path.name.removeprefix("step_")) < step - 1:
                    remove(path)
