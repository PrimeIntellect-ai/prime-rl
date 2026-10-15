import json
import threading
import time
from collections import defaultdict
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
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.clients import load_lora_adapter
from prime_rl.trainer.lora import get_lora_state, save_lora_config
from prime_rl.transports.weights.base import FINISHED_MARKER, WeightReceiver, WeightSender
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import wait_for_path
from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    FP8_BLOCK,
    MANIFEST,
    anchor_complete,
    as_bits,
    bits_dtype,
    encode_gaps,
    quantize_fp8_blocks,
    read_engine_layout,
    write_anchor_shards,
    write_rank_file,
)
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    resolve_fqn,
    resolve_wire_dtype,
    save_state_dict,
    save_state_dict_parallel,
)


class SparseDeltaEncoder:
    """Rank-local XOR deltas of the trainer's weight shards, in HF coordinates.

    Every rank keeps the wire-dtype bits of its own shards from the previous
    broadcast (the snapshot) and diffs against them, so no weights are gathered.
    The prime->HF conversion is replayed once on meta tensors: every HF tensor
    must come out as a contiguous view of one trainer tensor (renames, expert
    unstacking), which maps a changed trainer element to its HF name and index.

    ``encode`` runs on a background thread and a side CUDA stream while the
    next step trains, without collectives; it only reads the weights before
    ``weights_read`` is set. The snapshot lives on the GPU when it fits next to
    the first training step's peak memory, else in one host buffer. With a GPU
    snapshot, the same host buffer holds anchor copies instead, so a rank has at
    most one shard-sized host copy.

    Tensors the engines hold in fp8 (``set_fp8_layout``: HF name -> scale format,
    from the engine layout) are quantized with vLLM's own per-block quantizer,
    previous and current weights alike, and diffed as fp8 bytes. Their 128-row
    blocks must not straddle trainer ranks, else the trainer tensor is gathered.
    They are diffed on the training thread (``prepare``). A step that changes
    any block scale is published as a full checkpoint.
    """

    # Elements per transient chunk of the diff (bf16: 128 MiB per temporary)
    CHUNK = 1 << 26

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
        self.hf_numel: dict[str, int] = {}
        for name, tensor in convert_state_dict_to_hf(model, dict(meta)).items():
            base = tensor if tensor._base is None else tensor._base
            if id(base) not in owner or not tensor.is_contiguous():
                raise NotImplementedError(f"Sparse weight broadcast cannot map {name} to a contiguous trainer slice")
            if tensor.numel() >= 2**32:
                raise NotImplementedError(f"Sparse weight broadcast indexes tensors below 2^32 elements ({name})")
            segments[owner[id(base)]].append((tensor.storage_offset(), name))
            self.hf_tensors[name] = (list(tensor.shape), str(tensor.dtype).removeprefix("torch."))
            self.hf_numel[name] = tensor.numel()
        self.segments = {fqn: sorted(segs) for fqn, segs in segments.items()}
        self.numel = {fqn: tensor.numel() for fqn, tensor in meta.items()}
        # fqn -> (ue8m0 scales, columns) for trainer tensors held in fp8; quantized as one [rows, columns] matrix
        self.fp8: dict[str, tuple[bool, int]] = {}
        self.fp8_layout: dict[str, str] | None = None
        self.gather: set[str] = set()
        self.segment_starts: dict[str, torch.Tensor] = {}
        self.snapshot_placement = snapshot
        # key -> (local numel, global flat offset) of this rank's shard
        self.shards: dict[str, tuple[int, int]] = {}
        # key -> previous wire bits: GPU tensors, or views into the host buffer
        self.snapshot: dict[str, torch.Tensor] = {}
        self.previous: dict[str, torch.Tensor] = {}
        self.fp8_part: list | None = None
        self.on_gpu: bool | None = None  # None until placed (auto: at the first delta)
        self.host_buffer: torch.Tensor | None = None
        self.host: dict[str, torch.Tensor] = {}
        self.pinning: threading.Thread | None = None
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
                or any(len(shape) != 2 or shape[1] != num_cols for shape in shapes)
                or any(row % FP8_BLOCK[0] and row != num_rows for row in boundaries)
            ):
                raise NotImplementedError(f"Sparse weight broadcast cannot quantize {fqn} to fp8 as one matrix")
            self.fp8[fqn] = (modes.pop() == "ue8m0", num_cols)
            for _, name in segs:
                self.hf_tensors[name] = (self.hf_tensors[name][0], "float8_e4m3fn")
        self.gather = self._find_gathered(model, is_master)

    def _local_shard(self, key: str, value: torch.Tensor, is_master: bool) -> tuple[torch.Tensor, int] | None:
        """This rank's contiguous flat slice of ``value`` and its global flat offset,
        or None when another rank owns the elements."""
        if not isinstance(value, DTensor):
            return (value.detach().reshape(-1), 0) if is_master else None
        if all(placement.is_replicate() for placement in value.placements):
            return (value.to_local().detach().reshape(-1), 0) if is_master else None
        local_shape, global_offset = compute_local_shape_and_global_offset(
            value.shape, value.device_mesh, value.placements
        )
        if tuple(local_shape[1:]) != tuple(value.shape[1:]) or any(global_offset[1:]):
            raise NotImplementedError(f"Sparse weight broadcast needs dim-0 shards ({key}: {value.placements})")
        local = value.to_local().detach()[: local_shape[0]]
        return local.reshape(-1), global_offset[0] * prod(value.shape[1:])

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
        """This rank's ``(key, fqn, flat shard, global flat offset, gathered)`` for ``encode``, taken on the
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

    def _host_ready(self) -> None:
        """Wait for the page-locking thread: no CUDA copy may touch the buffer while it runs."""
        if self.pinning is not None:
            start = time.perf_counter()
            self.pinning.join()
            self.pinning, self.pinned = None, True
            get_logger().info(
                f"Page-locked the {self.host_buffer.numel() / 2**30:.1f} GiB sparse weight host buffer in "
                f"{time.perf_counter() - self.pin_start:.1f}s (waited {time.perf_counter() - start:.1f}s)"
            )

    def _pin_host(self) -> None:
        if self.host_buffer is None or self.pinned or self.pinning is not None:
            return
        buffer = self.host_buffer

        def register() -> None:
            torch.cuda.set_device(self.device)
            error = torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), buffer.numel(), 0)
            assert error == 0, f"cudaHostRegister failed: {error}"

        self.pin_start = time.perf_counter()
        self.pinning = threading.Thread(target=register)
        self.pinning.start()

    def _place(self, peak_reserved: int) -> None:
        """Keep the snapshot on the GPU if it fits next to the first training step's peak, else on the host."""
        num_bytes = sum(numel * self.wire_dtypes[key].itemsize for key, (numel, _) in self.shards.items())
        total = torch.cuda.get_device_properties(self.device).total_memory
        headroom = total - peak_reserved - num_bytes
        if self.snapshot_placement == "auto":
            self.on_gpu = headroom >= max(4 * 2**30, total // 20)
        else:
            self.on_gpu = self.snapshot_placement == "gpu"
        get_logger().info(
            f"Sparse weight snapshot ({num_bytes / 2**30:.1f} GiB) on the {'GPU' if self.on_gpu else 'host'} "
            f"(training peak {peak_reserved / 2**30:.1f} GiB of {total / 2**30:.1f} GiB, "
            f"snapshot={self.snapshot_placement})"
        )

    def _fill(self, shards: list[tuple]) -> None:
        """The first broadcast only records the weights: on the GPU if forced there, else in the host buffer
        (placed at the first delta, against the measured training peak)."""
        self.shards = {key: (shard.numel(), offset) for key, _, shard, offset, _ in shards}
        if self.snapshot_placement == "gpu":
            self.on_gpu = True
            for key, _, shard, _, _ in shards:
                self.snapshot[key] = as_bits(shard.to(f"cuda:{self.device}", self.wire_dtypes[key], copy=True))
            return
        self._allocate_host()
        for key, _, shard, _, _ in shards:
            for start in range(0, shard.numel(), self.CHUNK):
                end = start + self.CHUNK
                self.host[key][start:end].copy_(as_bits(shard[start:end].to(self.wire_dtypes[key])))
        self.snapshot = dict(self.host)

    def copy_snapshot_to_host(self) -> None:
        """Copy a GPU snapshot (the last broadcast's weights) into the host buffer, for an anchor. Callers wait
        for earlier readers of the host buffer first."""
        assert self.on_gpu
        if self.host_buffer is None:
            self._allocate_host()
        self._host_ready()
        with torch.cuda.stream(self.stream):
            for key, bits in self.snapshot.items():
                self.host[key].copy_(bits, non_blocking=self.pinned)
            self.stream.synchronize()
        self._pin_host()

    def anchor_parts(self) -> list[tuple[str, list[int], int, torch.Tensor]]:
        """This rank's slices of the full HF checkpoint of the last broadcast, from host memory (the CPU
        snapshot, or ``copy_snapshot_to_host``)."""
        parts = []
        for key, bits in self.host.items():
            values, (numel, offset) = bits.view(self.wire_dtypes[key]), self.shards[key]
            for start, name in self.segments[self.key_fqn[key]]:
                begin, end = max(start, offset), min(start + self.hf_numel[name], offset + numel)
                if begin < end:
                    parts.append((name, self.hf_tensors[name][0], begin - start, values[begin - offset : end - offset]))
        return parts

    @torch.no_grad()
    def prepare(self, shards: list[tuple], peak_reserved: int) -> None:
        """The training thread's part of an encode, before ``encode``: place the snapshot at the first delta,
        and diff the tensors held in fp8. vLLM's quantizer is torch.compile'd, and running it next to the
        training step breaks activation checkpointing there."""
        self.previous = self.snapshot
        self.fp8_part = None
        if not self.initialized:
            return
        if self.on_gpu is None:
            self._place(peak_reserved)
            if self.on_gpu:
                device = f"cuda:{self.device}"
                self.snapshot = {key: torch.empty_like(bits, device=device) for key, bits in self.host.items()}
        fp8 = [shard for shard in shards if shard[1] in self.fp8]
        if fp8:
            if not self.on_gpu or self.previous is not self.snapshot:
                self._host_ready()
            self.fp8_part = self._diff(fp8)

    @torch.no_grad()
    def encode(self, shards: list[tuple], ready: torch.cuda.Event, weights_read: threading.Event) -> tuple | None:
        """Diff the shards against the snapshot and advance it, on the side stream once ``ready`` (recorded
        after the optimizer step) has passed. Sets ``weights_read`` once done reading the weights. Returns
        ``(entries, gaps, xor value bytes, new-weight checksum per HF tensor part, scales changed on this
        rank)`` on the host, or None on the first call."""
        torch.cuda.set_device(self.device)
        try:
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                self.key_fqn = {key: fqn for key, fqn, *_ in shards}
                if not self.initialized:
                    self._fill(shards)
                    self.stream.synchronize()
                    self.initialized = True
                    return None
                if not self.on_gpu or self.previous is not self.snapshot:
                    self._host_ready()
                part = self._diff([shard for shard in shards if shard[1] not in self.fp8])
                self.stream.synchronize()
        finally:
            weights_read.set()
        self.previous = self.snapshot
        self._pin_host()
        if self.fp8_part is not None:
            part = [ours + theirs for ours, theirs in zip(part[:-1], self.fp8_part[:-1])] + [self.fp8_part[-1]]
        names, counts, gaps, values, checksum_names, checksums, scales_changed = part
        counts = torch.cat(counts).tolist() if counts else []
        checksums = dict(zip(checksum_names, torch.stack(checksums).tolist())) if checksums else {}
        entries = [(name, count) for name, count in zip(names, counts) if count]
        empty = torch.empty(0, dtype=torch.uint8)
        gaps = torch.cat(gaps) if gaps else empty
        return entries, gaps, torch.cat(values) if values else empty, checksums, scales_changed

    def _diff(self, shards: list[tuple]) -> list:
        """Diff ``shards`` against ``self.previous`` on the current stream, writing the new bits to the
        snapshot, in chunks of ``CHUNK`` elements (fp8: whole 128-row blocks)."""
        device = torch.device(f"cuda:{self.device}")
        names, counts, gaps, values = [], [], [], []
        checksum_names, checksums = [], []
        scales_changed = False
        for key, fqn, shard, offset, gathered in shards:
            previous, out, wire = self.previous[key], self.snapshot[key], self.wire_dtypes[key]
            segments = [
                (start - offset, start - offset + self.hf_numel[name], name)
                for start, name in self.segments[fqn]
                if start < offset + shard.numel() and start + self.hf_numel[name] > offset
            ]
            sums = [torch.zeros((), dtype=torch.int64, device=device) for _ in segments]
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
                        quantize_fp8_blocks(weights.reshape(-1, num_cols), ue8m0)
                        for weights in (previous_weights, current_weights)
                    )
                    scales_changed |= not torch.equal(previous_scales, current_scales)
                    if gathered is not None:
                        previous_fp8 = previous_fp8.view(-1)[offset : offset + shard.numel()]
                        current_fp8 = current_fp8.view(-1)[offset : offset + shard.numel()]
                    xor = previous_fp8.view(-1) ^ current_fp8.view(-1)
                    new_bits = current_fp8.view(-1)
                else:
                    xor = current ^ previous_bits
                    new_bits = current
                out[start:end].copy_(current, non_blocking=True)
                index = xor.nonzero().squeeze(1)
                changed.append(index + start)
                changed_values.append(xor[index])
                for i, (begin, stop, _) in enumerate(segments):
                    begin, stop = max(begin, start), min(stop, end)
                    if begin < stop:
                        sums[i] += new_bits[begin - start : stop - start].sum(dtype=torch.int64)
            checksum_names.extend(name for *_, name in segments)
            checksums.extend(sums)
            changed, changed_values = torch.cat(changed), torch.cat(changed_values)
            if fqn not in self.segment_starts:
                self.segment_starts[fqn] = torch.tensor([start for start, _ in self.segments[fqn]], device=device)
            starts = self.segment_starts[fqn]
            position = changed + offset
            segment = torch.searchsorted(starts, position, right=True) - 1
            entry_starts = torch.ones_like(segment, dtype=torch.bool)
            entry_starts[1:] = segment[1:] != segment[:-1]
            gaps.append(encode_gaps(position - starts[segment], entry_starts).cpu())
            values.append(changed_values.view(torch.uint8).cpu())
            counts.append(torch.bincount(segment, minlength=len(starts)))
            names.extend(name for _, name in self.segments[fqn])
        return [names, counts, gaps, values, checksum_names, checksums, scales_changed]


class FileSystemWeightSender(WeightSender):
    """Broadcast weights by saving a HF-compatible checkpoint (or, for LoRA
    runs, the PEFT-shaped adapter) to a shared filesystem. With ``sparse``,
    steps only carry the XOR delta against the previous broadcast (see
    ``prime_rl.utils.sparse_weights``). The delta is encoded and written in the
    background while the next step trains: the trainer only waits, before its
    next optimizer step, until the encode has read the weights. Full
    checkpoints (anchors) serve engines that fall off the delta chain."""

    def __init__(
        self,
        output_dir: Path,
        config: FileSystemWeightBroadcastConfig,
        lora_config: LoRAConfig | None = None,
    ):
        super().__init__(output_dir, config.timeout)
        if config.sparse and lora_config is not None:
            raise ValueError("Sparse weight broadcast does not support LoRA")
        self.config = config
        self.lora_config = lora_config
        self.encoder: SparseDeltaEncoder | None = None
        self.last_step: int | None = None
        self.anchor_step: int | None = None
        self.pending_anchor: int | None = None
        # Background work, each in order: encodes, delta writes and commits, periodic anchor writes
        self.encodes = ThreadPoolExecutor(1)
        self.writes = ThreadPoolExecutor(1)
        self.anchors = ThreadPoolExecutor(1)
        self.host_readers: list[Future] = []
        self.errors: list[BaseException] = []
        self.weights_read: threading.Event | None = None
        self.deferred_finish = False
        self.logger.debug("Initialized filesystem weight broadcast")

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
        if self.lora_config is not None:
            # All ranks must participate in DTensor gathering, but only master saves
            state_dict = get_lora_state().adapter_state_dict()
            for key, value in state_dict.items():
                if isinstance(value, DTensor):
                    value = value.full_tensor()
                if self.world.is_master:
                    state_dict[key] = value.to("cpu", non_blocking=False)
            if self.world.is_master:
                self.logger.debug(f"Saving adapter to {step_dir}")
                save_state_dict(state_dict, step_dir, save_sharded=False, adapter=True)
                save_lora_config(
                    model,
                    step_dir,
                    rank=self.lora_config.rank,
                    alpha=self.lora_config.alpha,
                    dropout=self.lora_config.dropout,
                )
            return

        if not self.config.sparse:
            self._save_full(model, step_dir)
            return
        self.wait_for_weight_reads()
        self._raise_errors()
        self.deferred_finish = False
        if self.world.is_master:
            # Engines that load a step with a delta dir report their weight layout next to it.
            (step_dir / DELTA_DIR).mkdir(exist_ok=True)
        if self.encoder is None:
            self.encoder = SparseDeltaEncoder(model, self.config.snapshot)
        if self.encoder.fp8_layout is None:
            # Until every engine worker has reported which tensors it holds in fp8, steps are full checkpoints.
            layout = [read_engine_layout(step_dir.parent) if self.world.is_master else None]
            dist.broadcast_object_list(layout, src=0)
            if layout[0] is not None:
                self.encoder.set_fp8_layout(model, layout[0], self.world.is_master)
        shards = self.encoder.local_shards(model, self.world.is_master)
        first = self.last_step is None
        base_step, self.last_step = self.last_step, step
        # The run's first broadcast is the checkpoint the engines booted from; a resumed run's is an anchor.
        boot = first and step == 0
        full = (first and step != 0) or (not first and self.encoder.fp8_layout is None)
        anchor = False
        if not first and not full:
            # Periodic anchors only serve engines that fall off the chain: written in the background.
            if (
                self.world.is_master
                and self.pending_anchor is not None
                and anchor_complete(self.step_dir(self.pending_anchor))
            ):
                self.anchor_step, self.pending_anchor = max(self.anchor_step or 0, self.pending_anchor), None
            start_anchor = [step % self.config.anchor_interval == 0 and self.pending_anchor is None]
            dist.broadcast_object_list(start_anchor, src=0)
            anchor = start_anchor[0]
            if anchor:
                self.pending_anchor = step
        self.encoder.prepare(shards, torch.cuda.max_memory_reserved())
        ready = torch.cuda.Event()
        ready.record()
        self.weights_read = threading.Event()
        args = (step, base_step, step_dir, shards, ready, self.weights_read)
        self._submit(self.encodes, self._encode, *args, boot, full, anchor)
        if boot:
            if self.world.is_master:
                manifest = {"step": step, "base_step": None, "files": [], "tensors": self.encoder.hf_tensors}
                (step_dir / DELTA_DIR / MANIFEST).write_text(json.dumps(manifest))
            self.anchor_step = step
            return
        self.deferred_finish = True

    def _wait_for_host_readers(self) -> None:
        for reader in self.host_readers:
            reader.exception()
        self.host_readers = []

    def _encode(
        self,
        step: int,
        base_step: int | None,
        step_dir: Path,
        shards,
        ready,
        weights_read: threading.Event,
        boot: bool,
        full: bool,
        anchor: bool,
    ) -> None:
        """Encode thread: diff v{step}, then hand the delta (or, on full steps, the checkpoint) to a writer."""
        encoder = self.encoder
        assert encoder is not None
        if not encoder.on_gpu:
            self._wait_for_host_readers()  # anchors written from the host snapshot, which the encode advances
        start = time.perf_counter()
        delta = encoder.encode(shards, ready, weights_read)
        encode_time = time.perf_counter() - start
        if delta is None:
            num_bytes = sum(numel * encoder.wire_dtypes[key].itemsize for key, (numel, _) in encoder.shards.items())
            self.logger.info(f"Recorded the {num_bytes / 2**30:.1f} GiB sparse weight snapshot in {encode_time:.1f}s")
            if boot:
                return
        elif encoder.fp8 and self._scales_changed_anywhere(step_dir, delta[4]):
            self.logger.info(f"fp8 block scales changed at v{step}: publishing a full checkpoint")
            full = True
        if full or anchor:
            if encoder.on_gpu:
                self._wait_for_host_readers()
                encoder.copy_snapshot_to_host()
            parts = encoder.anchor_parts()
        if full:
            self.host_readers.append(self._submit(self.writes, self._write_full, step, step_dir, parts))
            return
        self._submit(self.writes, self._write_delta, step, base_step, step_dir, delta[:4], encode_time)
        if anchor:
            self.host_readers.append(self._submit(self.anchors, self._write_anchor, step_dir, parts))

    def _scales_changed_anywhere(self, step_dir: Path, changed: bool) -> bool:
        """Agree through the step dir (the encode thread runs no collectives) whether any rank's fp8 block
        scales changed."""
        delta_dir = step_dir / DELTA_DIR
        delta_dir.mkdir(exist_ok=True)
        tmp = delta_dir / f".rank_{self.world.rank}.scales.tmp"
        tmp.write_text(str(int(changed)))
        tmp.rename(delta_dir / f"rank_{self.world.rank}.scales")
        flags = [delta_dir / f"rank_{rank}.scales" for rank in range(self.world.world_size)]
        while not all(flag.exists() for flag in flags):
            time.sleep(0.05)
        return any(flag.read_text() == "1" for flag in flags)

    def _write_delta(self, step: int, base_step: int, step_dir: Path, delta: tuple, encode_time: float) -> None:
        """Write this rank's delta file; the master then waits for every rank's file and commits
        the step (manifest, then the finished marker). Runs on a thread, without collectives."""
        assert self.encoder is not None
        start = time.perf_counter()
        entries, gaps, values, checksums = delta
        delta_dir = step_dir / DELTA_DIR
        files = [f"rank_{rank}.safetensors" for rank in range(self.world.world_size)]
        write_rank_file(
            delta_dir / files[self.world.rank],
            entries,
            gaps.numpy(),
            values.numpy(),
            self.encoder.hf_tensors,
            checksums,
        )
        if not self.world.is_master:
            return
        while not all((delta_dir / name).exists() for name in files):
            time.sleep(0.05)
        manifest = {
            "step": step,
            "base_step": base_step,
            "anchor_step": self.anchor_step,
            "files": files,
            "tensors": self.encoder.hf_tensors,
        }
        (delta_dir / MANIFEST).write_text(json.dumps(manifest))
        WeightSender._finish(self, step, step_dir)
        num_bytes = sum((delta_dir / name).stat().st_size for name in files)
        self.logger.info(
            f"Sparse weight delta v{step} (base v{base_step}): {num_bytes / 2**20:.1f} MiB, "
            f"encode {encode_time:.2f}s, write {time.perf_counter() - start:.2f}s (in the background)"
        )

    def _write_full(self, step: int, step_dir: Path, parts: list) -> None:
        """Publish v{step} as a full checkpoint: every rank writes its slices, the master commits."""
        self._write_anchor(step_dir, parts)
        if self.world.is_master:
            while not anchor_complete(step_dir):
                time.sleep(0.1)
            self.anchor_step = step
            WeightSender._finish(self, step, step_dir)

    def _write_anchor(self, step_dir: Path, parts: list) -> None:
        start = time.perf_counter()
        write_anchor_shards(parts, step_dir, self.world.rank, self.world.world_size)
        self.logger.debug(f"Wrote the {step_dir.name} anchor in {time.perf_counter() - start:.1f}s")

    def _finish(self, step: int, step_dir: Path) -> None:
        # A delta or full step is committed by the writer thread once every rank's files are on disk.
        if not self.deferred_finish:
            super()._finish(step, step_dir)

    def _save_full(self, model: nn.Module, step_dir: Path) -> None:
        state_dict = gather_weights_parallel(model)
        state_dict = convert_state_dict_to_hf(model, state_dict)
        self.logger.debug(f"Saving weights to {step_dir}")
        save_state_dict_parallel(state_dict, step_dir)

    def _clean(self, step: int) -> None:
        # Sparse chains replay from their anchor, so keep everything since it.
        super()._clean(step if self.anchor_step is None else min(step, self.anchor_step + 1))


class FileSystemWeightReceiver(WeightReceiver):
    """Loads broadcasts from the shared filesystem. The acknowledgement lets
    the trainer start writing; the engines are only touched once the weights
    are fully on disk. An adapter broadcast (PEFT dir) is hot-swapped under
    live traffic — an in-place adapter reload is a vLLM-native op that needs
    no engine pause; a full checkpoint pauses the engines for the load."""

    async def receive(self, step: int) -> None:
        weights_dir = self.step_dir(step)
        self._ack(step)
        await wait_for_path(weights_dir / FINISHED_MARKER)
        if (weights_dir / "adapter_config.json").exists():
            await load_lora_adapter(self.admin_plane, self.model_name, weights_dir)
            return
        if (weights_dir / DELTA_DIR / MANIFEST).exists():
            # Read and decode the delta while the engines keep serving; the pause only applies it.
            start = time.perf_counter()
            await self.admin_plane.prefetch_weights(weights_dir)
            self.logger.debug(f"Prefetched sparse weight delta v{step} in {time.perf_counter() - start:.2f}s")
        await self.admin_plane.update_weights(weights_dir, transport="filesystem", step=step)
