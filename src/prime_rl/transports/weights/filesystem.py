import json
import threading
import time
from collections import defaultdict
from math import prod
from pathlib import Path

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
    pin_in_background,
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
    broadcast in pinned host memory and diffs against them on device, so no
    weights are gathered. The prime->HF conversion is replayed once on meta
    tensors: every HF tensor must come out as a contiguous view of one trainer
    tensor (renames, expert unstacking), which maps a changed trainer element to
    its HF name and index.

    Tensors the engines hold in fp8 (``set_fp8_layout``: HF name -> scale format,
    from the engine layout) are quantized with vLLM's own per-block quantizer,
    previous and current weights alike, and diffed as fp8 bytes. Their 128-row
    blocks must not straddle trainer ranks, else the trainer tensor is gathered.
    A step that changes any block scale returns no delta, so it is published as
    a full checkpoint.
    """

    def __init__(self, model: nn.Module):
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
        self.snapshot: dict[str, torch.Tensor] = {}
        self.offsets: dict[str, int] = {}
        self.pinning: threading.Thread | None = None
        self.scales_changed = False
        self.initialized = False

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

    def _fp8_diff(
        self, key: str, fqn: str, value: torch.Tensor, bits: torch.Tensor, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Changed flat indices, XOR values and new fp8 bits of this rank's shard, with previous and
        current weights quantized as the engines do; advances the snapshot."""
        ue8m0, num_cols = self.fp8[fqn]
        previous = self.snapshot[key].to(bits.device, non_blocking=True).view(torch.bfloat16)
        current = bits.view(torch.bfloat16)
        if key in self.gather:
            local_shape = (previous.numel() // prod(value.shape[1:]), *value.shape[1:])
            previous = DTensor.from_local(
                previous.view(local_shape),
                value.device_mesh,
                value.placements,
                shape=value.shape,
                stride=value.stride(),
            ).full_tensor()
            current = value.full_tensor().to(torch.bfloat16)
        (previous_bits, previous_scales), (current_bits, current_scales) = (
            quantize_fp8_blocks(weights.reshape(-1, num_cols), ue8m0) for weights in (previous, current)
        )
        if not torch.equal(previous_scales, current_scales):
            self.scales_changed = True
        if key in self.gather:
            start = offset
            previous_bits = previous_bits.view(-1)[start : start + bits.numel()]
            current_bits = current_bits.view(-1)[start : start + bits.numel()]
        xor = previous_bits.view(-1) ^ current_bits.view(-1)
        self.snapshot[key].copy_(bits, non_blocking=True)
        changed = xor.nonzero().squeeze(1)
        return changed, xor[changed], current_bits.view(-1)

    def _allocate_snapshot(self, model: nn.Module, is_master: bool) -> None:
        """The previous broadcast's bits live in one host buffer, so trainer GPU memory is unchanged. It
        is page-locked on a thread while the next step trains."""
        sizes = {}
        for key, value in model.state_dict().items():
            shard = self._local_shard(key, value, is_master) if resolve_fqn(model, key) in self.segments else None
            if shard is not None and shard[0].numel():
                sizes[key] = (shard[0].numel(), shard[1], self.wire_dtypes[key])
        aligned = {key: -(-numel * dtype.itemsize // 8) * 8 for key, (numel, _, dtype) in sizes.items()}
        buffer, self.pinning = pin_in_background(sum(aligned.values()))
        self.pin_start = time.perf_counter()
        offset = 0
        for key, (numel, global_offset, dtype) in sizes.items():
            bits = bits_dtype(str(dtype).removeprefix("torch."))
            self.snapshot[key] = buffer[offset : offset + numel * dtype.itemsize].view(bits)
            self.offsets[key] = global_offset
            offset += aligned[key]

    def anchor_parts(self, model: nn.Module) -> list[tuple[str, list[int], int, torch.Tensor]]:
        """This rank's slices of the full HF checkpoint in the snapshot (the last broadcast's wire weights)."""
        parts = []
        for key, bits in self.snapshot.items():
            fqn = resolve_fqn(model, key)
            values, offset = bits.view(self.wire_dtypes[key]), self.offsets[key]
            for start, name in self.segments[fqn]:
                begin, end = max(start, offset), min(start + self.hf_numel[name], offset + bits.numel())
                if begin < end:
                    shape = self.hf_tensors[name][0]
                    parts.append((name, shape, begin - start, values[begin - offset : end - offset]))
        return parts

    @staticmethod
    def _diff(bits: torch.Tensor, previous: torch.Tensor, chunk: int = 1 << 28) -> tuple[torch.Tensor, torch.Tensor]:
        """Changed flat indices of ``bits`` vs the pinned host ``previous`` and their XOR values,
        streaming ``previous`` through the GPU in chunks and updating it in place."""
        changed, changed_values = [], []
        for start in range(0, bits.numel(), chunk):
            current = bits[start : start + chunk]
            xor = current ^ previous[start : start + chunk].to(bits.device, non_blocking=True)
            previous[start : start + chunk].copy_(current, non_blocking=True)
            index = xor.nonzero().squeeze(1)
            changed.append(index + start)
            changed_values.append(xor[index])
        return torch.cat(changed), torch.cat(changed_values)

    @torch.no_grad()
    def encode(
        self, model: nn.Module, is_master: bool
    ) -> tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor, dict[str, int]] | None:
        """Diff the current shards against the snapshot and advance it. Returns
        ``(entries, gaps, xor value bytes, new-weight checksum per HF tensor part)``
        on the host, or None on the first call and when an fp8 block scale changed."""
        first = not self.initialized
        self.initialized = True
        self.scales_changed = False
        if first:
            self._allocate_snapshot(model, is_master)
        elif self.pinning is not None:
            wait = time.perf_counter()
            self.pinning.join()
            self.pinning = None
            get_logger().info(
                f"Page-locked the sparse weight snapshot in {time.perf_counter() - self.pin_start:.1f}s "
                f"(waited {time.perf_counter() - wait:.1f}s)"
            )
        names, counts, gaps, values = [], [], [], []
        checksum_names, checksums = [], []
        for key, value in model.state_dict().items():
            fqn = resolve_fqn(model, key)
            if fqn not in self.segments:
                continue
            shard = self._local_shard(key, value, is_master)
            if shard is None or shard[0].numel() == 0:
                if key in self.gather and not first:
                    value.full_tensor()  # every rank joins the all-gathers of a gathered fp8 tensor
                    value.full_tensor()
                continue
            bits, offset = as_bits(shard[0].to(self.wire_dtypes[key])), shard[1]
            if first:
                self.snapshot[key].copy_(bits)
                continue
            if fqn in self.fp8:
                changed, changed_values, bits = self._fp8_diff(key, fqn, value, bits, offset)
            else:
                changed, changed_values = self._diff(bits, self.snapshot[key])
            for start, name in self.segments[fqn]:
                begin, end = max(start, offset), min(start + self.hf_numel[name], offset + bits.numel())
                if begin < end:
                    checksum_names.append(name)
                    checksums.append(bits[begin - offset : end - offset].sum(dtype=torch.int64))
            if fqn not in self.segment_starts:
                self.segment_starts[fqn] = torch.tensor([start for start, _ in self.segments[fqn]], device=bits.device)
            starts = self.segment_starts[fqn]
            position = changed + offset
            segment = torch.searchsorted(starts, position, right=True) - 1
            entry_starts = torch.ones_like(segment, dtype=torch.bool)
            entry_starts[1:] = segment[1:] != segment[:-1]
            gaps.append(encode_gaps(position - starts[segment], entry_starts).cpu())
            values.append(changed_values.view(torch.uint8).cpu())
            counts.append(torch.bincount(segment, minlength=len(starts)))
            names.extend(name for _, name in self.segments[fqn])
        if self.fp8 and dist.is_initialized():
            changed_anywhere = torch.tensor(int(self.scales_changed), device=next(model.parameters()).device)
            dist.all_reduce(changed_anywhere, op=dist.ReduceOp.MAX)
            self.scales_changed = bool(changed_anywhere.item())
        torch.cuda.synchronize()  # the snapshot is written asynchronously
        if first or self.scales_changed:
            return None
        counts = torch.cat(counts).tolist() if counts else []
        checksums = dict(zip(checksum_names, torch.stack(checksums).tolist())) if checksums else {}
        torch.cuda.empty_cache()
        entries = [(name, count) for name, count in zip(names, counts) if count]
        empty = torch.empty(0, dtype=torch.uint8)
        return entries, torch.cat(gaps) if gaps else empty, torch.cat(values) if values else empty, checksums


class FileSystemWeightSender(WeightSender):
    """Broadcast weights by saving a HF-compatible checkpoint (or, for LoRA
    runs, the PEFT-shaped adapter) to a shared filesystem. With ``sparse``,
    steps only carry the XOR delta against the previous broadcast (see
    ``prime_rl.utils.sparse_weights``), written on a background thread while
    training continues; full checkpoints (anchors) serve engines that fall off
    the delta chain."""

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
        # Background writes of the last step (its delta, and its anchor on anchor steps)
        self.writers: list[threading.Thread] = []
        self.deferred_finish = False
        self.logger.debug("Initialized filesystem weight broadcast")

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
        # A step's background writes read the snapshot, which the next encode updates.
        for writer in self.writers:
            writer.join()
        self.writers, self.deferred_finish = [], False
        if self.world.is_master:
            # Engines that load a step with a delta dir report their weight layout next to it.
            (step_dir / DELTA_DIR).mkdir(exist_ok=True)
        if self.encoder is None:
            self.encoder = SparseDeltaEncoder(model)
        if self.encoder.fp8_layout is None:
            # Until every engine worker has reported which tensors it holds in fp8, steps are full checkpoints.
            layout = [read_engine_layout(step_dir.parent) if self.world.is_master else None]
            dist.broadcast_object_list(layout, src=0)
            if layout[0] is not None:
                self.encoder.set_fp8_layout(model, layout[0], self.world.is_master)
        first = not self.encoder.initialized
        start = time.perf_counter()
        delta = self.encoder.encode(model, self.world.is_master)
        encode_time = time.perf_counter() - start
        if self.encoder.scales_changed:
            self.logger.info(f"fp8 block scales changed at v{step}: publishing a full checkpoint")
        base_step, self.last_step = self.last_step, step
        if first:
            num_bytes = sum(snapshot.nbytes for snapshot in self.encoder.snapshot.values())
            self.logger.info(
                f"Initialized the {num_bytes / 2**30:.1f} GiB sparse weight snapshot in {encode_time:.1f}s"
            )
        if first and step == 0:
            # The run's first broadcast is the checkpoint the engines booted from.
            if self.world.is_master:
                manifest = {"step": step, "base_step": None, "files": [], "tensors": self.encoder.hf_tensors}
                (step_dir / DELTA_DIR / MANIFEST).write_text(json.dumps(manifest))
            self.anchor_step = step
            return
        if delta is None or self.encoder.fp8_layout is None:
            self._save_anchor(model, step_dir).join()
            dist.barrier()
            self.anchor_step = step
            return
        writer = threading.Thread(target=self._write_delta, args=(step, base_step, step_dir, delta, encode_time))
        writer.start()
        self.writers.append(writer)
        self.deferred_finish = True
        # Periodic anchors only serve engines that fall off the chain: written in the background too.
        if (
            self.world.is_master
            and self.pending_anchor is not None
            and anchor_complete(self.step_dir(self.pending_anchor))
        ):
            self.anchor_step, self.pending_anchor = max(self.anchor_step, self.pending_anchor), None
        start_anchor = [step % self.config.anchor_interval == 0 and self.pending_anchor is None]
        dist.broadcast_object_list(start_anchor, src=0)
        if start_anchor[0]:
            self.writers.append(self._save_anchor(model, step_dir))
            self.pending_anchor = step

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
        super()._finish(step, step_dir)
        num_bytes = sum((delta_dir / name).stat().st_size for name in files)
        self.logger.info(
            f"Sparse weight delta v{step} (base v{base_step}): {num_bytes / 2**20:.1f} MiB, "
            f"encode {encode_time:.2f}s, write {time.perf_counter() - start:.2f}s (in the background)"
        )

    def _finish(self, step: int, step_dir: Path) -> None:
        # A delta step is committed by its writer thread once every rank's file is on disk.
        if not self.deferred_finish:
            super()._finish(step, step_dir)

    def _save_full(self, model: nn.Module, step_dir: Path) -> None:
        state_dict = gather_weights_parallel(model)
        state_dict = convert_state_dict_to_hf(model, state_dict)
        self.logger.debug(f"Saving weights to {step_dir}")
        save_state_dict_parallel(state_dict, step_dir)

    def _save_anchor(self, model: nn.Module, step_dir: Path) -> threading.Thread:
        """Write this rank's slices of the full checkpoint from the snapshot on a thread; engines treat
        the step as an anchor once every rank's marker exists."""
        assert self.encoder is not None
        parts = self.encoder.anchor_parts(model)
        start = time.perf_counter()

        def write() -> None:
            write_anchor_shards(parts, step_dir, self.world.rank, self.world.world_size)
            self.logger.debug(f"Wrote the {step_dir.name} anchor in {time.perf_counter() - start:.1f}s")

        thread = threading.Thread(target=write)
        thread.start()
        return thread

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
