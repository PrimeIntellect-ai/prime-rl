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
from prime_rl.utils.pathing import wait_for_path
from prime_rl.utils.sparse_weights import (
    DELTA_DIR,
    FP8_BLOCK,
    MANIFEST,
    anchor_complete,
    as_bits,
    encode_gaps,
    quantize_fp8_blocks,
    read_engine_layout,
    write_anchor_shards,
    write_rank_file,
)
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    owned_weight_bytes,
    resolve_fqn,
    resolve_wire_dtype,
    save_state_dict,
    save_state_dict_parallel,
)


class SparseDeltaEncoder:
    """Rank-local XOR deltas of the trainer's weight shards, in HF coordinates.

    Every rank keeps the wire-dtype bits of its own shards from the previous
    broadcast and diffs against them on device, so no weights are gathered.
    The prime->HF conversion is replayed once on meta tensors: every HF tensor
    must come out as a contiguous view of one trainer tensor (renames, expert
    unstacking), which maps a changed trainer element to its HF name and index.

    Tensors the engines hold in fp8 (``fp8``: HF name -> scale format, from the
    engine layout) are quantized with vLLM's own per-block quantizer and diffed
    as fp8 bytes. Their 128-row blocks must not straddle trainer ranks, else the
    trainer tensor is gathered. A step that changes any block scale returns no
    delta, so it is published as a full checkpoint.
    """

    def __init__(self, model: nn.Module, fp8: dict[str, str]):
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
            dtype = "float8_e4m3fn" if name in fp8 else str(tensor.dtype).removeprefix("torch.")
            self.hf_tensors[name] = (list(tensor.shape), dtype)
        self.segments = {fqn: sorted(segs) for fqn, segs in segments.items()}

        # fqn -> (ue8m0 scales, columns) for trainer tensors held in fp8; quantized as one [rows, columns] matrix
        self.fp8: dict[str, tuple[bool, int]] = {}
        for fqn, segs in self.segments.items():
            modes = {fp8.get(name) for _, name in segs}
            if modes == {None}:
                continue
            shapes = [self.hf_tensors[name][0] for _, name in segs]
            num_cols = shapes[0][-1]
            num_rows = meta[fqn].numel() // num_cols
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
        self.gather: set[str] | None = None
        self.segment_starts: dict[str, torch.Tensor] = {}
        self.snapshot: dict[str, torch.Tensor] = {}
        self.scales: dict[str, torch.Tensor] = {}
        self.scales_changed = False
        self.initialized = False

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

    def _fp8_bits(
        self, key: str, fqn: str, value: torch.Tensor, shard: tuple[torch.Tensor, int] | None
    ) -> tuple[torch.Tensor, int] | None:
        """fp8 bits of this rank's shard, quantized as the engines do."""
        ue8m0, num_cols = self.fp8[fqn]
        if key in self.gather:
            quantized, scales = quantize_fp8_blocks(value.full_tensor().to(torch.bfloat16).view(-1, num_cols), ue8m0)
            if shard is None:
                return None
            bits = quantized.view(-1)[shard[1] : shard[1] + shard[0].numel()]
        elif shard is None or shard[0].numel() == 0:
            return None
        else:
            quantized, scales = quantize_fp8_blocks(shard[0].to(torch.bfloat16).view(-1, num_cols), ue8m0)
            bits = quantized.view(-1)
        previous = self.scales.get(key)
        if previous is not None and not torch.equal(previous, scales):
            self.scales_changed = True
        self.scales[key] = scales
        return bits, shard[1]

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
    ) -> tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor] | None:
        """Diff the current shards against the snapshot and advance it. Returns
        ``(entries, gaps, xor value bytes)`` on the host, or None on the first
        call and when an fp8 block scale changed."""
        first = not self.initialized
        self.initialized = True
        if self.gather is None:
            self.gather = self._find_gathered(model, is_master)
        self.scales_changed = False
        names, counts, gaps, values = [], [], [], []
        for key, value in model.state_dict().items():
            fqn = resolve_fqn(model, key)
            if fqn not in self.segments:
                continue
            shard = self._local_shard(key, value, is_master)
            if fqn in self.fp8:
                shard = self._fp8_bits(key, fqn, value, shard)
            elif shard is not None:
                shard = (as_bits(shard[0].to(self.wire_dtypes[key])), shard[1])
            if shard is None or shard[0].numel() == 0:
                continue
            bits, offset = shard
            if first:
                # The previous broadcast's bits live in pinned host memory, so the trainer's GPU memory is unchanged.
                self.snapshot[key] = torch.empty(bits.shape, dtype=bits.dtype, pin_memory=True)
                self.snapshot[key].copy_(bits)
                continue
            changed, changed_values = self._diff(bits, self.snapshot[key])
            if fqn not in self.segment_starts:
                self.segment_starts[fqn] = torch.tensor([start for start, _ in self.segments[fqn]], device=bits.device)
            starts = self.segment_starts[fqn]
            position = changed + offset
            segment = torch.searchsorted(starts, position, right=True) - 1
            entry_starts = torch.ones_like(segment, dtype=torch.bool)
            entry_starts[1:] = segment[1:] != segment[:-1]
            gaps.append(encode_gaps(position - starts[segment], entry_starts).to("cpu", non_blocking=True))
            values.append(changed_values.view(torch.uint8).to("cpu", non_blocking=True))
            counts.append(torch.bincount(segment, minlength=len(starts)))
            names.extend(name for _, name in self.segments[fqn])
        if self.fp8 and dist.is_initialized():
            changed_anywhere = torch.tensor(int(self.scales_changed), device=next(model.parameters()).device)
            dist.all_reduce(changed_anywhere, op=dist.ReduceOp.MAX)
            self.scales_changed = bool(changed_anywhere.item())
        if first or self.scales_changed:
            return None
        counts = torch.cat(counts).tolist() if counts else []  # also waits for the copies to the host
        torch.cuda.empty_cache()
        entries = [(name, count) for name, count in zip(names, counts) if count]
        empty = torch.empty(0, dtype=torch.uint8)
        return entries, torch.cat(gaps) if gaps else empty, torch.cat(values) if values else empty


class FileSystemWeightSender(WeightSender):
    """Broadcast weights by saving a HF-compatible checkpoint (or, for LoRA
    runs, the PEFT-shaped adapter) to a shared filesystem. With ``sparse``,
    steps between anchors only carry the XOR delta against the previous
    broadcast (see ``prime_rl.utils.sparse_weights``)."""

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
        self.num_deltas = 0
        self.pending_anchor: int | None = None
        # Reused across anchors: a new one starts only after the previous one was written
        self.anchor_buffer: torch.Tensor | None = None
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
        if self.world.is_master:
            # Engines that load a step with a delta dir report their weight layout next to it.
            (step_dir / DELTA_DIR).mkdir(exist_ok=True)
        if self.encoder is None:
            # Deltas start once every engine worker has reported which tensors it holds in fp8.
            layout = [read_engine_layout(step_dir.parent) if self.world.is_master else None]
            dist.broadcast_object_list(layout, src=0)
            if layout[0] is not None:
                self.encoder = SparseDeltaEncoder(model, layout[0])
        wrote_delta = self.encoder is not None and self._save_delta(model, step, step_dir)
        self.num_deltas += wrote_delta
        self.last_step = step
        if not wrote_delta or self.num_deltas == 1:
            # Engines need this full checkpoint now; the first delta's step is one too, so engines
            # check the delta path against it right away.
            self._save_anchor(model, step_dir).join()
            dist.barrier()
            self.anchor_step = step
            return
        # Periodic anchors only serve engines that fall off the chain: write them off the critical path.
        if (
            self.world.is_master
            and self.pending_anchor is not None
            and anchor_complete(self.step_dir(self.pending_anchor))
        ):
            self.anchor_step, self.pending_anchor = max(self.anchor_step, self.pending_anchor), None
        start = [step % self.config.anchor_interval == 0 and self.pending_anchor is None]
        dist.broadcast_object_list(start, src=0)
        if start[0]:
            self._save_anchor(model, step_dir)
            self.pending_anchor = step

    def _save_full(self, model: nn.Module, step_dir: Path) -> None:
        state_dict = gather_weights_parallel(model)
        state_dict = convert_state_dict_to_hf(model, state_dict)
        self.logger.debug(f"Saving weights to {step_dir}")
        save_state_dict_parallel(state_dict, step_dir)

    def _save_anchor(self, model: nn.Module, step_dir: Path) -> threading.Thread:
        """Gather this rank's share of the full checkpoint to CPU, then write it on a thread
        (no collectives); engines treat the step as an anchor once every rank's marker exists."""
        start = time.perf_counter()
        if self.anchor_buffer is None:
            self.anchor_buffer = torch.empty(owned_weight_bytes(model), dtype=torch.uint8, pin_memory=True)
        state_dict = convert_state_dict_to_hf(model, gather_weights_parallel(model, host_buffer=self.anchor_buffer))
        torch.cuda.empty_cache()
        gather_time = time.perf_counter() - start

        def write() -> None:
            write_anchor_shards(state_dict, step_dir, self.world.rank, self.world.world_size)
            self.logger.debug(
                f"Wrote the {step_dir.name} anchor: gather {gather_time:.1f}s, write {time.perf_counter() - start - gather_time:.1f}s"
            )

        thread = threading.Thread(target=write)
        thread.start()
        return thread

    def _save_delta(self, model: nn.Module, step: int, step_dir: Path) -> bool:
        """Write this step's delta (all ranks). False when the step must be a full checkpoint instead."""
        assert self.encoder is not None
        start = time.perf_counter()
        delta = self.encoder.encode(model, self.world.is_master)
        if self.encoder.scales_changed:
            self.logger.info(f"fp8 block scales changed at v{step}: publishing a full checkpoint")
        if delta is None:
            return False
        entries, gaps, values = delta
        encode_time = time.perf_counter() - start
        file_name = f"rank_{self.world.rank}.safetensors"
        delta_dir = step_dir / DELTA_DIR
        dist.barrier()
        num_bytes = write_rank_file(
            delta_dir / file_name, entries, gaps.numpy(), values.numpy(), self.encoder.hf_tensors
        )
        stats = (file_name, num_bytes, len(gaps), encode_time, time.perf_counter() - start)
        all_stats: list = [None] * self.world.world_size
        dist.all_gather_object(all_stats, stats)
        if not self.world.is_master:
            return True
        manifest = {
            "step": step,
            "base_step": self.last_step,
            "anchor_step": self.anchor_step,
            "files": [file_name for file_name, *_ in all_stats],
            "tensors": self.encoder.hf_tensors,
        }
        (delta_dir / MANIFEST).write_text(json.dumps(manifest))
        num_changes = sum(stat[2] for stat in all_stats)
        num_elements = sum(prod(shape) for shape, _ in self.encoder.hf_tensors.values())
        self.logger.info(
            f"Sparse weight delta v{step} (base v{self.last_step}): {num_changes / num_elements:.3%} changed, "
            f"{sum(stat[1] for stat in all_stats) / 2**20:.1f} MiB, "
            f"encode {max(stat[3] for stat in all_stats):.2f}s, total {max(stat[4] for stat in all_stats):.2f}s"
        )
        return True

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
