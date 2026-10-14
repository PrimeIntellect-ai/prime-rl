import json
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
from prime_rl.utils.sparse_weights import DELTA_DIR, MANIFEST, as_bits, encode_gaps, write_rank_file
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
    broadcast and diffs against them on device, so no weights are gathered.
    The prime->HF conversion is replayed once on meta tensors: every HF tensor
    must come out as a contiguous view of one trainer tensor (renames, expert
    unstacking), which maps a changed trainer element to its HF name and index.
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
        for name, tensor in convert_state_dict_to_hf(model, dict(meta)).items():
            base = tensor if tensor._base is None else tensor._base
            if id(base) not in owner or not tensor.is_contiguous():
                raise NotImplementedError(f"Sparse weight broadcast cannot map {name} to a contiguous trainer slice")
            if tensor.numel() >= 2**32:
                raise NotImplementedError(f"Sparse weight broadcast indexes tensors below 2^32 elements ({name})")
            segments[owner[id(base)]].append((tensor.storage_offset(), name))
            self.hf_tensors[name] = (list(tensor.shape), str(tensor.dtype).removeprefix("torch."))
        self.segments = {fqn: sorted(segs) for fqn, segs in segments.items()}
        self.segment_starts: dict[str, torch.Tensor] = {}
        self.snapshot: dict[str, torch.Tensor] = {}
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

    @torch.no_grad()
    def encode(
        self, model: nn.Module, is_master: bool
    ) -> tuple[list[tuple[str, int]], torch.Tensor, torch.Tensor] | None:
        """Diff the current shards against the snapshot and advance it. Returns
        ``(entries, gaps, xor value bytes)`` on the host, or None on the first call."""
        first = not self.initialized
        self.initialized = True
        names, counts, gaps, values = [], [], [], []
        for key, value in model.state_dict().items():
            fqn = resolve_fqn(model, key)
            shard = self._local_shard(key, value, is_master) if fqn in self.segments else None
            if shard is None or shard[0].numel() == 0:
                continue
            local, offset = shard
            bits = as_bits(local.to(self.wire_dtypes[key]))
            if first:
                self.snapshot[key] = bits.clone()
                continue
            previous = self.snapshot[key]
            xor = bits ^ previous
            previous.copy_(bits)
            changed = xor.nonzero().squeeze(1)
            if fqn not in self.segment_starts:
                self.segment_starts[fqn] = torch.tensor([start for start, _ in self.segments[fqn]], device=xor.device)
            starts = self.segment_starts[fqn]
            position = changed + offset
            segment = torch.searchsorted(starts, position, right=True) - 1
            entry_starts = torch.ones_like(segment, dtype=torch.bool)
            entry_starts[1:] = segment[1:] != segment[:-1]
            gaps.append(encode_gaps(position - starts[segment], entry_starts))
            values.append(xor[changed].view(torch.uint8))
            counts.append(torch.bincount(segment, minlength=len(starts)))
            names.extend(name for _, name in self.segments[fqn])
        if first:
            return None
        counts = torch.cat(counts).tolist() if counts else []
        entries = [(name, count) for name, count in zip(names, counts) if count]
        empty = torch.empty(0, dtype=torch.uint8)
        gaps_host = torch.cat(gaps).cpu() if gaps else empty
        values_host = torch.cat(values).cpu() if values else empty
        return entries, gaps_host, values_host


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
        self.num_broadcasts = 0
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

        # The second broadcast is an anchor too, so engines check the first delta right away.
        self.num_broadcasts += 1
        anchor = not self.config.sparse or self.num_broadcasts <= 2 or step % self.config.anchor_interval == 0
        if self.config.sparse:
            if self.encoder is None:
                self.encoder = SparseDeltaEncoder(model)
            # Anchors also carry the delta, so engines can check the chain against them.
            self._save_delta(model, step, step_dir)
            self.last_step = step
        if anchor:
            state_dict = gather_weights_parallel(model)
            state_dict = convert_state_dict_to_hf(model, state_dict)
            self.logger.debug(f"Saving weights to {step_dir}")
            save_state_dict_parallel(state_dict, step_dir)
            self.anchor_step = step

    def _save_delta(self, model: nn.Module, step: int, step_dir: Path) -> None:
        assert self.encoder is not None
        start = time.perf_counter()
        delta = self.encoder.encode(model, self.world.is_master)
        if delta is None:
            return
        entries, gaps, values = delta
        encode_time = time.perf_counter() - start
        file_name = f"rank_{self.world.rank}.safetensors"
        delta_dir = step_dir / DELTA_DIR
        if self.world.is_master:
            delta_dir.mkdir()
        dist.barrier()
        num_bytes = write_rank_file(
            delta_dir / file_name, entries, gaps.numpy(), values.numpy(), self.encoder.hf_tensors
        )
        stats = (file_name, num_bytes, len(gaps), encode_time, time.perf_counter() - start)
        all_stats: list = [None] * self.world.world_size
        dist.all_gather_object(all_stats, stats)
        if not self.world.is_master:
            return
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
