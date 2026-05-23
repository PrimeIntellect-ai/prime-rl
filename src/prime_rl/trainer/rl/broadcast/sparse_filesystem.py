import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch import Tensor
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, LoRAConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.weights import get_max_layer_num
from prime_rl.trainer.world import get_world
from prime_rl.utils.sparse_weights import (
    SPARSE_WEIGHTS_FORMAT,
    SPARSE_WEIGHTS_MANIFEST,
    get_sparse_manifest_metrics,
    load_safetensors,
    parse_step_from_dir,
    read_sparse_manifest,
    save_safetensors,
    write_sparse_manifest,
)
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix


@dataclass
class _RunBroadcast:
    idx: int
    step: int
    save_dir: Path
    full: bool
    base_step: int | None = None
    weight_map: dict[str, str] = field(default_factory=dict)
    total_size: int = 0
    patch_files: list[dict] = field(default_factory=list)
    delta_numel: int = 0
    total_numel: int = 0
    delta_size: int = 0


def _layer_filename(layer_position: int, total_groups: int) -> str:
    return f"model-{layer_position + 1:05d}-of-{total_groups:05d}.safetensors"


def _patch_filename(layer_position: int, total_groups: int) -> str:
    return f"delta-{layer_position + 1:05d}-of-{total_groups:05d}.safetensors"


def _filter_state_dict_by_layers(
    state_dict: dict[str, Tensor], num_layers: int, layer_prefix: str
) -> list[tuple[int, dict[str, Tensor]]]:
    groups = [(-1, {key: value for key, value in state_dict.items() if not key.startswith(layer_prefix)})]
    groups.extend(
        (idx, {key: value for key, value in state_dict.items() if key.startswith(f"{layer_prefix}{idx}.")})
        for idx in range(num_layers)
    )
    return groups


def _preprocess_layer_checkpoint(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_state_dict):
        model.convert_layer_to_hf(layer_state_dict, layer_idx)
        return layer_state_dict

    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, layer_state_dict)


def _save_index(save_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(save_dir / SAFE_WEIGHTS_INDEX_NAME, "w", encoding="utf-8") as f:
        f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")


def _tensor_delta(current: Tensor, previous: Tensor, chunk_elems: int) -> tuple[Tensor, Tensor]:
    current_flat = current.contiguous().view(-1)
    previous_flat = previous.contiguous().view(-1)
    if current_flat.shape != previous_flat.shape or current_flat.dtype != previous_flat.dtype:
        raise ValueError(
            f"Sparse delta shape/dtype mismatch: current={list(current.shape)} {current.dtype}, "
            f"previous={list(previous.shape)} {previous.dtype}"
        )

    index_chunks: list[Tensor] = []
    value_chunks: list[Tensor] = []
    for start in range(0, current_flat.numel(), chunk_elems):
        end = min(start + chunk_elems, current_flat.numel())
        changed = current_flat[start:end] != previous_flat[start:end]
        local_indices = torch.nonzero(changed, as_tuple=False).flatten()
        if local_indices.numel() == 0:
            continue
        index_chunks.append((local_indices + start).to(torch.int64))
        value_chunks.append(current_flat[start:end][local_indices])

    if not index_chunks:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=current_flat.dtype)

    return torch.cat(index_chunks), torch.cat(value_chunks)


class SparseFileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast checkpoint-format weights with full snapshots plus sparse per-layer deltas."""

    def __init__(
        self,
        output_dir: Path,
        config: FileSystemWeightBroadcastConfig,
        lora_config: LoRAConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if lora_config is not None:
            raise ValueError("Sparse filesystem weight broadcast does not support LoRA adapters yet.")
        super().__init__(output_dir, lora_config)
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.dtype = dtype
        self.full_sync_interval = config.full_sync_interval
        self.chunk_elems = 16_777_216
        self._materialized_dirs: dict[int, Path] = {}
        self._materialized_steps: dict[int, int] = {}
        self._last_metrics: dict[str, float | int] = {}
        self.logger.debug(f"Sparse filesystem broadcast initialized (full_sync_interval={self.full_sync_interval})")

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int, force_full: bool = False) -> None:
        self.logger.debug("Starting sparse filesystem weight broadcast")
        self._last_metrics = {}
        start_time = time.perf_counter()
        run_broadcasts = self._prepare_run_broadcasts(force_full)
        if not run_broadcasts:
            return

        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        layer_groups = _filter_state_dict_by_layers(state_dict, num_layers, layer_prefix)
        total_groups = len(layer_groups)

        for layer_position, (layer_idx, layer_state_dict) in enumerate(layer_groups):
            if not layer_state_dict:
                continue

            current_state = self._resolve_state_dict(model, layer_state_dict)
            if not self.world.is_master:
                continue

            current_state = _preprocess_layer_checkpoint(model, current_state, layer_idx)
            if not current_state:
                continue
            full_filename = _layer_filename(layer_position, total_groups)
            patch_filename = _patch_filename(layer_position, total_groups)
            for run in run_broadcasts:
                if run.full:
                    self._write_full_group(run, current_state, full_filename)
                else:
                    self._write_delta_group(run, current_state, full_filename, patch_filename)

            del current_state

        if self.world.is_master:
            for run in run_broadcasts:
                if run.full:
                    _save_index(run.save_dir, run.weight_map, run.total_size)
                    write_sparse_manifest(
                        run.save_dir,
                        {
                            "type": "full",
                            "step": run.step,
                            "total_numel": run.total_numel,
                            "total_size": run.total_size,
                        },
                    )
                    self._materialized_dirs[run.idx] = run.save_dir
                    self._materialized_steps[run.idx] = run.step
                else:
                    write_sparse_manifest(
                        run.save_dir,
                        {
                            "type": "delta",
                            "base_step": run.base_step,
                            "step": run.step,
                            "patch_files": run.patch_files,
                            "delta_numel": run.delta_numel,
                            "total_numel": run.total_numel,
                            "delta_size": run.delta_size,
                            "total_size": run.total_size,
                        },
                    )
                    self._materialized_steps[run.idx] = run.step

                (run.save_dir / "STABLE").touch()
                self.multi_run_manager.ready_to_update[run.idx] = False

            self._last_metrics = self._compute_broadcast_metrics(run_broadcasts)
            self.logger.debug(f"Sparse filesystem weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _prepare_run_broadcasts(self, force_full: bool) -> list[_RunBroadcast]:
        run_broadcasts: list[_RunBroadcast] = []
        for idx in self.multi_run_manager.ready_to_update_idxs:
            run_step = self.multi_run_manager.progress[idx].step
            save_dir = get_step_path(get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)), run_step)
            full = self._should_write_full(idx, run_step, force_full)

            if self.world.is_master:
                shutil.rmtree(save_dir, ignore_errors=True)
                save_dir.mkdir(parents=True, exist_ok=True)

            base_step = None if full else self._materialized_steps[idx]
            run = _RunBroadcast(idx=idx, step=run_step, save_dir=save_dir, full=full, base_step=base_step)
            if self.world.is_master and not full:
                self._ensure_materialized_cache(idx)
            run_broadcasts.append(run)

        return run_broadcasts

    def _should_write_full(self, idx: int, step: int, force_full: bool) -> bool:
        if force_full or idx not in self._materialized_steps:
            return True
        return bool(self.full_sync_interval and step % self.full_sync_interval == 0)

    def _cache_dir(self, idx: int) -> Path:
        run_dir = self.multi_run_manager.get_run_dir(idx)
        return get_broadcast_dir(run_dir) / ".sparse_cache"

    def _ensure_materialized_cache(self, idx: int) -> None:
        materialized_dir = self._materialized_dirs[idx]
        cache_dir = self._cache_dir(idx)
        if materialized_dir == cache_dir:
            return

        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.copytree(
            materialized_dir,
            cache_dir,
            ignore=shutil.ignore_patterns("STABLE", SPARSE_WEIGHTS_MANIFEST),
        )
        self._materialized_dirs[idx] = cache_dir

    def _resolve_state_dict(self, model: nn.Module, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        resolved: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if isinstance(value, DTensor):
                value = cast(DTensor, value.to(self.dtype)).full_tensor()

            if self.world.is_master:
                fqns = get_fqns(model, key)
                assert len(fqns) == 1
                resolved_key = next(iter(fqns))
                resolved[resolved_key] = value.to("cpu", non_blocking=False)

        dist.barrier()
        return resolved

    def _write_full_group(self, run: _RunBroadcast, state_dict: dict[str, Tensor], filename: str) -> None:
        save_safetensors(state_dict, run.save_dir / filename)
        for name, tensor in state_dict.items():
            run.weight_map[name] = filename
            run.total_numel += tensor.numel()
            run.total_size += tensor.numel() * tensor.element_size()

    def _write_delta_group(
        self,
        run: _RunBroadcast,
        current_state: dict[str, Tensor],
        full_filename: str,
        patch_filename: str,
    ) -> None:
        materialized_dir = self._materialized_dirs[run.idx]
        previous_state = load_safetensors(materialized_dir / full_filename)
        patch_state: dict[str, Tensor] = {}
        patch_entries: list[dict] = []

        for name, current in current_state.items():
            if name not in previous_state:
                raise KeyError(f"Previous materialized weights are missing {name}")

            previous = previous_state[name]
            indices, values = _tensor_delta(current, previous, self.chunk_elems)
            run.total_numel += current.numel()
            run.total_size += current.numel() * current.element_size()
            run.delta_numel += indices.numel()
            if indices.numel() == 0:
                continue

            indices_key = f"{name}.indices"
            values_key = f"{name}.values"
            patch_state[indices_key] = indices
            patch_state[values_key] = values
            patch_entries.append(
                {
                    "name": name,
                    "indices_key": indices_key,
                    "values_key": values_key,
                    "shape": list(current.shape),
                    "dtype": str(current.dtype),
                    "numel": current.numel(),
                    "num_changed": indices.numel(),
                }
            )

        if patch_entries:
            save_file(patch_state, run.save_dir / patch_filename, metadata={"format": "pt"})
            run.delta_size += (run.save_dir / patch_filename).stat().st_size
            run.patch_files.append({"file": patch_filename, "tensors": patch_entries})

        save_safetensors(current_state, materialized_dir / full_filename)

    def _compute_broadcast_metrics(self, run_broadcasts: list[_RunBroadcast]) -> dict[str, float | int]:
        num_full = sum(run.full for run in run_broadcasts)
        num_delta = len(run_broadcasts) - num_full
        total_numel = sum(run.total_numel for run in run_broadcasts)
        total_size = sum(run.total_size for run in run_broadcasts)
        delta_numel = sum(run.delta_numel if not run.full else run.total_numel for run in run_broadcasts)
        delta_size = sum(run.delta_size if not run.full else run.total_size for run in run_broadcasts)
        patch_files = sum(len(run.patch_files) for run in run_broadcasts)
        manifest_metrics = get_sparse_manifest_metrics(
            {
                "format": SPARSE_WEIGHTS_FORMAT,
                "type": "delta" if num_delta else "full",
                "delta_numel": delta_numel,
                "total_numel": total_numel,
                "delta_size": delta_size,
                "total_size": total_size,
                "patch_files": [{}] * patch_files,
            }
        )
        return {
            **manifest_metrics,
            "weight_broadcast/sparse/runs": len(run_broadcasts),
            "weight_broadcast/sparse/full_syncs": num_full,
            "weight_broadcast/sparse/delta_syncs": num_delta,
        }

    def get_metrics(self) -> dict[str, float | int]:
        return self._last_metrics

    def _protected_base_steps(self, broadcast_dir: Path) -> set[int]:
        protected: set[int] = set()
        for step_dir in broadcast_dir.glob("step_*"):
            manifest = read_sparse_manifest(step_dir)
            if manifest is None or manifest.get("type") != "delta":
                continue
            base_step = manifest.get("base_step")
            if isinstance(base_step, int):
                protected.add(base_step)
        return protected

    def maybe_clean(self, max_async_level: int, interval_to_keep: int | None):
        for idx in self.multi_run_manager.used_idxs:
            broadcast_dir = get_broadcast_dir(self.multi_run_manager.get_run_dir(idx))
            oldest_candidate_step = max(self.multi_run_manager.progress[idx].step - (max_async_level + 1), 0)
            protected_base_steps = self._protected_base_steps(broadcast_dir)

            for step_dir in broadcast_dir.glob("step_*"):
                step = parse_step_from_dir(step_dir)
                if step > oldest_candidate_step:
                    continue
                if interval_to_keep and step % interval_to_keep == 0:
                    continue
                if step in protected_base_steps:
                    continue

                self.logger.debug(f"Removing sparse broadcast path {step_dir}")
                shutil.rmtree(step_dir, ignore_errors=True)
