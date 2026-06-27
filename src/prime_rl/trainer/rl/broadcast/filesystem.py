import shutil
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, LoRAConfig
from prime_rl.trainer.lora import save_lora_config
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import maybe_clean
from prime_rl.trainer.weights import (
    filter_state_dict_by_layers,
    gather_weights_on_master,
    get_max_layer_num,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.sparse_update import SparseUpdateStats, save_sparse_update, to_compute_tensor
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via shared filesystem."""

    def __init__(
        self, output_dir: Path, config: FileSystemWeightBroadcastConfig, lora_config: LoRAConfig | None = None
    ):
        super().__init__(output_dir, lora_config)
        self.sparse = config.sparse
        self.kernel_format = config.sparse and config.kernel_format
        self.save_format: Literal["safetensors", "torch"] = config.save_format
        self.save_sharded = config.save_sharded if lora_config is None else False
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self._sparse_update_previous_state_dict: dict[str, Tensor] | None = None
        self._sparse_update_previous_step = 0
        self.last_metrics: dict[str, float | int] = {}
        if self.sparse:
            if lora_config is not None:
                raise ValueError("Sparse filesystem broadcast is only supported for full-model weight broadcasts.")
            if self.multi_run_manager.max_runs > 1:
                raise ValueError("Sparse filesystem broadcast is not supported with multi-run training.")
        self.logger.debug(
            f"Filesystem broadcast initialized (sparse={self.sparse}, save_format={config.save_format}, "
            f"save_sharded={self.save_sharded})"
        )

    def _collect_model_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
        if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
            model.convert_to_hf(state_dict)
        else:
            from transformers.core_model_loading import revert_weight_conversion

            state_dict = revert_weight_conversion(model, state_dict)
        return state_dict

    def _collect_kernel_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        """Gather model weights and convert to vLLM kernel format per-layer.

        Uses ``convert_layer_to_vllm_kernel`` for layer weights (handles stacking
        and optional FP8 quantization). Non-layer weights are kept in HF format.
        """
        from torch.distributed.tensor import DTensor as _DTensor

        raw_state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
        if not self.world.is_master:
            return {}

        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(raw_state_dict, layer_prefix)
        kernel_state: dict[str, Tensor] = {}

        for layer_idx, layer_sd in filter_state_dict_by_layers(raw_state_dict, num_layers, layer_prefix):
            # Resolve any remaining DTensors
            for key in list(layer_sd):
                if isinstance(layer_sd[key], _DTensor):
                    layer_sd[key] = layer_sd[key].to(torch.bfloat16).full_tensor()

            if layer_idx < 0 or not isinstance(model, PreTrainedModelPrimeRL):
                # Non-layer weights or non-custom models: keep as-is
                kernel_state.update(layer_sd)
            else:
                converted = model.convert_layer_to_vllm_kernel(layer_sd, layer_idx, quantize_fp8=False)
                kernel_state.update(converted)

        # Move everything to CPU for diffing
        return {name: tensor.to("cpu").contiguous() for name, tensor in kernel_state.items()}

    def prepare_baseline(self, model: nn.Module, step: int) -> None:
        if not self.sparse:
            return

        if self.kernel_format:
            state_dict = self._collect_kernel_state_dict(model)
        else:
            state_dict = self._collect_model_state_dict(model)
        if self.world.is_master:
            self._sparse_update_previous_state_dict = {name: tensor.contiguous() for name, tensor in state_dict.items()}
            self._sparse_update_previous_step = step
            total_numel = sum(tensor.numel() for tensor in self._sparse_update_previous_state_dict.values())
            self.logger.info(
                f"Prepared sparse update baseline at step {step} "
                f"(kernel_format={self.kernel_format}, "
                f"{len(self._sparse_update_previous_state_dict)} tensors, {total_numel:,} values)"
            )

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        self.last_metrics = {}
        adapter_only = self.lora_config is not None

        if not adapter_only:
            state_dict = self._collect_model_state_dict(model)

        for idx in self.multi_run_manager.ready_to_update_idxs:
            self.logger.debug(
                f"Broadcasting weights for run {idx} (ready_to_update={self.multi_run_manager.ready_to_update[idx]})"
            )

            if adapter_only:
                # For adapter-only, MultiRunManager creates state dict directly for each run
                # All ranks must participate in DTensor gathering, but only master saves
                state_dict = self.multi_run_manager.get_state_dict_for_run(idx)
                for key, value in state_dict.items():
                    if isinstance(value, DTensor):
                        value = value.full_tensor()
                    if self.world.is_master:
                        state_dict[key] = value.to("cpu", non_blocking=False)

            # TODO: Broadcast ready to update in sync, then we dont need to gather on not ready
            if self.world.is_master:
                try:
                    save_dir = get_step_path(
                        get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                        self.multi_run_manager.progress[idx].step,
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)

                    self.logger.debug(f"Saving weights for run {idx} to {save_dir}")
                    if self.sparse and not adapter_only:
                        if self.kernel_format:
                            state_dict = self._collect_kernel_state_dict(model)
                        self._save_sparse_update_patch(state_dict, save_dir, step)
                    else:
                        save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded, adapter=adapter_only)
                    if adapter_only:
                        orch_lora = self.multi_run_manager.config[idx].model.lora
                        save_lora_config(
                            model,
                            save_dir,
                            rank=orch_lora.rank,
                            alpha=orch_lora.alpha,
                            dropout=self.lora_config.dropout,
                        )

                    self._notify_orchestrator(save_dir)

                    # If the run is deleted, remove the run directory
                    # This is avoid the creation of zombie runs when the directory is deleted while we are broadcasting which recreates the directory
                    if self.multi_run_manager.get_orchestrator_config(self.multi_run_manager.idx_2_id[idx]) is None:
                        shutil.rmtree(self.multi_run_manager.get_run_dir(idx))

                except FileNotFoundError:
                    self.logger.warning(f"Run {idx} is deleted, skipping")
                except Exception as e:
                    self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
                finally:
                    self.multi_run_manager.ready_to_update[idx] = False

        if self.world.is_master:
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _save_sparse_update_patch(self, state_dict: dict[str, Tensor], save_dir: Path, step: int) -> SparseUpdateStats:
        if self._sparse_update_previous_state_dict is None:
            raise RuntimeError("Sparse update baseline was not prepared before the first sparse broadcast.")

        if self.kernel_format:
            current_state = {name: tensor.contiguous() for name, tensor in state_dict.items()}
            compute_dtype = None
        else:
            current_state = {name: to_compute_tensor(tensor, torch.bfloat16) for name, tensor in state_dict.items()}
            compute_dtype = torch.bfloat16
        stats = save_sparse_update(
            self._sparse_update_previous_state_dict,
            current_state,
            save_dir,
            step=step,
            base_step=self._sparse_update_previous_step,
            compute_dtype=compute_dtype,
        )
        self._sparse_update_previous_state_dict = current_state
        self._sparse_update_previous_step = step
        self.last_metrics = {
            "sparse_update/total_numel": stats.total_numel,
            "sparse_update/changed_numel": stats.changed_numel,
            "sparse_update/sparsity": stats.sparsity,
            "sparse_update/patch_bytes": stats.patch_bytes,
            "sparse_update/patched_tensors": stats.patched_tensors,
        }
        self.logger.info(
            f"Saved sparse update patch for step {step}: {stats.changed_numel:,}/{stats.total_numel:,} values changed "
            f"(sparsity={stats.sparsity:.4%}, patch={stats.patch_bytes / 1024**2:.2f} MiB)"
        )
        return stats

    def _notify_orchestrator(self, save_dir: Path):
        """Notify the orchestrator that the weights have been broadcast by writing a 'STABLE' file to a shared filesystem."""
        stable_file = save_dir / "STABLE"
        stable_file.touch()

    def maybe_clean(self, interval_to_keep: int | None):
        if self.sparse:
            return
        for idx in self.multi_run_manager.used_idxs:
            maybe_clean(
                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                self.multi_run_manager.progress[idx].step,
                interval_to_keep,
            )
