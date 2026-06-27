import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from prime_rl.configs.trainer import LoRAConfig, SparseFileSystemWeightBroadcastConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.weights import (
    filter_state_dict_by_layers,
    gather_weights_on_master,
    get_max_layer_num,
)
from prime_rl.utils.sparse_update import SparseUpdateStats, save_sparse_update, to_compute_tensor
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix


class SparseFileSystemWeightBroadcast(FileSystemWeightBroadcast):
    """Broadcast sparse weight patches via shared filesystem.

    Instead of writing a full checkpoint at each policy update, diffs the current
    model state against a running baseline and writes only the changed values as a
    sparse patch. The inference worker applies patches incrementally.

    When ``kernel_format`` is enabled, patches are written in vLLM kernel format
    (stacked parameter names matching ``model.named_parameters()``), allowing the
    receiver to apply them directly to GPU parameters via ``index_copy_`` without
    a CPU weight cache.
    """

    def __init__(
        self, output_dir: Path, config: SparseFileSystemWeightBroadcastConfig, lora_config: LoRAConfig | None = None
    ):
        # Sparse broadcast doesn't use the parent's checkpoint-saving fields, but we inherit
        # _collect_model_state_dict, _notify_orchestrator, and the multi-run/world infrastructure.
        from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig

        super().__init__(
            output_dir,
            FileSystemWeightBroadcastConfig(),
            lora_config,
        )
        self.kernel_format = config.kernel_format
        self._sparse_update_previous_state_dict: dict[str, Tensor] | None = None
        self._sparse_update_previous_step = 0
        self.last_metrics: dict[str, float | int] = {}

        if lora_config is not None:
            raise ValueError("Sparse filesystem broadcast is only supported for full-model weight broadcasts.")
        if self.multi_run_manager.max_runs > 1:
            raise ValueError("Sparse filesystem broadcast is not supported with multi-run training.")
        self.logger.debug(f"Sparse filesystem broadcast initialized (kernel_format={self.kernel_format})")

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
            for key in list(layer_sd):
                if isinstance(layer_sd[key], _DTensor):
                    layer_sd[key] = layer_sd[key].to(torch.bfloat16).full_tensor()

            if layer_idx < 0 or not isinstance(model, PreTrainedModelPrimeRL):
                kernel_state.update(layer_sd)
            else:
                converted = model.convert_layer_to_vllm_kernel(layer_sd, layer_idx, quantize_fp8=False)
                kernel_state.update(converted)

        return {name: tensor.to("cpu").contiguous() for name, tensor in kernel_state.items()}

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights by saving a sparse patch to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting sparse weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        self.last_metrics = {}

        if self.kernel_format:
            state_dict = self._collect_kernel_state_dict(model)
        else:
            state_dict = self._collect_model_state_dict(model)

        # Lazily initialize the baseline on the first broadcast
        if self._sparse_update_previous_state_dict is None:
            if self.world.is_master:
                self._sparse_update_previous_state_dict = {
                    name: tensor.contiguous() for name, tensor in state_dict.items()
                }
                self._sparse_update_previous_step = step
                total_numel = sum(t.numel() for t in self._sparse_update_previous_state_dict.values())
                self.logger.info(
                    f"Prepared sparse update baseline at step {step} "
                    f"(kernel_format={self.kernel_format}, "
                    f"{len(self._sparse_update_previous_state_dict)} tensors, {total_numel:,} values)"
                )
            return

        for idx in self.multi_run_manager.ready_to_update_idxs:
            if self.world.is_master:
                save_dir = get_step_path(
                    get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                    self.multi_run_manager.progress[idx].step,
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                self._save_sparse_update_patch(state_dict, save_dir, step)
                self._notify_orchestrator(save_dir)

                if self.multi_run_manager.get_orchestrator_config(self.multi_run_manager.idx_2_id[idx]) is None:
                    shutil.rmtree(self.multi_run_manager.get_run_dir(idx))
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

    def maybe_clean(self, interval_to_keep: int | None):
        # Don't clean old sparse patches — the inference worker may need them for recovery.
        return
