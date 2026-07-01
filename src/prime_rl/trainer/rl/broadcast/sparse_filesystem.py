import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor as _DTensor

from prime_rl.configs.trainer import LoRAConfig, SparseFileSystemWeightBroadcastConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.optim_hooks import (
    clear_sparse_diffs,
    ensure_diffs_on_device,
    get_sparse_diffs,
)
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.weights import get_max_layer_num
from prime_rl.utils.sparse_update import SparseUpdateStats, save_sparse_update_from_diff
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix


class SparseFileSystemWeightBroadcast(FileSystemWeightBroadcast):
    """Broadcast sparse weight patches via shared filesystem.

    Instead of writing a full checkpoint at each policy update, captures the
    boolean diff of each parameter during ``optimizer.step()`` via pre/post-step
    hooks, then sparsifies only the changed values into a sparse patch. The
    inference worker applies patches incrementally.

    The diff is stored in optimizer state as a DTensor, so it is automatically
    offloaded by ``CPUOffloadOptimizer`` when CPU optimizer offload is enabled.

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
        self.compress = config.compress
        self._optimizer: torch.optim.Optimizer | None = None
        self._sparse_update_base_step = 0
        self.last_metrics: dict[str, float | int] = {}

        if lora_config is not None:
            raise ValueError("Sparse filesystem broadcast is only supported for full-model weight broadcasts.")
        if self.multi_run_manager.max_runs > 1:
            raise ValueError("Sparse filesystem broadcast is not supported with multi-run training.")
        self.logger.debug(
            f"Sparse filesystem broadcast initialized (kernel_format={self.kernel_format}, compress={self.compress})"
        )

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Store the optimizer reference for reading sparse diffs."""
        self._optimizer = optimizer

    def _collect_layer_weights_and_diffs(
        self, model: nn.Module, param_to_diff: dict
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]] | None:
        """Gather model weights and their diffs per-layer on master rank.

        Returns (weights_dict, diffs_dict) in PrimeRL training format on GPU,
        or None on non-master ranks.
        """
        if not self.world.is_master:
            return None

        # Build param→name mapping for trainable params that have diffs
        param_to_name: dict = {}
        for name, param in model.named_parameters():
            if param in param_to_diff:
                param_to_name.setdefault(param, []).append(name)

        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(
            {name: t for name, t in model.state_dict().items()},
            layer_prefix,
        )

        all_weights: dict[str, Tensor] = {}
        all_diffs: dict[str, Tensor] = {}

        for layer_idx, layer_weights_sd, layer_diffs_sd in self._iter_layer_weights_and_diffs(
            model, param_to_diff, param_to_name, num_layers, layer_prefix
        ):
            # Resolve DTensors to full tensors on GPU
            for key in list(layer_weights_sd):
                val = layer_weights_sd[key]
                if isinstance(val, _DTensor):
                    layer_weights_sd[key] = val.to(torch.bfloat16).full_tensor()

            for key in list(layer_diffs_sd):
                val = layer_diffs_sd[key]
                if isinstance(val, _DTensor):
                    layer_diffs_sd[key] = val.to(torch.bool).full_tensor()

            if self.kernel_format and layer_idx >= 0 and isinstance(model, PreTrainedModelPrimeRL):
                converted_weights = model.convert_layer_to_vllm_kernel(layer_weights_sd, layer_idx, quantize_fp8=False)
                converted_diffs = model.convert_layer_to_vllm_kernel(layer_diffs_sd, layer_idx, quantize_fp8=False)
            elif isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_weights_sd):
                # HF format: convert in-place
                model.convert_layer_to_hf(layer_weights_sd, layer_idx)
                model.convert_layer_to_hf(layer_diffs_sd, layer_idx)
                converted_weights = layer_weights_sd
                converted_diffs = layer_diffs_sd
            else:
                from transformers.core_model_loading import revert_weight_conversion

                converted_weights = revert_weight_conversion(model, layer_weights_sd)
                converted_diffs = revert_weight_conversion(model, layer_diffs_sd)

            all_weights.update(converted_weights)
            all_diffs.update(converted_diffs)

        return all_weights, all_diffs

    def _iter_layer_weights_and_diffs(
        self, model: nn.Module, param_to_diff: dict, param_to_name: dict, num_layers: int, layer_prefix: str
    ):
        """Yield (layer_idx, layer_weights, layer_diffs) per layer.

        Weights come from model.state_dict(); diffs come from the optimizer hook.
        Both are matched by parameter object → name → layer.
        """
        # Build name→diff mapping for all names (including tied weight aliases)
        name_to_diff: dict[str, Tensor] = {}
        for param, names in param_to_name.items():
            diff = param_to_diff[param]
            for name in names:
                name_to_diff[name] = diff

        state_dict = model.state_dict()

        # Non-layer weights first
        non_layer_weights = {key: value for key, value in state_dict.items() if not key.startswith(layer_prefix)}
        non_layer_diffs = {key: name_to_diff[key] for key in non_layer_weights if key in name_to_diff}
        # Include non-trainable params with no diff (they don't change)
        for key in non_layer_weights:
            if key not in non_layer_diffs:
                non_layer_diffs[key] = torch.zeros_like(non_layer_weights[key], dtype=torch.bool)

        yield -1, non_layer_weights, non_layer_diffs

        # Per-layer
        for i in range(num_layers):
            layer_weights = {key: value for key, value in state_dict.items() if key.startswith(f"{layer_prefix}{i}.")}
            layer_diffs = {}
            for key in layer_weights:
                if key in name_to_diff:
                    layer_diffs[key] = name_to_diff[key]
                else:
                    layer_diffs[key] = torch.zeros_like(layer_weights[key], dtype=torch.bool)
            yield i, layer_weights, layer_diffs

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights by saving a sparse patch to shared filesystem."""
        self.logger.debug("Starting broadcasting sparse weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        self.last_metrics = {}

        if self._optimizer is None:
            raise RuntimeError("Optimizer not set on sparse broadcast; call set_optimizer() after init.")

        # Move diffs to GPU for DTensor all-gather
        ensure_diffs_on_device(self._optimizer, "cuda")
        param_to_diff = get_sparse_diffs(self._optimizer)

        # Gather weights + diffs on master, apply format conversion to both
        result = self._collect_layer_weights_and_diffs(model, param_to_diff)
        if result is not None:
            weights, diffs = result

        for idx in self.multi_run_manager.ready_to_update_idxs:
            if self.world.is_master:
                save_dir = get_step_path(
                    get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                    self.multi_run_manager.progress[idx].step,
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                self._save_sparse_update_patch(weights, diffs, save_dir, step)
                self._notify_orchestrator(save_dir)

                if self.multi_run_manager.get_orchestrator_config(self.multi_run_manager.idx_2_id[idx]) is None:
                    shutil.rmtree(self.multi_run_manager.get_run_dir(idx))
            self.multi_run_manager.ready_to_update[idx] = False

        # Clean up diffs from optimizer state to free memory
        clear_sparse_diffs(self._optimizer)

        if self.world.is_master:
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _save_sparse_update_patch(
        self, weights: dict[str, Tensor], diffs: dict[str, Tensor], save_dir: Path, step: int
    ) -> SparseUpdateStats:
        compute_dtype = None if self.kernel_format else torch.bfloat16

        stats = save_sparse_update_from_diff(
            weights,
            diffs,
            save_dir,
            step=step,
            base_step=self._sparse_update_base_step,
            compute_dtype=compute_dtype,
            compress=self.compress,
        )
        self._sparse_update_base_step = step
        self.last_metrics = {
            "sparse_update/total_numel": stats.total_numel,
            "sparse_update/changed_numel": stats.changed_numel,
            "sparse_update/sparsity": stats.sparsity,
            "sparse_update/patch_bytes": stats.patch_bytes,
            "sparse_update/patched_tensors": stats.patched_tensors,
            "sparse_update/diff_s": stats.diff_s,
            "sparse_update/save_s": stats.save_s,
        }
        self.logger.info(
            f"Saved sparse update patch for step {step}: {stats.changed_numel:,}/{stats.total_numel:,} values changed "
            f"(sparsity={stats.sparsity:.4%}, patch={stats.patch_bytes / 1024**2:.2f} MiB, "
            f"diff={stats.diff_s:.2f}s, save={stats.save_s:.2f}s)"
        )
        return stats

    def maybe_clean(self, interval_to_keep: int | None):
        # Don't clean old sparse patches — the inference worker may need them for recovery.
        return
