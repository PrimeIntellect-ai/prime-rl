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
    get_sparse_diffs,
    move_diff_to_device,
)
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.weights import get_fqns, get_max_layer_num
from prime_rl.utils.sparse_update import SparseUpdateStats, sparsify_tensors_into, write_sparse_patch
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

    def _collect_sparse_patch(self, model: nn.Module, param_to_diff: dict):
        """Gather weights + diffs per layer and sparsify each layer on GPU.

        ``full_tensor()`` is a collective all-gather across the trainer mesh: every
        rank holding a shard must call it, so all ranks run the gather loop. Only
        master sparsifies and accumulates the (tiny) sparse patch; the full gathered
        layer is freed before the next one, so the whole unsharded model never piles
        onto one GPU and the diff runs on GPU (fast). Returns the accumulated patch
        ``(patch_tensors, tensor_entries, total_tensors, total_numel, changed_numel,
        diff_s)`` on master, or ``None`` on non-master ranks.
        """
        compute_dtype = None if self.kernel_format else torch.bfloat16

        # Build param→name mapping for trainable params that have diffs. Canonicalize via
        # get_fqns: named_parameters() keys carry wrapper prefixes (e.g. `_checkpoint_wrapped_module.`,
        # `_orig_module.`) but state_dict() keys are already stripped clean, so raw names would never
        # match the weights and every wrapped (layer) param would be silently dropped from the patch.
        param_to_name: dict = {}
        for name, param in model.named_parameters():
            if param in param_to_diff:
                for fqn in get_fqns(model, name):
                    param_to_name.setdefault(param, []).append(fqn)

        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(
            {name: t for name, t in model.state_dict().items()},
            layer_prefix,
        )

        patch_tensors: dict[str, Tensor] = {}
        tensor_entries: list[dict] = []
        total_tensors = 0
        total_numel = 0
        changed_numel = 0
        diff_s = 0.0

        for layer_idx, layer_weights_sd, layer_diffs_sd in self._iter_layer_weights_and_diffs(
            model, param_to_diff, param_to_name, num_layers, layer_prefix
        ):
            # Resolve DTensors to full tensors on GPU (all_gather — every rank participates).
            for key in list(layer_weights_sd):
                val = layer_weights_sd[key]
                if isinstance(val, _DTensor):
                    layer_weights_sd[key] = val.to(torch.bfloat16).full_tensor()

            for key in list(layer_diffs_sd):
                val = layer_diffs_sd[key]
                if isinstance(val, _DTensor):
                    # Diffs live on CPU (optimizer offload); move each shard to GPU only for its
                    # all_gather rather than moving every diff up front (that spike OOMs the gather).
                    val = move_diff_to_device(val, "cuda")
                    layer_diffs_sd[key] = val.to(torch.bool).full_tensor()

            # Collectives done; non-master ranks drop their gathered copies immediately.
            if not self.world.is_master:
                layer_weights_sd.clear()
                layer_diffs_sd.clear()
                continue

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

            # Sparsify this layer on GPU; only the sparse indices/values are copied to CPU. The
            # full gathered layer is dropped right after, so GPU holds one layer at a time.
            diff_start = time.perf_counter()
            n_slots, layer_numel, layer_changed = sparsify_tensors_into(
                converted_weights,
                converted_diffs,
                patch_tensors=patch_tensors,
                tensor_entries=tensor_entries,
                tensor_idx_offset=total_tensors,
                compute_dtype=compute_dtype,
            )
            diff_s += time.perf_counter() - diff_start
            total_tensors += n_slots
            total_numel += layer_numel
            changed_numel += layer_changed

            converted_weights.clear()
            converted_diffs.clear()
            layer_weights_sd.clear()
            layer_diffs_sd.clear()

        if not self.world.is_master:
            return None

        return patch_tensors, tensor_entries, total_tensors, total_numel, changed_numel, diff_s

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

        # Diffs are moved to GPU lazily, one layer at a time, inside the gather loop below
        # (moving them all up front spikes memory and OOMs the all_gather).
        param_to_diff = get_sparse_diffs(self._optimizer)

        # Gather + sparsify per layer on GPU; only the sparse patch is returned (on master).
        result = self._collect_sparse_patch(model, param_to_diff)

        for idx in self.multi_run_manager.ready_to_update_idxs:
            if self.world.is_master:
                assert result is not None
                save_dir = get_step_path(
                    get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                    self.multi_run_manager.progress[idx].step,
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                self._write_sparse_patch(result, save_dir, step)
                self._notify_orchestrator(save_dir)

                if self.multi_run_manager.get_orchestrator_config(self.multi_run_manager.idx_2_id[idx]) is None:
                    shutil.rmtree(self.multi_run_manager.get_run_dir(idx))
            self.multi_run_manager.ready_to_update[idx] = False

        # Clean up diffs from optimizer state to free memory
        clear_sparse_diffs(self._optimizer)

        if self.world.is_master:
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _write_sparse_patch(self, patch, save_dir: Path, step: int) -> SparseUpdateStats:
        compute_dtype = None if self.kernel_format else torch.bfloat16
        patch_tensors, tensor_entries, total_tensors, total_numel, changed_numel, diff_s = patch

        stats = write_sparse_patch(
            patch_tensors,
            tensor_entries,
            save_dir,
            step=step,
            base_step=self._sparse_update_base_step,
            total_tensors=total_tensors,
            total_numel=total_numel,
            changed_numel=changed_numel,
            compute_dtype=compute_dtype,
            compress=self.compress,
            diff_s=diff_s,
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
