from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import torch
from torch.nn import Module

from prime_rl.inference.vllm.worker.filesystem import FileSystemWeightUpdateWorker
from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise
from prime_rl.utils.sparse_update import (
    apply_sparse_update,
    apply_sparse_update_to_params,
    is_sparse_update_dir,
    load_sparse_update_manifest,
    to_compute_tensor,
)

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class SparseFileSystemWeightUpdateWorker(FileSystemWeightUpdateWorker):
    """vLLM worker extension for applying sparse weight patches via shared filesystem.

    Detects sparse patches by the presence of a ``sparse_update_manifest.json`` and
    dispatches accordingly:

    - **Kernel format** (``compute_dtype=null`` in manifest): applies patches directly
      to GPU parameters via ``index_copy_`` — no CPU cache needed.
    - **HF format** (``compute_dtype="bfloat16"``): reconstructs a dense CPU state dict
      cache and reloads through vLLM's normal layerwise checkpoint path.

    Full checkpoints (no manifest) fall through to the parent's checkpoint loading path.
    """

    def init_broadcaster(self) -> None:
        self._sparse_update_step = 0
        self._sparse_update_state_dict: dict[str, torch.Tensor] | None = None
        self._sparse_update_last_full_weight_path: str | None = None
        self._sparse_update_last_full_weight_step: int | None = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize sparse state if init_broadcaster was not called."""
        if not hasattr(self, "_sparse_update_step"):
            self.init_broadcaster()

    def update_weights_from_path(self, weight_path: str) -> None:
        self._ensure_initialized()
        model = self._get_model()
        path = Path(weight_path)

        if is_sparse_update_dir(path):
            manifest = load_sparse_update_manifest(path)
            if manifest.get("compute_dtype") is None:
                self._apply_kernel_patch(model, path)
            else:
                self._apply_hf_patch(model, path)
            return

        # Full checkpoint: load through normal vLLM path
        super().update_weights_from_path(weight_path)
        self._sparse_update_state_dict = None
        self._sparse_update_last_full_weight_path = weight_path
        self._sparse_update_last_full_weight_step = self._extract_step(path)
        # Sync the step counter so the next sparse patch validates correctly
        if self._sparse_update_last_full_weight_step is not None:
            self._sparse_update_step = self._sparse_update_last_full_weight_step

    def _apply_kernel_patch(self, model: Module, patch_dir: Path) -> None:
        """Apply a kernel-format sparse patch directly to GPU params via index_copy_."""
        manifest = apply_sparse_update_to_params(
            model,
            patch_dir,
            expected_base_step=self._sparse_update_step,
            device=self.device,
        )
        self._sparse_update_step = manifest["step"]

    def _apply_hf_patch(self, model: Module, patch_dir: Path) -> None:
        """Apply an HF-format sparse patch using a dense CPU state dict cache."""
        self._ensure_state_dict_cache(model)
        manifest = apply_sparse_update(
            self._sparse_update_state_dict,
            patch_dir,
            expected_base_step=self._sparse_update_step,
        )
        load_weights_checkpoint_layerwise(
            model,
            self._iter_state_dict_cache(),
            self.model_runner.model_config,
            self.vllm_config,
        )
        self._sparse_update_step = manifest["step"]

    def _ensure_state_dict_cache(self, model: Module) -> None:
        if self._sparse_update_state_dict is not None:
            return

        source = self._sparse_update_last_full_weight_path
        if source is None:
            source = getattr(self.model_runner.model_config, "model", None)
        if source is None:
            raise RuntimeError("Cannot initialize sparse update receiver cache: vLLM model path is unknown.")

        self._sparse_update_state_dict = {
            name: to_compute_tensor(tensor, torch.bfloat16)
            for name, tensor in self._get_weights_iterator(model, source)
        }
        if (
            "lm_head.weight" not in self._sparse_update_state_dict
            and "model.embed_tokens.weight" in self._sparse_update_state_dict
        ):
            self._sparse_update_state_dict["lm_head.weight"] = self._sparse_update_state_dict[
                "model.embed_tokens.weight"
            ].clone()
        self._sparse_update_step = self._sparse_update_last_full_weight_step or 0

    def _iter_state_dict_cache(self) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in self._sparse_update_state_dict.items():
            yield name, tensor

    def _extract_step(self, path: Path) -> int | None:
        for part in (path, *path.parents):
            if not part.name.startswith("step_"):
                continue
            try:
                return int(part.name.removeprefix("step_"))
            except ValueError:
                continue
        return None
