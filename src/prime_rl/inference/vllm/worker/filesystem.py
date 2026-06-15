from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import torch
from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise
from prime_rl.utils.sparse_update import apply_sparse_update, is_sparse_update_dir, to_compute_tensor

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem."""

    def init_broadcaster(self) -> None:
        """Initialize the broadcaster."""
        ...

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_path: str) -> None:
        """Update weights from a specified path in shared filesystem containing a HF-compatible checkpoint."""
        model = self._get_model()
        path = Path(weight_path)

        if is_sparse_update_dir(path):
            self._update_weights_from_sparse_update_patch(model, path)
            return

        weights_iterator = self._get_weights_iterator(model, path)
        load_weights_checkpoint_layerwise(
            model,
            weights_iterator,
            self.model_runner.model_config,
            self.vllm_config,
        )
        self._sparse_update_state_dict = None
        self._sparse_update_last_full_weight_path = path
        self._sparse_update_last_full_weight_step = self._extract_step(path)

    def _get_model(self) -> Module:
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)
        return model

    def _get_weights_iterator(self, model: Module, weight_path: Path | str):
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        revision = None
        if not Path(weight_path).exists():
            revision = getattr(self.model_runner.model_config, "revision", None)
        local_source = DefaultModelLoader.Source(
            str(weight_path),
            revision=revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        return model_loader._get_weights_iterator(local_source)

    def _update_weights_from_sparse_update_patch(self, model: Module, patch_dir: Path) -> None:
        self._ensure_sparse_update_state_dict(model)
        manifest = apply_sparse_update(
            self._sparse_update_state_dict,
            patch_dir,
            expected_base_step=self._sparse_update_step,
        )
        load_weights_checkpoint_layerwise(
            model,
            self._iter_sparse_update_state_dict(),
            self.model_runner.model_config,
            self.vllm_config,
        )
        self._sparse_update_step = manifest["step"]

    def _ensure_sparse_update_state_dict(self, model: Module) -> None:
        if hasattr(self, "_sparse_update_state_dict") and self._sparse_update_state_dict is not None:
            return

        source = getattr(self, "_sparse_update_last_full_weight_path", None)
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
        self._sparse_update_step = getattr(self, "_sparse_update_last_full_weight_step", None) or 0

    def _iter_sparse_update_state_dict(self) -> Iterable[tuple[str, torch.Tensor]]:
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
