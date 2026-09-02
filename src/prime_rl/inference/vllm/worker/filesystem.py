from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.model_executor.model_loader import DefaultModelLoader, get_model_loader

from prime_rl.inference.vllm.worker.weight_transfer import load_weights_checkpoint_layerwise
from prime_rl.trainer.sparse_update import apply_sparse_update

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem."""

    def init_broadcaster(self) -> None:
        """Initialize sparse-update receiver state."""
        self._sparse_state_dict: dict[str, torch.Tensor] | None = None
        self._sparse_step = 0
        self._sparse_base_path: str | None = None

    def liveness_probe(self) -> None:
        """No-op RPC used by the API server liveness endpoint."""
        return None

    def update_weights_from_path(self, weight_path: str) -> None:
        """Load a full checkpoint or apply a sparse HF patch chain."""
        if not hasattr(self, "_sparse_step"):
            self.init_broadcaster()
        path = Path(weight_path)
        if (path / "sparse_manifest.json").exists():
            self._ensure_sparse_cache()
            self._sparse_step = apply_sparse_update(self._sparse_state_dict, path, expected_base_step=self._sparse_step)
            model = (
                self.model_runner.model.runnable
                if hasattr(self.model_runner.model, "runnable")
                else self.model_runner.model
            )
            load_weights_checkpoint_layerwise(
                model,
                self._sparse_state_dict.items(),
                self.model_runner.model_config,
                self.vllm_config,
            )
            return

        model = (
            self.model_runner.model.runnable
            if hasattr(self.model_runner.model, "runnable")
            else self.model_runner.model
        )
        assert isinstance(model, Module)
        weights_iterator = self._weights_iterator(model, path)
        load_weights_checkpoint_layerwise(model, weights_iterator, self.model_runner.model_config, self.vllm_config)
        self._sparse_state_dict = None
        self._sparse_base_path = weight_path
        self._sparse_step = self._extract_step(path) or 0

    def _weights_iterator(self, model: Module, path: Path):
        model_loader = get_model_loader(self.load_config)
        assert isinstance(model_loader, DefaultModelLoader)
        source = DefaultModelLoader.Source(
            str(path),
            revision=None,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        return model_loader._get_weights_iterator(source)

    def _ensure_sparse_cache(self) -> None:
        if self._sparse_state_dict is not None:
            return
        model = (
            self.model_runner.model.runnable
            if hasattr(self.model_runner.model, "runnable")
            else self.model_runner.model
        )
        source = Path(self._sparse_base_path or self.model_runner.model_config.model)
        self._sparse_state_dict = {
            name: tensor.detach().to("cpu", dtype=torch.bfloat16).contiguous()
            for name, tensor in self._weights_iterator(model, source)
        }
        if "lm_head.weight" not in self._sparse_state_dict and "model.embed_tokens.weight" in self._sparse_state_dict:
            self._sparse_state_dict["lm_head.weight"] = self._sparse_state_dict["model.embed_tokens.weight"].clone()

    @staticmethod
    def _extract_step(path: Path) -> int | None:
        for parent in (path, *path.parents):
            if parent.name.startswith("step_"):
                try:
                    return int(parent.name.removeprefix("step_"))
                except ValueError:
                    return None
        return None
