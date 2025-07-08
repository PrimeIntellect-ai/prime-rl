import threading
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from zeroband.training.config import WeightCheckpointConfig
from zeroband.training.model import Model
from zeroband.training.world import get_world
from zeroband.utils.logger import get_logger


class WeightCheckpointManager:
    """Utility class to save and load weight HF-compatible weight checkpoints."""

    def __init__(self, config: WeightCheckpointConfig):
        self.path = config.path
        self.async_save = config.save_async
        self._logger = get_logger()
        self._world = get_world()
        self._is_master = self._world.rank == 0

    @classmethod
    def get_step_path(cls, weight_dir: Path, step: int) -> Path:
        return weight_dir / f"step_{step}"

    @classmethod
    def get_model_path(cls, weight_dir: Path, step: int) -> Path:
        return cls.get_step_path(weight_dir, step) / "pytorch_model.bin"

    def _get_model_path(self, step: int) -> Path:
        return self.get_model_path(self.path, step)

    def _get_step_path(self, step: int) -> Path:
        return self.get_step_path(self.path, step)

    def _gather_weights(self, model: Model, dtype: torch.dtype = torch.bfloat16) -> dict[str, Tensor]:
        """Gather distributed weights for weight checkpoint."""
        start_time = time.time()
        self._logger.debug("Gathering sharded weights")

        cpu_state = {}
        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                value = value.to(dtype)
                # only gather after the downcast to dtype as it will be faster
                value = value.full_tensor()

            if self._is_master:
                key = get_fqns(model, key)
                assert len(key) == 1
                key = next(iter(key))
                # TODO(Sami) Blocking to avoid race condition, should make non-blocking long-term tho
                cpu_state[key] = value.to("cpu", non_blocking=False)

        torch.distributed.barrier()
        self._logger.debug(f"Gathered sharded weights in {time.time() - start_time:.2f} seconds")

        return cpu_state

    def _save_to_path(self, cpu_state: dict[str, Tensor], model: Model, tokenizer: AutoTokenizer, step: int):
        """Save weight checkpoint for given step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)

        self._logger.debug(f"Saving weight checkpoint to {step_path}")
        start_time = time.time()

        # Save model weights to temporary file to avoid race condition
        model_path = self._get_model_path(step)
        tmp_model_path = model_path.with_suffix(".tmp")
        torch.save(cpu_state, tmp_model_path)
        # Rename temporary file to indicate checkpoint is complete
        tmp_model_path.rename(model_path)

        # Save model config, generation arguments and tokenizer
        model.config.save_pretrained(step_path)
        model.generation_config.save_pretrained(step_path)
        tokenizer.save_pretrained(step_path)

        self._logger.debug(f"Saved weight checkpoint to {step_path} in {time.time() - start_time:.2f} seconds")

    def _save(self, model: Model, tokenizer: AutoTokenizer, dtype: torch.dtype, step: int):
        cpu_state = self._gather_weights(model, dtype)
        self._save_to_path(cpu_state, model, tokenizer, step)

    def save(
        self,
        model: Model,
        tokenizer: AutoTokenizer,
        step: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Save a HF-compatible weight-only checkpoint to the specified path."""

        if self.async_save:
            thread = threading.Thread(
                target=self._save,
                args=(model, tokenizer, dtype, step),
                name=f"weight-checkpoint-save-{step}",
            )
            thread.start()
        else:
            self._save(model, tokenizer, dtype, step)
        return self._get_model_path(step)
