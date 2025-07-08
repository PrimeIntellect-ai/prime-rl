import shutil
import threading
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from zeroband.training.config import CheckpointConfig, WeightCheckpointConfig
from zeroband.training.model import Model
from zeroband.training.world import get_world
from zeroband.utils.logger import get_logger


class WeightCheckpointManager:
    """Utility class to save and cleanup HF-compatible weight checkpoints."""

    def __init__(self, config: WeightCheckpointConfig, ckpt_config: CheckpointConfig, async_level: int):
        self.config = config
        self.ckpt_config = ckpt_config
        self.async_level = async_level
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
        return self.get_model_path(self.config.path, step)

    def _get_step_path(self, step: int) -> Path:
        return self.get_step_path(self.config.path, step)

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
        """Synchronous helper of `save`."""
        cpu_state = self._gather_weights(model, dtype)
        self._save_to_path(cpu_state, model, tokenizer, step)

    def save(
        self,
        model: Model,
        tokenizer: AutoTokenizer,
        step: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Save a HF-compatible weight-only checkpoint for a given step."""

        if self.config.save_async:
            thread = threading.Thread(
                target=self._save,
                args=(model, tokenizer, dtype, step),
                name=f"weight-checkpoint-save-{step}",
            )
            thread.start()
        else:
            self._save(model, tokenizer, dtype, step)
        return self._get_model_path(step)

    def _clean(self, step: int):
        """Synchronous helper of `clean`."""
        candidate_path_to_delete = self._get_step_path(step)
        self._logger.debug(f"Considering to delete weight checkpoint {candidate_path_to_delete}")
        keep_for_eval = self.config.interval and step % self.config.interval == 0
        # For checkpointing step x, we need all weight checkpoints in [x-async_level, x] (for logprob model)
        # To get [n-k, n] with interval n and buffer k over all natural numbers x, we use the condition (n - (x % n)) % n <= k
        keep_for_ckpt = (
            self.ckpt_config
            and (self.ckpt_config.interval - (step % self.ckpt_config.interval)) % self.ckpt_config.interval
            <= self.async_level
        )
        if not (keep_for_eval or keep_for_ckpt):
            self._logger.debug(
                f"Removing past weight checkpoint {candidate_path_to_delete} ({keep_for_eval=}, {keep_for_ckpt=})"
            )
            shutil.rmtree(candidate_path_to_delete, ignore_errors=True)

    def clean(self, step: int):
        """
        Considers deleting a weight checkpoint for a given step. There are two reasons not to delete a checkpoint:
        1. The step is an evaluation step (e.g. step % weights.interval == 0)
        2. The step is a checkpoint step or at most async_level steps earlier
        """
        if self.config.save_async:
            thread = threading.Thread(
                target=self._clean,
                args=(step,),
                name=f"weight-checkpoint-clean-{step}",
            )
            thread.start()
        else:
            self._clean(step)
