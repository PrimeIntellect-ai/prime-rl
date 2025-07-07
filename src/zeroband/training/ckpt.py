import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim.optimizer import Optimizer

from zeroband.training.model import Model
from zeroband.training.world import get_world
from zeroband.utils.logger import get_logger


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


class CheckpointManager:
    """Utility class to save and load training checkpoints to resume training."""

    def __init__(self, path: Path):
        self.path = path
        self._logger = get_logger()
        self._world = get_world()

    def _get_step_path(self, step: int) -> Path:
        return self.path / f"step_{step}"

    def _get_ckpt_path(self, step: int) -> Path:
        ckpt_name = f"trainer_{self._world.local_rank}.pt" if self._world.world_size > 1 else "trainer.pt"
        return self._get_step_path(step) / ckpt_name

    def _save_to_path(self, ckpt_path: Path, model: Model, optimizers: list[Optimizer], progress: Progress):
        self._logger.debug(f"Saving training checkpoint to {ckpt_path}")
        start_time = time.time()

        # Increment the progress step that is going to be saved
        progress_copy = deepcopy(progress)
        progress_copy.step += 1

        # Create checkpoint state
        ckpt_state = {
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "progress": progress_copy,
        }
        # Create checkpoint directory if it doesn't exist
        with open(ckpt_path, "wb") as f:
            torch.save(ckpt_state, f)
        self._logger.debug(f"Training checkpoint saved in {time.time() - start_time:.2f} seconds")

    def load_from_path(self, ckpt_path: Path, model: Model, optimizers: list[Optimizer], progress: Progress):
        """Loads a checkpoint from a given path in-place."""
        self._logger.debug(f"Loading training checkpoint from {ckpt_path}")
        start_time = time.time()

        # Load checkpoint state
        with open(ckpt_path, "rb") as f:
            state = torch.load(f, weights_only=False)

        # Load checkpoint state in-place
        model.load_state_dict(state["model"])
        for optimizer, optimizer_state in zip(optimizers, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)

        # Load progress
        progress.total_tokens = state["progress"].total_tokens
        progress.step = state["progress"].step
        progress.total_samples = state["progress"].total_samples

        self._logger.debug(f"Training checkpoint loaded in {time.time() - start_time:.2f} seconds")
        self._logger.info(f"Resuming from {progress=}")

    def save(
        self,
        model: Model,
        optimizers: list[Optimizer],
        progress: dict,
        step: int,
    ):
        """Saves the full checkpoint state for a specified step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._get_ckpt_path(step)
        self._save_to_path(ckpt_path, model, optimizers, progress)


def get_ckpt_manager(path: Path) -> CheckpointManager:
    """Returns a checkpoint manager for a given checkpoint directory."""
    return CheckpointManager(path)
