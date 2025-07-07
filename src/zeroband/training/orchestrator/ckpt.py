import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from zeroband.utils.logger import get_logger


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


class CheckpointManager:
    """Utility class to save and load orchestrator checkpoints to resume orchestrator."""

    def __init__(self, path: Path):
        self.path = path
        self._logger = get_logger()

    def _get_step_path(self, step: int) -> Path:
        return self.path / f"step_{step}"

    def _get_ckpt_path(self, step: int) -> Path:
        return self._get_step_path(step) / "orchestrator.pt"

    def _save_to_path(self, ckpt_path: Path, weight_ckpt_path: Path, progress: Progress):
        self._logger.debug(f"Saving orchestrator checkpoint to {ckpt_path}")
        start_time = time.time()

        # Increment the progress step that is going to be saved
        progress_copy = deepcopy(progress)
        progress_copy.step += 1

        # Create checkpoint state
        ckpt_state = {
            "weight_ckpt_path": weight_ckpt_path,
            "progress": progress,
        }

        # Save checkpoint state
        with open(ckpt_path, "wb") as f:
            torch.save(ckpt_state, f)

        self._logger.debug(f"Orchestrator checkpoint saved in {time.time() - start_time:.2f} seconds")

    def load_from_path(self, ckpt_path: Path, progress: Progress) -> Path:
        """Loads a checkpoint from a given path in-place."""
        self._logger.debug(f"Loading checkpoint from {ckpt_path}")
        start_time = time.time()

        # Load checkpoint state
        with open(ckpt_path, "rb") as f:
            state = torch.load(f, weights_only=False)

        # Load checkpoint state in-place
        progress.step = state["progress"].step
        progress.total_tokens = state["progress"].total_tokens
        progress.total_samples = state["progress"].total_samples
        progress.total_problems = state["progress"].total_problems

        self._logger.debug(f"Orchestrator checkpoint loaded in {time.time() - start_time:.2f} seconds")
        self._logger.info(f"Resuming from {progress=}")

        return state["weight_ckpt_path"]

    def save(
        self,
        weight_ckpt_path: Path,
        progress: Progress,
        step: int,
    ):
        """Saves the full checkpoint state for a specified step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._get_ckpt_path(step)
        self._save_to_path(ckpt_path, weight_ckpt_path, progress)


def get_ckpt_manager(path: Path) -> CheckpointManager:
    """Returns a checkpoint manager for a given checkpoint directory."""
    return CheckpointManager(path)
