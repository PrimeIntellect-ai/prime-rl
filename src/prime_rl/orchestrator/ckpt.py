import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from prime_rl.orchestrator.config import CheckpointConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_ckpt_dir


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


class CheckpointManager:
    """Utility class to save and load orchestrator checkpoints to resume orchestrator."""

    def __init__(self, outputs_dir: Path, config: CheckpointConfig):
        self.ckpt_dir = get_ckpt_dir(outputs_dir)
        self._logger = get_logger()
        self._keep = getattr(config, "keep", None)

    def _get_step_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}"

    def _get_ckpt_path(self, step: int) -> Path:
        return self._get_step_path(step) / "orchestrator.pt"

    def _save_to_path(self, ckpt_path: Path, progress: Progress):
        self._logger.debug(f"Saving orchestrator checkpoint to {ckpt_path}")
        start_time = time.time()

        # Create checkpoint state
        ckpt_state = {"progress": progress}

        # Save checkpoint state
        with open(ckpt_path, "wb") as f:
            torch.save(ckpt_state, f)

        self._logger.debug(f"Orchestrator checkpoint saved in {time.time() - start_time:.2f} seconds")

    def _load_from_path(self, ckpt_path: Path, progress: Progress) -> None:
        """Loads a checkpoint from a given path in-place."""
        self._logger.debug(f"Loading checkpoint from {ckpt_path}")
        start_time = time.time()

        # Load checkpoint state
        with open(ckpt_path, "rb") as f:
            state = torch.load(f, weights_only=False)

        # Load checkpoint state in-place
        for key, value in asdict(state["progress"]).items():
            setattr(progress, key, value)

        self._logger.debug(f"Orchestrator checkpoint loaded in {time.time() - start_time:.2f} seconds")

    def _cleanup_old_checkpoints(self):
        if not self._keep:
            return
        try:
            # Collect step directories of the form step_<int>
            step_dirs = []
            if self.ckpt_dir.exists():
                for child in self.ckpt_dir.iterdir():
                    if child.is_dir() and child.name.startswith("step_"):
                        try:
                            step_num = int(child.name.split("_")[-1])
                            step_dirs.append((step_num, child))
                        except ValueError:
                            continue
            # Sort by step number descending (newest first)
            step_dirs.sort(key=lambda x: x[0], reverse=True)
            # Determine which to delete beyond the first `keep`
            to_delete = step_dirs[self._keep :]
            for step_num, path in to_delete:
                self._logger.debug(f"Removing past orchestrator checkpoint {path}")
                import shutil

                shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            self._logger.warning(f"Failed to cleanup old orchestrator checkpoints: {e}")

    def load(self, progress: Progress, step: int) -> Path:
        """Loads a checkpoint from a given path."""
        ckpt_path = self._get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self._load_from_path(ckpt_path, progress)

    def save(
        self,
        progress: Progress,
        step: int,
    ):
        """Saves the full checkpoint state for a specified step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._get_ckpt_path(step)
        self._save_to_path(ckpt_path, progress)

        # Cleanup old checkpoints after saving
        self._cleanup_old_checkpoints()
