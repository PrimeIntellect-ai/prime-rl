import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from prime_rl.trainer.config import CheckpointConfig
from prime_rl.trainer.model import Model
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_ckpt_dir


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


class CheckpointManager:
    """Utility class to save and load training checkpoints to resume training."""

    def __init__(self, outputs_dir: Path, config: CheckpointConfig):
        self.ckpt_dir = get_ckpt_dir(outputs_dir)
        self.save_async = config.save_async
        self._logger = get_logger()
        self._world = get_world()
        self._is_master = self._world.rank == 0
        self._keep = getattr(config, "keep", None)

    def _get_step_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}"

    def _get_ckpt_path(self, step: int) -> Path:
        ckpt_name = f"trainer_{self._world.local_rank}.pt" if self._world.world_size > 1 else "trainer.pt"
        return self._get_step_path(step) / ckpt_name

    def _save_to_path(
        self, ckpt_path: Path, model: Model, optimizers: list[Optimizer], scheduler: LRScheduler, progress: Progress
    ):
        self._logger.debug(f"Saving training checkpoint to {ckpt_path}")
        start_time = time.time()

        # Create checkpoint state
        ckpt_state = {
            "model": model.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "scheduler": scheduler.state_dict(),
            "progress": progress,
        }
        # Create checkpoint directory if it doesn't exist
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ckpt_path, "wb") as f:
            torch.save(ckpt_state, f)
        self._logger.debug(f"Training checkpoint saved in {time.time() - start_time:.2f} seconds")

    def _load_from_path(
        self, ckpt_path: Path, model: Model, optimizers: list[Optimizer], scheduler: LRScheduler, progress: Progress
    ):
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
        scheduler.load_state_dict(state["scheduler"])

        # Load progress
        for key, value in asdict(state["progress"]).items():
            setattr(progress, key, value)

        self._logger.debug(f"Training checkpoint loaded in {time.time() - start_time:.2f} seconds")

    def load(
        self, model: Model, optimizers: list[Optimizer], scheduler: LRScheduler, progress: Progress, step: int
    ) -> None:
        """Loads a checkpoint from a given path in-place."""
        ckpt_path = self._get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self._load_from_path(ckpt_path, model, optimizers, scheduler, progress)

    def _cleanup_old_checkpoints(self):
        if not self._keep:
            return
        # Only master rank should perform cleanup to avoid races between ranks
        if not getattr(self, "_is_master", True):
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
                self._logger.debug(f"Removing past full checkpoint {path}")
                # Remove directory tree safely
                import shutil

                shutil.rmtree(path, ignore_errors=True)
        except Exception as e:
            self._logger.warning(f"Failed to cleanup old checkpoints: {e}")

    def save(
        self,
        model: Model,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        step: int,
    ):
        """Saves the full checkpoint state for a specified step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._get_ckpt_path(step)

        if self.save_async:
            # Run save in a separate thread
            thread = threading.Thread(
                target=self._save_to_path,
                args=(ckpt_path, model, optimizers, scheduler, progress),
                name=f"ckpt-save-{step}",
            )
            thread.start()
        else:
            # Run save synchronously
            self._save_to_path(ckpt_path, model, optimizers, scheduler, progress)

        # Cleanup old checkpoints after saving
        self._cleanup_old_checkpoints()
