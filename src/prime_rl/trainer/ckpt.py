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
        self.config = config
        self.ckpt_dir = get_ckpt_dir(outputs_dir)
        self._logger = get_logger()
        self._world = get_world()
        self._is_master = self._world.rank == 0
        self.ckpt_steps: list[int] = []  # Sorted list of steps that have been checkpointed, only used on master rank

    def _get_step_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}"

    def _get_ckpt_path(self, step: int) -> Path:
        ckpt_name = f"trainer_{self._world.local_rank}.pt" if self._world.world_size > 1 else "trainer.pt"
        return self._get_step_path(step) / ckpt_name

    def _save_to_path(
        self,
        ckpt_path: Path,
        ckpt_step: int,
        model: Model,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
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

        # Append to list of saved steps
        if self._is_master:
            self.ckpt_steps.append(ckpt_step)

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

    def save(
        self,
        model: Model,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        step: int,
    ) -> None:
        """Saves the full checkpoint state for a specified step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._get_ckpt_path(step)

        if self.config.save_async:
            # Run save in a separate thread
            thread = threading.Thread(
                target=self._save_to_path,
                args=(ckpt_path, step, model, optimizers, scheduler, progress),
                name=f"ckpt-save-{step}",
            )
            thread.start()
        else:
            # Run save synchronously
            self._save_to_path(ckpt_path, step, model, optimizers, scheduler, progress)

    def maybe_clean(self) -> None:
        """Deletes past local checkpoints beyond the most recent `config.keep` steps. No-op if `config.keep` is None."""
        if self.config.keep is None:
            return

        # Get all the checkpoint steps to delete
        assert list(self.ckpt_steps) == sorted(self.ckpt_steps)
        ckpt_steps_to_keep = self.ckpt_steps[-self.config.keep :]
        ckpt_steps_to_delete = self.ckpt_steps[: -self.config.keep]
        for ckpt_step in ckpt_steps_to_delete:
            ckpt_path = self._get_ckpt_path(ckpt_step)
            if ckpt_path.exists():
                self._logger.debug(
                    f"Removing past trainer checkpoint for step {ckpt_step} ({ckpt_path}), because got checkpoints for {ckpt_steps_to_keep} ({len(self.ckpt_steps)} > {self.config.keep})"
                )
                ckpt_path.unlink(missing_ok=True)

        # Update checkpoint steps
        self.ckpt_steps = self.ckpt_steps[-self.config.keep :]
