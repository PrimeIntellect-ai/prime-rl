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
        self.keep = config.keep
        self._saved_steps: list[int] = []
        self._lock = threading.Lock()

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
        # Record saved step after successful write
        try:
            step_str = ckpt_path.parent.name.split("_")[-1]
            step_num = int(step_str)
            with self._lock:
                self._saved_steps.append(step_num)
                self._saved_steps.sort()
        except Exception:
            # Best-effort: ignore if parsing fails
            pass
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

    def maybe_clean(self, step: int):
        """Delete this rank's past trainer checkpoints beyond the most recent `keep` steps.
        No-op if `keep` is None. Only executes on master rank to avoid races.
        """
        if self.keep is None:
            return
        if not self._is_master:
            return
        try:
            with self._lock:
                # Ensure sorted ascending
                self._saved_steps.sort()
                # Determine how many to delete
                num_to_delete = max(0, len(self._saved_steps) - self.keep)
                steps_to_delete = [self._saved_steps.pop(0) for _ in range(num_to_delete)]
            # Delete only this rank's trainer checkpoint file for those steps
            for old_step in steps_to_delete:
                path = self._get_ckpt_path(old_step)
                if path.exists():
                    self._logger.debug(f"Removing past trainer checkpoint {path}")
                    try:
                        path.unlink(missing_ok=True)
                    except TypeError:
                        # For Python versions without missing_ok
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass
        except Exception as e:
            self._logger.warning(f"Failed to cleanup old trainer checkpoints: {e}")

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
