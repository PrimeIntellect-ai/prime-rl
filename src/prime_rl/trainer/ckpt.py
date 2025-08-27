import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from prime_rl.trainer.config import CheckpointConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_ckpt_dir


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


class AppState(Stateful):
    """
    A wrapper for checkpointing the trainer with sharded weights and optimizer
    to allow resuming in any world size using torch.distributed.checkpoint
    utilities.

    https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    """

    def __init__(self, model: Module, optimizers: list[Optimizer], scheduler: LRScheduler, progress: Progress, dataloader_state: dict = None):
        self.model, self.optimizers, self.scheduler, self.progress = model, optimizers, scheduler, progress
        self.dataloader_state = dataloader_state

    def state_dict(self) -> dict[str, Any]:
        # Automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizers)
        scheduler_state_dict = self.scheduler.state_dict()
        progress_state_dict = asdict(self.progress)
        return {
            "model": model_state_dict,
            "optimizers": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
            "progress": progress_state_dict,
            "dataloader": self.dataloader_state
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        set_state_dict(
            self.model, self.optimizers, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optimizers"]
        )
        self.scheduler.load_state_dict(state_dict["scheduler"])
        for key, value in state_dict["progress"].items():
            setattr(self.progress, key, value)
        if "dataloader" in state_dict:
            self.dataloader_state = state_dict["dataloader"]


class CheckpointManager:
    """Utility class to save and load training checkpoints to resume training."""

    def __init__(self, output_dir: Path, config: CheckpointConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self._logger = get_logger()
        self._world = get_world()
        self._is_master = self._world.is_master
        self.ckpt_steps: list[int] = []  # Sorted list of steps that have been checkpointed, only used on master rank

    def _get_ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}"

    def _save_to_path(
        self,
        ckpt_path: Path,
        ckpt_step: int,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        dataloader_state: dict = None,
    ):
        self._logger.debug(f"Saving training checkpoint to {ckpt_path}")
        start_time = time.time()

        # Create checkpoint state
        state_dict = {"app": AppState(model, optimizers, scheduler, progress, dataloader_state)}

        # Save sharded state
        dcp.save(state_dict, checkpoint_id=ckpt_path)

        # Append to list of saved steps
        if self._is_master:
            self.ckpt_steps.append(ckpt_step)

        self._logger.debug(f"Training checkpoint saved in {time.time() - start_time:.2f} seconds")

    def _load_from_path(
        self, ckpt_path: Path, model: nn.Module, optimizers: list[Optimizer], scheduler: LRScheduler, progress: Progress, dataloader_state: dict = None
    ):
        """Loads a checkpoint from a given path in-place."""
        self._logger.debug(f"Loading training checkpoint from {ckpt_path}")
        start_time = time.time()

        # Load sharded state
        app_state = AppState(model, optimizers, scheduler, progress, dataloader_state)
        state_dict = {"app": app_state}
        dcp.load(state_dict=state_dict, checkpoint_id=ckpt_path)
        
        # Update dataloader_state in-place if provided and present in checkpoint
        if dataloader_state is not None and "dataloader" in app_state.state_dict():
            dataloader_state.clear()
            dataloader_state.update(app_state.state_dict()["dataloader"])

        self._logger.debug(f"Training checkpoint loaded in {time.time() - start_time:.2f} seconds")

    def load(
        self, model: nn.Module, optimizers: list[Optimizer], scheduler: LRScheduler, progress: Progress, step: int, dataloader_state: dict = None
    ) -> None:
        """Loads a checkpoint from a given path in-place."""
        ckpt_path = self._get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self._load_from_path(ckpt_path, model, optimizers, scheduler, progress, dataloader_state)

    def save(
        self,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        step: int,
        dataloader_state: dict = None,
    ) -> None:
        """Saves the full checkpoint state for a specified step."""
        ckpt_path = self._get_ckpt_path(step)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_to_path(ckpt_path, step, model, optimizers, scheduler, progress, dataloader_state)

    def maybe_clean(self) -> None:
        """Deletes past local checkpoints beyond the most recent `config.keep` steps. No-op if `config.keep` is None."""
        if self.config.keep is None:
            return

        # Get all the checkpoint steps to delete
        assert list(self.ckpt_steps) == sorted(self.ckpt_steps)
        ckpt_steps_to_delete = self.ckpt_steps[: -self.config.keep]
        for ckpt_step in ckpt_steps_to_delete:
            ckpt_path = self._get_ckpt_path(ckpt_step)
            if ckpt_path.exists():
                self._logger.debug(f"Removing past trainer checkpoint for step {ckpt_step} ({ckpt_path})")
                # TODO: Handle this more gracefully, e.g. each rank should only delete its own checkpoint
                shutil.rmtree(ckpt_path)

        # Update checkpoint steps
        self.ckpt_steps = self.ckpt_steps[-self.config.keep :]


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)
