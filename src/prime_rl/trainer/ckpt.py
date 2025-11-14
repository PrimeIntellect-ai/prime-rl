import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.trainer.config import CheckpointConfig
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.tensor_hashing import get_module_signature, get_optimizer_signature
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

    def __init__(
        self,
        model: Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler | None,
        progress: Progress | None,
    ):
        self.model = model
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.progress = progress

    def state_dict(self) -> dict[str, Any]:
        # Automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizers)
        state_dict = {
            "model": model_state_dict,
            "optimizers": optimizer_state_dict,
        }
        if self.scheduler is not None:
            scheduler_state_dict = self.scheduler.state_dict()
            state_dict["scheduler"] = scheduler_state_dict
        if self.progress is not None:
            progress_state_dict = asdict(self.progress)
            state_dict["progress"] = progress_state_dict
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        set_state_dict(self.model, [], model_state_dict=state_dict["model"], optim_state_dict=state_dict["optimizers"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.progress is not None:
            for key, value in state_dict["progress"].items():
                setattr(self.progress, key, value)


class CheckpointManager:
    """Utility class to save and load training checkpoints to resume training."""

    def __init__(self, output_dir: Path, config: CheckpointConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self._logger = get_logger()
        self._world = get_world()
        self._is_master = self._world.is_master
        self.ckpt_steps: list[int] = []  # Sorted list of steps that have been checkpointed, only used on master rank

    def get_step_path(self, step: int) -> Path:
        """Returns the path to the step directory (containing both orchestrator and trainer checkpoints)."""
        return self.ckpt_dir / f"step_{step}"

    def get_ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}" / "trainer"

    def _is_complete_checkpoint(self, step: int) -> bool:
        """Checks if a checkpoint step is complete (both orchestrator and trainer checkpoints exist)."""
        step_path = self.get_step_path(step)
        orchestrator_path = step_path / "orchestrator"
        trainer_path = step_path / "trainer"
        return orchestrator_path.exists() and trainer_path.exists()

    def get_latest_step(self) -> int:
        step_dirs = list(self.ckpt_dir.glob("step_*"))
        if len(step_dirs) == 0:
            raise ValueError(f"No checkpoints found in {self.ckpt_dir}")
        steps = sorted([int(step_dir.name.split("_")[-1]) for step_dir in step_dirs])
        latest_step = steps[-1]
        self._logger.info(f"Found latest checkpoint in {self.ckpt_dir}: {latest_step}")
        return latest_step

    def _save_to_path(
        self,
        ckpt_path: Path,
        ckpt_step: int,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        dataloader: StatefulDataLoader | None = None,
    ):
        self._logger.debug(f"Saving training checkpoint to {ckpt_path}")
        start_time = time.time()

        # Create checkpoint state
        state_dict = {"app": AppState(model, optimizers, scheduler, progress)}

        # Checkpoint the local dataloader
        if dataloader is not None:
            dataloader_dir = ckpt_path / "dataloader"
            dataloader_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dataloader.state_dict(), dataloader_dir / f"rank_{self._world.rank}.pt")

        # Save sharded state
        dcp_save(state_dict, checkpoint_id=ckpt_path)

        # Append to list of saved steps
        if self._is_master:
            self.ckpt_steps.append(ckpt_step)

        self._logger.debug(f"Training checkpoint saved in {time.time() - start_time:.2f} seconds")

    def _load_from_path(
        self,
        ckpt_path: Path,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler | None,
        progress: Progress | None,
        dataloader: StatefulDataLoader | None = None,
    ):
        """Loads a checkpoint from a given path in-place."""
        self._logger.debug(f"Loading training checkpoint from {ckpt_path}")
        start_time = time.time()

        # Load sharded state
        app_state = AppState(model, optimizers, scheduler, progress)
        state_dict = {"app": app_state}
        dcp_load(state_dict=state_dict, checkpoint_id=ckpt_path)

        # Load the dataloader
        if dataloader is not None:
            dataloader_path = ckpt_path / "dataloader" / f"rank_{self._world.rank}.pt"
            if not dataloader_path.exists():
                self._logger.warning(
                    f"Did not find local dataloader checkpoint at path {dataloader_path}. This might be because you tried restarting the trainer with a different world size. Falling back to using the master rank's dataloader checkpoint. Note, that this may cause training inconsistencies."
                )
                dataloader_path = ckpt_path / "dataloader" / "rank_0.pt"
                if not dataloader_path.exists():
                    raise RuntimeError(
                        f"Couldn't fallback to using the master rank's dataloader checkpoint, because dataloder checkpoint was not found at path {dataloader_path}. Cannot resume training."
                    )
            dataloader.load_state_dict(torch.load(dataloader_path))

        self._logger.debug(f"Training checkpoint loaded in {time.time() - start_time:.2f} seconds")

    def load(
        self,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler | None,
        progress: Progress | None,
        step: int,
        dataloader: StatefulDataLoader | None = None,
    ) -> None:
        """Loads a checkpoint from a given path in-place."""
        if step == -1:
            step = self.get_latest_step()

        ckpt_path = self.get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self._load_from_path(ckpt_path, model, optimizers, scheduler, progress, dataloader)
        self._logger.debug(
            f"Signatures after loading training checkpoint: model={get_module_signature(model, compress=True)}, optimizers={', '.join(get_optimizer_signature(optimizer, compress=True) for optimizer in optimizers)}"
        )

    def save(
        self,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        step: int,
        dataloader: StatefulDataLoader | None = None,
    ) -> None:
        """Saves the full checkpoint state for a specified step."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger.debug(
            f"Signatures before saving training checkpoint: model={get_module_signature(model, compress=True)}, optimizers={', '.join(get_optimizer_signature(optimizer, compress=True) for optimizer in optimizers)}"
        )
        self._save_to_path(ckpt_path, step, model, optimizers, scheduler, progress, dataloader)

    def maybe_clean(self) -> None:
        """
        Deletes past checkpoints beyond the most recent `config.keep` complete checkpoints.
        The trainer checkpoint manager is responsible for cleanup since it lags behind the orchestrator.
        This ensures we always have complete checkpoints (both orchestrator and trainer).
        No-op if `config.keep` is None.
        """
        if self.config.keep is None:
            return

        # Only master rank maintains the list of checkpoint steps and determines what to delete
        if not self._is_master:
            # Non-master ranks need to know which steps to delete
            # We'll broadcast this information so all ranks can participate in cleanup
            ckpt_steps_to_delete = []
        else:
            assert self.ckpt_steps == sorted(self.ckpt_steps)

            # Get the last `config.keep` steps (these are the "kept" steps)
            kept_steps = (
                self.ckpt_steps[-self.config.keep :] if len(self.ckpt_steps) >= self.config.keep else self.ckpt_steps
            )

            # Find the newest complete checkpoint among the kept steps
            newest_complete_step = None
            for step in reversed(kept_steps):
                if self._is_complete_checkpoint(step):
                    newest_complete_step = step
                    break

            # If there's no complete checkpoint in the kept steps, don't delete anything
            # This prevents deleting checkpoints when we don't have a complete one to keep
            if newest_complete_step is None:
                if self.ckpt_steps:
                    self._logger.debug(
                        f"Skipping cleanup: no complete checkpoint found in kept steps {kept_steps}. "
                        "Waiting for a complete checkpoint before cleaning up."
                    )
                return

            # Find all steps older than the newest complete checkpoint
            steps_older_than_complete = [step for step in self.ckpt_steps if step < newest_complete_step]

            # Among those older steps, find the complete ones and keep the last `config.keep` of them
            older_complete_steps = [step for step in steps_older_than_complete if self._is_complete_checkpoint(step)]
            if len(older_complete_steps) > self.config.keep:
                kept_older_complete_steps = older_complete_steps[-self.config.keep :]
            else:
                kept_older_complete_steps = older_complete_steps

            # Steps to keep: newest_complete_step + kept_older_complete_steps
            steps_to_keep = {newest_complete_step} | set(kept_older_complete_steps)

            # Delete all steps that are not in steps_to_keep
            ckpt_steps_to_delete = [step for step in self.ckpt_steps if step not in steps_to_keep]

        # Broadcast the list of steps to delete to all ranks
        # Note: broadcast_object_list mutates the list in-place and returns None
        if dist.is_initialized():
            ckpt_steps_to_delete_list = [ckpt_steps_to_delete]
            dist.broadcast_object_list(ckpt_steps_to_delete_list, src=0)
            ckpt_steps_to_delete = ckpt_steps_to_delete_list[0]
        # If distributed is not initialized, only master will have the list (single-process case)

        if not ckpt_steps_to_delete:
            # Update checkpoint steps (only on master) before returning
            if self._is_master:
                self.ckpt_steps = self.ckpt_steps[-self.config.keep :]
            return

        # Each rank deletes its own dataloader checkpoint files
        for ckpt_step in ckpt_steps_to_delete:
            ckpt_path = self.get_ckpt_path(ckpt_step)
            dataloader_file = ckpt_path / "dataloader" / f"rank_{self._world.rank}.pt"
            if dataloader_file.exists():
                self._logger.debug(
                    f"Removing past dataloader checkpoint for step {ckpt_step}, rank {self._world.rank} ({dataloader_file})"
                )
                dataloader_file.unlink()

        # Synchronize all ranks before master deletes the rest
        if dist.is_initialized():
            dist.barrier()

        # Master rank deletes the entire step directory (including orchestrator checkpoint if it exists)
        # This is safe because the trainer lags behind, so if we're deleting a step, the orchestrator
        # has already moved on to newer steps
        if self._is_master:
            for ckpt_step in ckpt_steps_to_delete:
                step_path = self.get_step_path(ckpt_step)
                if step_path.exists():
                    self._logger.debug(
                        f"Removing past checkpoint step {ckpt_step} (including orchestrator and trainer) ({step_path})"
                    )
                    # Remove the entire step directory (orchestrator + trainer)
                    shutil.rmtree(step_path, ignore_errors=True)

        # Update checkpoint steps (only on master)
        if self._is_master:
            # Remove deleted steps from the list
            self.ckpt_steps = [step for step in self.ckpt_steps if step not in ckpt_steps_to_delete]


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)
