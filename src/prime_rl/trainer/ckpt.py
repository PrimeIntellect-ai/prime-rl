import bisect
import gc
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_model_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.shared import ResumeConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.models.fusions import (
    load_packed_optimizer_state,
    optimizer_state_dict_for_checkpoint,
    optimizer_state_dict_for_runtime,
)
from prime_rl.trainer.optim import OffloadOptimizer, OptimizerLike
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.utils import get_all_ckpt_steps, get_ckpt_dir, get_step_path


@dataclass
class Progress:
    step: int = 1
    total_tokens: int = 0
    total_samples: int = 0


def _try_rmtree(path: Path, logger) -> None:
    """Remove a directory tree, logging and skipping on failure."""
    try:
        shutil.rmtree(path)
    except OSError as e:
        logger.warning(f"Failed to remove {path}: {e}, skipping cleanup")


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
        optimizers: list[OptimizerLike],
        scheduler: LRScheduler | None,
        progress: Progress | None,
    ):
        self.model = model
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.progress = progress

    def _get_checkpoint_optimizers(self) -> list[Optimizer]:
        """Expose optimizers keyed by their model parameters for DCP."""
        return [
            optimizer.checkpoint_optimizer() if isinstance(optimizer, OffloadOptimizer) else optimizer
            for optimizer in self.optimizers
        ]

    def _has_cpu_offload(self) -> bool:
        return any(isinstance(optimizer, OffloadOptimizer) for optimizer in self.optimizers)

    def state_dict(self) -> dict[str, Any]:
        for optimizer in self.optimizers:
            if isinstance(optimizer, OffloadOptimizer):
                optimizer.prepare_checkpoint_save()

        # Automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        checkpoint_optimizers = self._get_checkpoint_optimizers()
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, checkpoint_optimizers)
        # Runtime fusions own one physical state tensor per fused parameter; the checkpoint
        # sees it under the canonical names. This dict is also the template dcp_load writes
        # into, and load_state_dict packs the loaded entries back into the runtime state.
        state_dict = {
            "model": model_state_dict,
            "optimizers": optimizer_state_dict_for_checkpoint(self.model, optimizer_state_dict),
        }
        if self.scheduler is not None:
            scheduler_state_dict = self.scheduler.state_dict()
            state_dict["scheduler"] = scheduler_state_dict
        if self.progress is not None:
            progress_state_dict = asdict(self.progress)
            state_dict["progress"] = progress_state_dict

        for optimizer in self.optimizers:
            if isinstance(optimizer, OffloadOptimizer):
                optimizer.finish_checkpoint_save()

        if self._has_cpu_offload():
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        checkpoint_optimizers = self._get_checkpoint_optimizers()
        has_cpu_offload = self._has_cpu_offload()

        if has_cpu_offload:
            # When CPU offload is on, the optimizer is already loaded by the time we
            # get here: state_dict() handed dcp_load a template whose tensors share
            # storage with optim.state[p][k], and dcp_load wrote the checkpoint bytes
            # directly into those tensors via target_tensor.copy_(...). Running
            # set_state_dict on the optimizer would route the loaded CPU values
            # through Optimizer.load_state_dict, whose _cast hook does
            # value.to(param.dtype, param.device) and would allocate a fresh GPU
            # copy of every state tensor — undoing the in-place CPU load and
            # detaching optim.state from the tensors we just populated. So we only
            # apply the model side here and flip the wrappers to initialized so
            # subsequent steps take the steady-state path.
            set_model_state_dict(self.model, model_state_dict=state_dict["model"])
            # The template only aliases a packed parameter's state where the packing
            # dimension is unsharded, so the loaded logical entries are packed back in.
            load_packed_optimizer_state(self.model, checkpoint_optimizers, state_dict["optimizers"])
            for optimizer in self.optimizers:
                if isinstance(optimizer, OffloadOptimizer):
                    optimizer.finish_checkpoint_load()
        else:
            # The runtime state dict read back here keys the fused parameters' state by their
            # physical names; the loaded logical entries are packed into it.
            _, runtime_optimizer_state_dict = get_state_dict(self.model, checkpoint_optimizers)
            set_state_dict(
                self.model,
                checkpoint_optimizers,
                model_state_dict=state_dict["model"],
                optim_state_dict=optimizer_state_dict_for_runtime(
                    self.model, state_dict["optimizers"], runtime_optimizer_state_dict
                ),
            )

        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.progress is not None:
            for key, value in state_dict["progress"].items():
                setattr(self.progress, key, value)

        # state_dict is the same dict object that dcp_load held internally; clearing
        # it drops the last references to the loaded tensor wrappers so the cuda
        # allocator can release whatever blocks it cached during the read.
        if has_cpu_offload:
            state_dict.clear()
            gc.collect()
            torch.cuda.empty_cache()


class CheckpointManager:
    """Utility class to save and load trainer checkpoints to resume SFT and RL training."""

    def __init__(self, output_dir: Path, config: CheckpointConfig, resume: ResumeConfig | None = None):
        self.config = config
        self.skip_optimizer = config.skip_optimizer
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.logger = get_logger()
        self.world = get_world()

        all_steps = get_all_ckpt_steps(self.ckpt_dir)
        if resume is not None and resume.step is not None:
            self.ckpt_steps = [s for s in all_steps if s <= resume.step]
        else:
            self.ckpt_steps = all_steps

    def get_ckpt_path(self, step: int) -> Path:
        """Get the path to write the trainer checkpoint for a given step."""
        return get_step_path(self.ckpt_dir, step) / "trainer"

    def save_to_path(
        self,
        path: Path,
        model: nn.Module,
        optimizers: list[OptimizerLike],
        scheduler: LRScheduler,
        progress: Progress,
        dataloader: StatefulDataLoader | None = None,
    ):
        """Save the trainer checkpoint to a given path."""
        self.logger.debug(f"Saving training checkpoint to {path}")
        start_time = time.perf_counter()

        # Create checkpoint state
        state_dict = {"app": AppState(model, optimizers, scheduler, progress)}

        # Checkpoint the local dataloader
        if dataloader is not None:
            dataloader_dir = path / "dataloader"
            # Only the master creates the dir; the rest wait at a barrier. On a
            # parallel FS (beegfs), concurrent mkdir from every rank can re-raise
            # FileExistsError (EEXIST + stale is_dir() metadata).
            if self.world.is_master:
                dataloader_dir.mkdir(parents=True, exist_ok=True)
            torch.distributed.barrier()
            torch.save(dataloader.state_dict(), dataloader_dir / f"rank_{self.world.rank}.pt")

        # Save sharded state
        dcp_save(state_dict, checkpoint_id=path)

        self.logger.debug(f"Saved training checkpoint in {format_time(time.perf_counter() - start_time)}")

    def load_from_path(
        self,
        path: Path,
        model: nn.Module,
        optimizers: list[OptimizerLike],
        scheduler: LRScheduler | None,
        progress: Progress | None,
        dataloader: StatefulDataLoader | None = None,
    ):
        """Load the trainer checkpoint from a given path (in-place)."""
        self.logger.debug(f"Loading training checkpoint from {path}")
        start_time = time.perf_counter()

        # Load sharded state
        app_state = AppState(model, optimizers if not self.skip_optimizer else [], scheduler, progress)
        state_dict = {"app": app_state}
        dcp_load(state_dict=state_dict, checkpoint_id=path)
        if self.skip_optimizer:
            for optimizer in optimizers:
                if isinstance(optimizer, OffloadOptimizer):
                    optimizer.finish_model_only_checkpoint_load()

        # Load the dataloader
        if dataloader is not None:
            dataloader_path = path / "dataloader" / f"rank_{self.world.rank}.pt"
            if not dataloader_path.exists():
                self.logger.warning(
                    f"Did not find local dataloader checkpoint at path {dataloader_path}. This might be because you tried restarting the trainer with a different world size. Falling back to using the master rank's dataloader checkpoint. Note, that this may cause training inconsistencies."
                )
                dataloader_path = path / "dataloader" / "rank_0.pt"
                if not dataloader_path.exists():
                    raise RuntimeError(
                        f"Couldn't fallback to using the master rank's dataloader checkpoint, because dataloder checkpoint was not found at path {dataloader_path}. Cannot resume training."
                    )
            dataloader.load_state_dict(torch.load(dataloader_path, weights_only=False))

        self.logger.debug(f"Loaded training checkpoint in {format_time(time.perf_counter() - start_time)}")

    def load(
        self,
        step: int,
        model: nn.Module,
        optimizers: list[OptimizerLike],
        scheduler: LRScheduler | None,
        progress: Progress | None,
        dataloader: StatefulDataLoader | None = None,
        path: Path | None = None,
    ) -> None:
        """Load the trainer checkpoint for a given step (in-place). ``path`` overrides
        where the checkpoint is read from (an external run's ``step_<N>/trainer``)."""
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self.load_from_path(ckpt_path, model, optimizers, scheduler, progress, dataloader)

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizers: list[OptimizerLike],
        scheduler: LRScheduler,
        progress: Progress,
        dataloader: StatefulDataLoader | None = None,
    ) -> None:
        """Save the full checkpoint state for a specified step."""
        ckpt_path = self.get_ckpt_path(step)
        # Master-only mkdir + barrier: concurrent mkdir from every rank can
        # re-raise FileExistsError on a parallel FS (see save_to_path).
        if self.world.is_master:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        self.save_to_path(ckpt_path, model, optimizers, scheduler, progress, dataloader)
        bisect.insort(self.ckpt_steps, step)

    def maybe_clean(self) -> None:
        """Deletes past checkpoints based on keep_last and keep_interval policies. No-op if both are None."""
        if self.config.keep_last is None and self.config.keep_interval is None:
            return

        # Get all the checkpoint steps to delete
        assert list(self.ckpt_steps) == sorted(self.ckpt_steps)

        # Determine which steps to keep
        steps_to_keep = set()

        # Keep the most recent keep_last steps
        if self.config.keep_last is not None:
            steps_to_keep.update(self.ckpt_steps[-self.config.keep_last :])

        # Keep steps at keep_interval intervals
        if self.config.keep_interval is not None:
            for step in self.ckpt_steps:
                if step % self.config.keep_interval == 0:
                    steps_to_keep.add(step)

        # Delete steps not in steps_to_keep (only master rank deletes to avoid race condition)
        ckpt_steps_to_delete = [step for step in self.ckpt_steps if step not in steps_to_keep]
        if self.world.is_master:
            for ckpt_step in ckpt_steps_to_delete:
                trainer_ckpt_path = self.get_ckpt_path(ckpt_step)
                ckpt_path = trainer_ckpt_path.parent
                if ckpt_path.exists():
                    self.logger.debug(f"Removing past checkpoint for step {ckpt_step} ({ckpt_path})")
                    _try_rmtree(ckpt_path, self.logger)

        # Update checkpoint steps
        self.ckpt_steps = [step for step in self.ckpt_steps if step in steps_to_keep]


def setup_ckpt_manager(
    output_dir: Path,
    ckpt_config: CheckpointConfig | None,
    resume: ResumeConfig | None = None,
) -> CheckpointManager:
    """The checkpoint manager always exists: ``resume`` decides whether it loads,
    ``ckpt`` whether it saves (a resume without ``ckpt`` loads but saves nothing)."""
    ckpt_output_dir = (ckpt_config.output_dir if ckpt_config else None) or output_dir
    return CheckpointManager(ckpt_output_dir, ckpt_config or CheckpointConfig(), resume)
