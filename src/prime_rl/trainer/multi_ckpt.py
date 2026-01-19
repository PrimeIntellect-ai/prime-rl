"""Multi-run checkpointing for RL training.

MultiCheckpointManager owns per-run CheckpointManagers and AppStates,
each saving to its own run directory.
"""

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful

from prime_rl.trainer.ckpt import CheckpointManager
from prime_rl.trainer.config import CheckpointConfig
from prime_rl.trainer.runs import Progress, get_runs
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import resolve_latest_ckpt_step

if TYPE_CHECKING:
    from prime_rl.trainer.optim import MultiLoRAOptimizer
    from prime_rl.trainer.scheduler import MultiLoRAScheduler


class RunState(Stateful):
    """Per-run state wrapper - just like AppState but for adapter weights."""

    def __init__(
        self,
        model_state_dict: dict[str, Any],
        optimizer,
        scheduler,
        progress: Progress,
    ):
        self.model_state_dict = model_state_dict
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.progress = progress

    def state_dict(self) -> dict[str, Any]:
        state = {
            "model": self.model_state_dict,
            "progress": asdict(self.progress),
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Load adapter weights
        for key, value in state_dict["model"].items():
            if key in self.model_state_dict:
                self.model_state_dict[key].copy_(value)
        # Load optimizer
        if "optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        # Load scheduler
        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        # Don't load progress because it resets packers count
        # There will be a step issue if we load progress


class MultiCheckpointManager:
    """Owns per-run CheckpointManagers and AppStates."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.runs = get_runs()
        self.world = get_world()
        self.logger = get_logger()
        self.managers: list[CheckpointManager | None] = [None] * self.runs.max_runs
        self.runs.register_deletion_hook(self._run_deletion_hook)
        self.runs.register_creation_hook(self._run_creation_hook)

    def _run_deletion_hook(self, idx: int, run_id: str) -> None:
        self.managers[idx] = None

    def _run_creation_hook(self, idx: int, run_id: str) -> None:
        self.managers[idx] = self._maybe_create_manager(idx)

    def _maybe_create_manager(self, idx: int) -> CheckpointManager | None:
        ckpt_config = self.runs.config[idx].ckpt
        if ckpt_config is None:
            return None

        config = CheckpointConfig(
            interval=ckpt_config.interval,
            keep_last=ckpt_config.keep_last,
            keep_interval=ckpt_config.keep_interval,
        )
        run_dir = self.runs.get_run_dir(idx)
        manager = CheckpointManager(run_dir, config)
        self.managers[idx] = manager
        return manager

    def _should_save(self, idx: int, step: int) -> bool:
        """Determine if a checkpoint should be saved for a given run and step."""
        ckpt_config = self.runs.config[idx].ckpt
        if ckpt_config is None or ckpt_config.interval is None:
            return False
        if step <= 0 or step % ckpt_config.interval != 0:
            return False
        # Check if already saved this step
        return step not in self.managers[idx].ckpt_steps

    def save(
        self,
        optimizer: "MultiLoRAOptimizer",
        scheduler: "MultiLoRAScheduler",
    ) -> None:
        for idx in self.runs.used_idxs:
            step = self.runs.progress[idx].step
            if not self._should_save(idx, step):
                continue

            manager = self.managers[idx]

            # We have a very wide try-except because we dont want to crash the trainer over one run having issues
            try:
                model_state_dict = {k: v.data.detach().clone() for k, v in self.runs.get_named_parameters_for_run(idx)}
                run_state = RunState(
                    model_state_dict,
                    optimizer.optimizers[idx],
                    scheduler.schedulers[idx],
                    self.runs.progress[idx],
                )
                ckpt_path = manager.get_ckpt_path(step)
                ckpt_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(
                    f"Saving checkpoint for run {idx} at step {step} to {ckpt_path / f'rank_{self.world.rank}.pt'}"
                )
                torch.save(run_state.state_dict(), ckpt_path / f"rank_{self.world.rank}.pt")

                # Copy broadcast folder to checkpoint
                # This way, we only need to save the checkpoint folder
                if self.world.is_master:
                    run_dir = self.runs.get_run_dir(idx)
                    broadcast_src = run_dir / "broadcasts" / f"step_{step}"
                    weight_dst = run_dir / "checkpoints" / f"step_{step}" / "weight"
                    try:
                        shutil.copytree(broadcast_src, weight_dst)
                    except FileNotFoundError:
                        self.logger.error(
                            f"Broadcast folder not found for run {idx} at step {step}. Looking for it in {broadcast_src}"
                        )
            except FileNotFoundError:
                self.logger.warning(f"Run {idx} deleted during checkpoint, skipping")
            except Exception as e:
                self.logger.error(f"Error checkpointing run {idx}: {e}")
            dist.barrier()
            manager.mark_stable(step)
            manager.ckpt_steps.append(step)
        dist.barrier()

    def load_run(
        self,
        idx: int,
        optimizer: "MultiLoRAOptimizer",
        scheduler: "MultiLoRAScheduler",
    ) -> bool:
        if self.runs.config[idx].ckpt is None or self.runs.config[idx].ckpt.resume_step is None:
            return False

        manager = self.managers[idx]
        if manager is None:
            return False

        step = self.runs.config[idx].ckpt.resume_step
        if step == -1:
            step = resolve_latest_ckpt_step(manager.ckpt_dir)
            if step is None:
                return False

        try:
            model_state_dict = dict(self.runs.get_named_parameters_for_run(idx))
            run_state = RunState(
                model_state_dict,
                optimizer.optimizers[idx],
                scheduler.schedulers[idx],
                self.runs.progress[idx],
            )
            ckpt_path = manager.get_ckpt_path(step)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            self.logger.info(f"Loading checkpoint from {ckpt_path}")
            state_dict = torch.load(ckpt_path / f"rank_{self.world.rank}.pt")
            run_state.load_state_dict(state_dict)

            self.logger.info(f"Resumed run {self.runs.idx_2_id[idx]} from step {step}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading checkpoint for run {idx}: {e}")
            return False

    def maybe_clean(self) -> None:
        if not self.world.is_master:
            return
        for idx in self.runs.used_idxs:
            if self.managers[idx] is None:
                continue
            self.managers[idx].maybe_clean()


def setup_multi_checkpoint_manager(output_dir: Path) -> tuple[MultiCheckpointManager, None]:
    return MultiCheckpointManager(output_dir), None
