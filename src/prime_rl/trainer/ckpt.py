import shutil
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.config import CheckpointConfig, LoRAConfig, WeightCheckpointConfig
from prime_rl.trainer.lora import has_lora_layers, save_lora_config
from prime_rl.trainer.weights import (
    convert_tt_to_hf_moe,
    gather_weights_on_master,
    get_adapter_state_dict,
    has_tt_moe_layers,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.tensor_hashing import get_module_signature, get_optimizer_signature
from prime_rl.utils.utils import get_ckpt_dir, get_step_path, get_weights_dir


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
        self.logger = get_logger()
        self.world = get_world()
        self.ckpt_steps: list[int] = []  # Sorted list of steps that have been checkpointed, only used on master rank
        self.save_thread: threading.Thread | None = None

    def get_ckpt_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}" / "trainer"

    def get_latest_step(self) -> int:
        step_dirs = list(self.ckpt_dir.glob("step_*"))
        if len(step_dirs) == 0:
            raise ValueError(f"No checkpoints found in {self.ckpt_dir}")
        steps = sorted([int(step_dir.name.split("_")[-1]) for step_dir in step_dirs])
        latest_step = steps[-1]
        self.logger.info(f"Found latest checkpoint in {self.ckpt_dir}: {latest_step}")
        return latest_step

    def save_to_path(
        self,
        path: Path,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        dataloader: StatefulDataLoader | None = None,
    ):
        self.logger.debug(f"Saving training checkpoint to {path}")
        start_time = time.time()

        # Create checkpoint state
        state_dict = {"app": AppState(model, optimizers, scheduler, progress)}

        # Checkpoint the local dataloader
        if dataloader is not None:
            dataloader_dir = path / "dataloader"
            dataloader_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dataloader.state_dict(), dataloader_dir / f"rank_{self.world.rank}.pt")

        # Save sharded state
        dcp_save(state_dict, checkpoint_id=path)

        self.logger.debug(f"Training checkpoint saved in {time.time() - start_time:.2f} seconds")

    def load_from_path(
        self,
        path: Path,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler | None,
        progress: Progress | None,
        dataloader: StatefulDataLoader | None = None,
    ):
        """Loads a checkpoint from a given path in-place."""
        self.logger.debug(f"Loading training checkpoint from {path}")
        start_time = time.time()

        # Load sharded state
        app_state = AppState(model, optimizers, scheduler, progress)
        state_dict = {"app": app_state}
        dcp_load(state_dict=state_dict, checkpoint_id=path)

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
            dataloader.load_state_dict(torch.load(dataloader_path))

        self.logger.debug(f"Training checkpoint loaded in {time.time() - start_time:.2f} seconds")

    def load(
        self,
        step: int,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler | None,
        progress: Progress | None,
        dataloader: StatefulDataLoader | None = None,
    ) -> None:
        """Loads a checkpoint from a given path in-place."""
        if step == -1:
            step = self.get_latest_step()

        ckpt_path = self.get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self.load_from_path(ckpt_path, model, optimizers, scheduler, progress, dataloader)
        self.logger.debug(
            f"Signatures after loading training checkpoint: model={get_module_signature(model, compress=True)}, optimizers={', '.join(get_optimizer_signature(optimizer, compress=True) for optimizer in optimizers)}"
        )

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler,
        progress: Progress,
        dataloader: StatefulDataLoader | None = None,
    ) -> None:
        """Saves the full checkpoint state for a specified step."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.debug(
            f"Signatures before saving training checkpoint: model={get_module_signature(model, compress=True)}, optimizers={', '.join(get_optimizer_signature(optimizer, compress=True) for optimizer in optimizers)}"
        )

        def save_ckpt_and_update_ckpt_steps():
            """Save checkpoint to disk and update checkpoint steps"""
            self.save_to_path(ckpt_path, model, optimizers, scheduler, progress, dataloader)
            if self.world.is_master:
                self.ckpt_steps.append(step)

        if self.config.save_async:
            self.wait_for_thread()
            assert self.save_thread is None
            self.save_thread = threading.Thread(
                target=save_ckpt_and_update_ckpt_steps,
                name=f"checkpoint-save-{step}",
            )
            self.save_thread.start()
        else:
            save_ckpt_and_update_ckpt_steps()
        torch.distributed.barrier()

    def maybe_clean(self) -> None:
        """Deletes past checkpoints beyond the most recent config.keep steps. No-op if config.keep is None."""
        if self.config.keep is None:
            return

        # Get all the checkpoint steps to delete
        assert list(self.ckpt_steps) == sorted(self.ckpt_steps)
        ckpt_steps_to_delete = self.ckpt_steps[: -self.config.keep]
        for ckpt_step in ckpt_steps_to_delete:
            trainer_ckpt_path = self.get_ckpt_path(ckpt_step)
            ckpt_path = trainer_ckpt_path.parent
            if ckpt_path.exists():
                self.logger.debug(f"Removing past checkpoint for step {ckpt_step} ({ckpt_path})")
                shutil.rmtree(ckpt_path)

        # Update checkpoint steps
        self.ckpt_steps = self.ckpt_steps[-self.config.keep :]

    def wait_for_thread(self):
        if self.save_thread is None:
            return
        # Don't try to join if we're in the save thread itself
        if threading.current_thread() is self.save_thread:
            return
        self.save_thread.join()
        self.save_thread = None


class WeightCheckpointManager:
    """Utility class to save and cleanup HF-compatible weight checkpoints."""

    def __init__(
        self,
        output_dir: Path,
        config: WeightCheckpointConfig,
        lora_config: LoRAConfig | None = None,
        save_async: bool = False,
        keep: int | None = None,
    ):
        self.weights_dir = get_weights_dir(output_dir)
        self.config = config
        self.lora_config = lora_config
        self.save_async = save_async
        self.logger = get_logger()
        self.world = get_world()
        self.ckpt_steps: list[int] = []  # Sorted list of steps that have been checkpointed, only used on master rank
        self.save_thread: threading.Thread | None = None
        self.keep = keep

    def get_step_path(self, step: int) -> Path:
        return get_step_path(self.weights_dir, step)

    def save_lora_adapters(self, lora_state: dict[str, Tensor], model: nn.Module, step: int):
        """Save LoRA adapters to separate directory."""
        adapter_path = self.get_step_path(step) / "lora_adapters"
        adapter_path.mkdir(parents=True, exist_ok=True)

        torch.save(lora_state, adapter_path / "adapter_model.bin")

        if self.lora_config:
            save_lora_config(self.lora_config, model, adapter_path)  # Pass model

        self.logger.debug(f"Saved LoRA adapters to {adapter_path}")

    def save_to_path(
        self,
        path: Path,
        state_dict: dict[str, Tensor],
        model,
        tokenizer,
    ):
        """Save weight checkpoint for given step."""
        # Save weight checkpoint temporary dir to avoid race condition
        path.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Saving weight checkpoint to {path}")
        start_time = time.time()
        # Suppress torch.distributed warnings during checkpoint saving
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

            # Save weights
            save_state_dict(state_dict, path, self.config.save_format, self.config.save_sharded)

            # Save model config, generation arguments and tokenizer
            model.config.save_pretrained(path)
            if model.generation_config:
                model.generation_config.save_pretrained(path)
            tokenizer.save_pretrained(path)

        self.logger.debug(f"Saved weight checkpoint to {path} in {time.time() - start_time:.2f} seconds")

    def save(
        self,
        step: int,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Save a HF-compatible weight-only checkpoint for a given step."""

        # Save LoRA adapters separately if configured
        has_lora = has_lora_layers(model)
        if self.config.save_adapter_separately and has_lora:
            if self.world.is_master:
                lora_state = get_adapter_state_dict(model, self.world.is_master)
                self.save_lora_adapters(lora_state, model, step)
            torch.distributed.barrier()

        step_path = self.get_step_path(step)
        state_dict = gather_weights_on_master(model, self.world.is_master, dtype, has_lora_layers=has_lora)
        if has_tt_moe_layers(state_dict):
            convert_tt_to_hf_moe(state_dict)

        def save_weights_and_update_ckpt_steps():
            """Save weight checkpoint"""
            self.save_to_path(step_path, state_dict, model, tokenizer)
            if self.world.is_master:
                self.ckpt_steps.append(step)

        if self.world.is_master:
            if self.save_async:
                self.wait_for_thread()
                assert self.save_thread is None
                self.save_thread = threading.Thread(
                    target=save_weights_and_update_ckpt_steps,
                    name=f"weight-checkpoint-save-{step}",
                )
                self.save_thread.start()
            else:
                save_weights_and_update_ckpt_steps()
        torch.distributed.barrier()

    def maybe_clean(self) -> None:
        """Deletes past checkpoints beyond the most recent config.keep steps. No-op if config.keep is None."""
        if self.keep is None:
            return

        # Get all the checkpoint steps to delete
        assert list(self.ckpt_steps) == sorted(self.ckpt_steps)
        ckpt_steps_to_delete = self.ckpt_steps[: -self.keep]
        for ckpt_step in ckpt_steps_to_delete:
            ckpt_path = self.get_step_path(ckpt_step)
            if ckpt_path.exists():
                self.logger.debug(f"Removing past checkpoint for step {ckpt_step} ({ckpt_path})")
                shutil.rmtree(ckpt_path)

        # Update checkpoint steps
        self.ckpt_steps = self.ckpt_steps[-self.keep :]

    def wait_for_thread(self):
        if self.save_thread is None:
            return
        # Don't try to join if we're in the save thread itself
        if threading.current_thread() is self.save_thread:
            return
        self.save_thread.join()
        self.save_thread = None


def setup_ckpt_managers(
    output_dir: Path, ckpt_config: CheckpointConfig | None, lora_config: LoRAConfig | None = None
) -> tuple[CheckpointManager | None, WeightCheckpointManager | None]:
    if ckpt_config is None:
        return None, None
    ckpt_manager = CheckpointManager(output_dir, ckpt_config)
    if ckpt_config.weights:
        weight_ckpt_manager = WeightCheckpointManager(
            output_dir,
            ckpt_config.weights,
            lora_config=lora_config,
            save_async=ckpt_config.save_async,
            keep=ckpt_config.keep,
        )
    else:
        weight_ckpt_manager = None
    return ckpt_manager, weight_ckpt_manager
