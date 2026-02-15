import bisect
import shutil
import time
import warnings
from dataclasses import asdict
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
from transformers.core_model_loading import revert_weight_conversion
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.config import CheckpointConfig, LoRAConfig, WeightCheckpointConfig
from prime_rl.trainer.lora import has_lora_layers, save_lora_config
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.optim import CPUOffloadOptimizer
from prime_rl.trainer.runs import Progress
from prime_rl.trainer.weights import (
    gather_weights_on_master,
    get_adapter_state_dict,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.tensor_hashing import get_module_signature, get_optimizer_signature
from prime_rl.utils.utils import get_all_ckpt_steps, get_ckpt_dir, get_step_path, get_weights_dir


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
        optimizers: list[Optimizer],
        scheduler: LRScheduler | None,
        progress: Progress | None,
    ):
        self.model = model
        self.optimizers = optimizers
        self.scheduler = scheduler
        self.progress = progress

    def _get_base_optimizers(self) -> list[Optimizer]:
        """Extract base optimizers from wrappers like CPUOffloadOptimizer."""
        return [opt.base_optimizer if isinstance(opt, CPUOffloadOptimizer) else opt for opt in self.optimizers]

    def state_dict(self) -> dict[str, Any]:
        # Move CPU-offloaded states to GPU before checkpointing
        for opt in self.optimizers:
            if isinstance(opt, CPUOffloadOptimizer) and opt._initialized:
                opt._move_states("cuda")
                torch.cuda.synchronize()

        # Automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        base_optimizers = self._get_base_optimizers()
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, base_optimizers)
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

        # Move states back to CPU
        for opt in self.optimizers:
            if isinstance(opt, CPUOffloadOptimizer) and opt._initialized:
                opt._move_states("cpu")

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        base_optimizers = self._get_base_optimizers()
        set_state_dict(
            self.model, base_optimizers, model_state_dict=state_dict["model"], optim_state_dict=state_dict["optimizers"]
        )

        # Re-initialize CPU offload wrappers after loading
        for opt in self.optimizers:
            if isinstance(opt, CPUOffloadOptimizer):
                opt._move_states("cpu")
                opt._initialized = True

        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.progress is not None:
            for key, value in state_dict["progress"].items():
                setattr(self.progress, key, value)


class CheckpointManager:
    """Utility class to save and load trainer checkpoints to resume SFT and RL training."""

    def __init__(self, output_dir: Path, config: CheckpointConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.logger = get_logger()
        self.world = get_world()

        all_steps = get_all_ckpt_steps(self.ckpt_dir)
        if config.resume_step is not None and config.resume_step >= 0:
            self.ckpt_steps = [s for s in all_steps if s <= config.resume_step]
        else:
            self.ckpt_steps = all_steps

    def get_ckpt_path(self, step: int) -> Path:
        """Get the path to write the trainer checkpoint for a given step."""
        return get_step_path(self.ckpt_dir, step) / "trainer"

    def mark_stable(self, step: int) -> None:
        """Write STABLE file to indicate checkpoint is complete (for eval to safely read)."""
        if self.world.is_master:
            step_path = get_step_path(self.ckpt_dir, step)
            (step_path / "STABLE").touch()

    def save_to_path(
        self,
        path: Path,
        model: nn.Module,
        optimizers: list[Optimizer],
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
            dataloader_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dataloader.state_dict(), dataloader_dir / f"rank_{self.world.rank}.pt")

        # Save sharded state
        dcp_save(state_dict, checkpoint_id=path)

        self.logger.debug(f"Training checkpoint saved in {time.perf_counter() - start_time:.2f} seconds")

    def load_from_path(
        self,
        path: Path,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler | None,
        progress: Progress | None,
        dataloader: StatefulDataLoader | None = None,
    ):
        """Load the trainer checkpoint from a given path (in-place)."""
        self.logger.debug(f"Loading training checkpoint from {path}")
        start_time = time.perf_counter()

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

        self.logger.debug(f"Training checkpoint loaded in {time.perf_counter() - start_time:.2f} seconds")

    def load(
        self,
        step: int,
        model: nn.Module,
        optimizers: list[Optimizer],
        scheduler: LRScheduler | None,
        progress: Progress | None,
        dataloader: StatefulDataLoader | None = None,
    ) -> None:
        """Load the trainer checkpoint for a given step (in-place)."""
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
        """Save the full checkpoint state for a specified step."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.debug(
            f"Signatures before saving training checkpoint: model={get_module_signature(model, compress=True)}, optimizers={', '.join(get_optimizer_signature(optimizer, compress=True) for optimizer in optimizers)}"
        )

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


class WeightCheckpointManager:
    """Utility class to save HF-compatible weight checkpoints."""

    def __init__(
        self,
        output_dir: Path,
        config: WeightCheckpointConfig,
        lora_config: LoRAConfig | None = None,
        save_async: bool = False,
        keep_last: int | None = None,
        keep_interval: int | None = None,
        resume_step: int | None = None,
    ):
        self.weights_dir = get_weights_dir(output_dir)
        self.config = config
        self.lora_config = lora_config
        self.logger = get_logger()
        self.world = get_world()
        if self.world.is_master:
            all_steps = get_all_ckpt_steps(self.weights_dir)
            if resume_step is not None and resume_step >= 0:
                self.ckpt_steps = [s for s in all_steps if s <= resume_step]
            else:
                self.ckpt_steps = all_steps
        else:
            self.ckpt_steps = []
        self.keep_last = keep_last
        self.keep_interval = keep_interval

    def get_step_path(self, step: int) -> Path:
        """Get the path to write the weight checkpoint for a given step."""
        return get_step_path(self.weights_dir, step)

    def save_to_path(
        self,
        path: Path,
        state_dict: dict[str, Tensor],
        lora_state_dict: dict[str, Tensor] | None,
        model,
        tokenizer: PreTrainedTokenizer,
    ):
        """Save HF-compatible weight checkpoint to a given path."""
        path.mkdir(parents=True, exist_ok=True)
        start_time = time.perf_counter()

        self.logger.debug(f"Saving weight checkpoint to {path}")
        # Suppress torch.distributed warnings during checkpoint saving
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
            warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

            # Save weights
            save_state_dict(state_dict, path, self.config.save_format, self.config.save_sharded)

            # Save model config, generation arguments and tokenizer
            model.config.save_pretrained(path)
            if model.generation_config:
                # training sets use_cache=False which can conflict with
                # cache_implementation — save with use_cache=True without
                # mutating the model's config
                from copy import deepcopy

                gen_config = deepcopy(model.generation_config)
                gen_config.use_cache = True
                gen_config.save_pretrained(path)
            tokenizer.save_pretrained(path)

        if self.config.save_adapter_separately and lora_state_dict is not None:
            adapter_path = path / "lora_adapters"
            adapter_path.mkdir(parents=True, exist_ok=True)
            torch.save(lora_state_dict, adapter_path / "adapter_model.bin")
            if self.lora_config:
                save_lora_config(
                    model,
                    adapter_path,
                    rank=self.lora_config.rank,
                    alpha=self.lora_config.alpha,
                    dropout=self.lora_config.dropout,
                )
        self.logger.debug(f"Saved weight checkpoint to {path} in {time.perf_counter() - start_time:.2f} seconds")

    def save(
        self,
        step: int,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
    ):
        """Save a HF-compatible weight-only checkpoint for a given step."""
        step_path = self.get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)

        # Gather all weights on master rank
        self.logger.debug("Gathering weights on master rank for weight checkpoint")
        start_time = time.perf_counter()
        state_dict = gather_weights_on_master(model, self.world.is_master, dtype=torch.bfloat16)
        self.logger.debug(f"Gathered weights on master rank in {time.perf_counter() - start_time:.2f} seconds")

        # Remove tied weight keys to match original model format
        if getattr(model.config, "tie_word_embeddings", False):
            for key in getattr(model, "_tied_weights_keys", []):
                state_dict.pop(key, None)

        if has_lora_layers(model):
            self.logger.debug("Getting LoRA state dict on master rank for weight checkpoint")
            start_time = time.perf_counter()
            lora_state_dict = get_adapter_state_dict(model, self.world.is_master)
            self.logger.debug(f"Got LoRA state dict on master rank in {time.perf_counter() - start_time:.2f} seconds")
        else:
            lora_state_dict = None

        # Convert to HF hub format if needed
        if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
            self.logger.debug("Converting PrimeRL format to HF format for weight checkpoint")
            start_time = time.perf_counter()
            model.convert_to_hf(state_dict)
            state_dict = revert_weight_conversion(model, state_dict)
            self.logger.debug(
                f"Converted PrimeRL format to HF format in {time.perf_counter() - start_time:.2f} seconds"
            )
        else:
            # For regular transformers models, revert internal format to original HF hub format
            self.logger.debug("Reverting transformers internal format to HF hub format for weight checkpoint")
            start_time = time.perf_counter()
            state_dict = revert_weight_conversion(model, state_dict)
            self.logger.debug(f"Reverted to HF hub format in {time.perf_counter() - start_time:.2f} seconds")

        # Save weight checkpoint on master rank
        if self.world.is_master:
            self.save_to_path(step_path, state_dict, lora_state_dict, model, tokenizer)
            # Write STABLE file to indicate checkpoint is complete (for eval to safely read)
            (step_path / "STABLE").touch()
        bisect.insort(self.ckpt_steps, step)

    def maybe_clean(self) -> None:
        """Deletes past checkpoints based on keep_last and keep_interval policies. No-op if both are None."""
        if self.keep_last is None and self.keep_interval is None:
            return

        # Get all the checkpoint steps to delete
        assert list(self.ckpt_steps) == sorted(self.ckpt_steps)

        # Determine which steps to keep
        steps_to_keep = set()

        # Keep the most recent keep_last steps
        if self.keep_last is not None:
            steps_to_keep.update(self.ckpt_steps[-self.keep_last :])

        # Keep steps at keep_interval intervals
        if self.keep_interval is not None:
            for step in self.ckpt_steps:
                if step % self.keep_interval == 0:
                    steps_to_keep.add(step)

        # Delete steps not in steps_to_keep (only master rank deletes to avoid race condition)
        ckpt_steps_to_delete = [step for step in self.ckpt_steps if step not in steps_to_keep]
        if self.world.is_master:
            for ckpt_step in ckpt_steps_to_delete:
                ckpt_path = self.get_step_path(ckpt_step)
                if ckpt_path.exists():
                    self.logger.debug(f"Removing past checkpoint for step {ckpt_step} ({ckpt_path})")
                    _try_rmtree(ckpt_path, self.logger)

        # Update checkpoint steps
        self.ckpt_steps = [step for step in self.ckpt_steps if step in steps_to_keep]


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
            keep_last=ckpt_config.keep_last,
            keep_interval=ckpt_config.keep_interval,
            resume_step=ckpt_config.resume_step,
        )
    else:
        weight_ckpt_manager = None
    return ckpt_manager, weight_ckpt_manager
