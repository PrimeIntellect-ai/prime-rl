import bisect
import gc
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PretrainedConfig, PreTrainedTokenizerBase, ProcessorMixin

from prime_rl.configs.shared import ResumeConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.lora import has_lora_layers
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.conversion_ops import view_mapping
from prime_rl.trainer.models.fusions import (
    join_loaded_optimizer_state_for_runtime,
    split_packed_optimizer_state_for_checkpoint,
    write_back_loaded_packed_optimizer_state,
)
from prime_rl.trainer.optim import OffloadOptimizer, OptimizerLike
from prime_rl.trainer.utils import get_dcp_group
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_ckpt_dir, get_step_path


@dataclass
class Progress:
    step: int = 1
    total_tokens: int = 0
    total_samples: int = 0


def save_model_files(
    model_config: PretrainedConfig,
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin | None,
    directory: Path,
) -> None:
    """Write the model's ``config.json``, the tokenizer files and a VLM's processor files into ``directory``, so that
    it loads as a model name for the architecture, the tokenizer and the processor."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(model_config.to_json_string(use_diff=False))
    # The processor saves its own tokenizer, which the configured one (pad token, chat template) overrides.
    if processor is not None:
        processor.save_pretrained(directory)
    tokenizer.save_pretrained(directory)


# Fields of a saved config.json that are metadata, not architecture.
NOT_ARCHITECTURE = {"transformers_version", "_name_or_path", "architectures", "dtype", "use_cache"}
MISSING = "(missing)"


def describe_change(key: str, old: Any, new: Any) -> str:
    """One field of an architecture mismatch: scalars as ``old -> new``, lists by their first differing index or
    their lengths, dicts by name."""
    if isinstance(old, dict) or isinstance(new, dict):
        return key
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return f"{key} ({len(old)} -> {len(new)} entries)"
        index = next(index for index, (a, b) in enumerate(zip(old, new)) if a != b)
        return f"{key} (differs from index {index})"
    old, new = (f"{len(value)} entries" if isinstance(value, list) else value for value in (old, new))
    return f"{key} {old} -> {new}"


def check_model_config(model_config: PretrainedConfig, directory: Path) -> None:
    """Fail when the checkpoint in ``directory`` was saved with another model architecture than ``model_config``. A
    checkpoint without a config.json (saved before this check, or copied without it) loads after a warning."""
    path = directory / "config.json"
    if not path.exists():
        if get_world().is_master:
            get_logger().warning(
                f"Checkpoint {directory} has no config.json; skipping the architecture check, so a resume with another "
                "architecture would load silently (checkpoints saved before this check have none)"
            )
        return
    saved = json.loads(path.read_text())
    current = json.loads(model_config.to_json_string(use_diff=False))
    # A field only one config has (e.g. from another transformers version) has the class default in the other.
    defaults = json.loads(type(model_config)().to_json_string(use_diff=False))
    changes = []
    for key in sorted((saved.keys() | current.keys()) - NOT_ARCHITECTURE):
        old, new = saved.get(key, MISSING), current.get(key, MISSING)
        if (defaults.get(key) if old == MISSING else old) != (defaults.get(key) if new == MISSING else new):
            changes.append(describe_change(key, old, new))
    if changes:
        more = f", +{len(changes) - 5} more" if len(changes) > 5 else ""
        raise ValueError(f"Checkpoint at {directory} has another architecture: {', '.join(changes[:5])}{more}")


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
            "optimizers": split_packed_optimizer_state_for_checkpoint(self.model, optimizer_state_dict),
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
            write_back_loaded_packed_optimizer_state(self.model, checkpoint_optimizers, state_dict["optimizers"])
            # The in-place load reaches tensors only: restore the param groups' values too (dion's Muon keeps its
            # step count there, so AdamW's bias correction would restart).
            loaded_groups = iter(state_dict["optimizers"]["param_groups"])
            for optimizer in checkpoint_optimizers:
                for group in optimizer.param_groups:
                    group.update({key: value for key, value in next(loaded_groups).items() if key != "params"})
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
                optim_state_dict=join_loaded_optimizer_state_for_runtime(
                    self.model, state_dict["optimizers"], runtime_optimizer_state_dict
                ),
            )

        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.progress is not None:
            for key, value in state_dict["progress"].items():
                setattr(self.progress, key, value)


class CheckpointManager:
    """Utility class to save and load trainer checkpoints to resume training. Every checkpoint step directory also holds
    the model's ``config.json``, the tokenizer files and a VLM's processor files, and a checkpoint with another
    architecture does not load."""

    def __init__(
        self,
        output_dir: Path,
        config: CheckpointConfig,
        model_config: PretrainedConfig,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin | None,
        resume: ResumeConfig | None = None,
    ):
        self.config = config
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.processor = processor
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
            torch.distributed.barrier(group=get_dcp_group())
            torch.save(dataloader.state_dict(), dataloader_dir / f"rank_{self.world.rank}.pt")

        # Before the sharded state, so the mapping is in place once .metadata appears
        self._save_mapping(path, model)
        dcp_save(state_dict, checkpoint_id=path, process_group=get_dcp_group())

        self.logger.debug(f"Saved training checkpoint in {format_time(time.perf_counter() - start_time)}")

    def _save_mapping(self, path: Path, model: nn.Module) -> None:
        """Write the prime -> HF conversion of the saved model weights as views, so a reader can
        serve the checkpoint in HF names without converting it."""
        if not isinstance(model, PreTrainedModelPrimeRL) or has_lora_layers(model):
            self.logger.info("Not writing mapping.json: only full-weight PrimeRL models declare their HF conversion")
            return
        shapes = {key: value.shape for key, value in get_model_state_dict(model).items()}
        tensors = view_mapping(model.conversion_chain(model.config), shapes)
        if tensors is None:
            self.logger.info("Not writing mapping.json: the conversion to HF is more than views")
            return
        if self.world.is_master:
            path.mkdir(parents=True, exist_ok=True)
            (path / "mapping.json").write_text(json.dumps({"prefix": "app.model.", "tensors": tensors}))

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

        _load_app_state(path, model, optimizers, scheduler, progress, model_only=self.skip_optimizer)

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

    def load_weights(self, path: Path, model: nn.Module, optimizers: list[OptimizerLike]) -> None:
        """Load only the model weights of the trainer checkpoint at ``path`` (in-place)."""
        if not path.exists():
            raise FileNotFoundError(f"model.init_from has no trainer checkpoint at {path}")
        self.logger.info(f"Initializing weights from {path}")
        start_time = time.perf_counter()
        _load_app_state(path, model, optimizers, scheduler=None, progress=None, model_only=True)
        self.logger.debug(f"Loaded weights in {format_time(time.perf_counter() - start_time)}")

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
        check_model_config(self.model_config, ckpt_path.parent)
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
            save_model_files(self.model_config, self.tokenizer, self.processor, ckpt_path.parent)
        torch.distributed.barrier(group=get_dcp_group())

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
    model_config: PretrainedConfig,
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin | None,
    resume: ResumeConfig | None = None,
) -> CheckpointManager:
    """The checkpoint manager always exists: ``resume`` decides whether it loads,
    ``ckpt`` whether it saves (a resume without ``ckpt`` loads but saves nothing)."""
    ckpt_output_dir = (ckpt_config.output_dir if ckpt_config else None) or output_dir
    return CheckpointManager(
        ckpt_output_dir, ckpt_config or CheckpointConfig(), model_config, tokenizer, processor, resume
    )


def _load_app_state(
    path: Path,
    model: nn.Module,
    optimizers: list[OptimizerLike],
    scheduler: LRScheduler | None,
    progress: Progress | None,
    *,
    model_only: bool,
) -> None:
    app_state = AppState(model, [] if model_only else optimizers, scheduler, progress)
    dcp_load(state_dict={"app": app_state}, checkpoint_id=path, process_group=get_dcp_group())
    # DCP materializes the optimizer state with a zero-grad step, which allocates a full set of
    # gradients. Left in the allocator cache, those blocks fragment the pool and the first step
    # reserves on top of them instead of reusing them.
    gc.collect()
    torch.cuda.empty_cache()
    if model_only:
        for optimizer in optimizers:
            if isinstance(optimizer, OffloadOptimizer):
                optimizer.finish_model_only_checkpoint_load()
