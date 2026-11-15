import bisect
import copy
import gc
import json
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor, Shard
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from prime_rl.configs.shared import ResumeConfig
from prime_rl.configs.trainer import CheckpointConfig
from prime_rl.trainer.lora import has_lora_layers
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.conversion_ops import view_mapping
from prime_rl.trainer.models.fusions import (
    get_model_packed_parameters,
    join_loaded_optimizer_state_for_runtime,
    split_packed_optimizer_state_for_checkpoint,
    write_back_loaded_packed_optimizer_state,
)
from prime_rl.trainer.optim import OffloadOptimizer, OptimizerLike
from prime_rl.trainer.utils import DCP_TIMEOUT, empty_pinned, get_dcp_group
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_ckpt_dir, get_step_path, is_ckpt_step_complete


@dataclass
class Progress:
    step: int = 1
    total_tokens: int = 0
    total_samples: int = 0


@dataclass
class Staging:
    """A copy to pinned host memory in flight: the copied state dict, the event recorded after the copies, and the
    source state dict, kept alive (with its GPU temporaries) until the copies are done."""

    state_dict: dict[str, Any]
    copied: torch.cuda.Event
    sources: dict[str, Any] | None

    def wait(self) -> dict[str, Any]:
        self.copied.synchronize()
        self.sources = None
        return self.state_dict


@dataclass
class PendingSave:
    """An async save whose write to storage may still be running."""

    step: int
    staging: Staging | None
    """The copy to host memory; None once it is known to be done."""
    write: Future
    start_time: float
    end_time: float = 0.0

    def record_end_time(self, _: Future) -> None:
        self.end_time = time.perf_counter()


class PinnedStager:
    """Copies a state dict into pinned host buffers that are kept and reused across saves (one per tensor path).
    GPU tensors are copied on a side stream, so the copy can run alongside the next step. CPU tensors are copied
    synchronously. Non-tensor values are deep-copied, so later updates do not reach the checkpoint. DCP's
    ``DefaultStager`` cannot stage this trainer's DTensors ("invalid python storage"), so DTensors are staged through
    their local shard and rewrapped with their spec."""

    def __init__(self):
        self.buffers: dict[tuple, torch.Tensor] = {}
        self.stream = torch.cuda.Stream()

    def stage(self, state_dict: dict[str, Any]) -> Staging:
        # The side stream must see everything the training stream wrote, including the optimizer step
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            staged = self._copy(state_dict, ())
        copied = torch.cuda.Event()
        copied.record(self.stream)
        return Staging(staged, copied, state_dict)

    def _copy(self, value: Any, path: tuple) -> Any:
        if isinstance(value, DTensor):
            return DTensor(self._copy_tensor(value._local_tensor, path), value._spec, requires_grad=False)
        if isinstance(value, torch.Tensor):
            return self._copy_tensor(value, path)
        if isinstance(value, dict):
            return {key: self._copy(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, list):
            return [self._copy(item, (*path, index)) for index, item in enumerate(value)]
        return copy.deepcopy(value)

    def _copy_tensor(self, tensor: torch.Tensor, path: tuple) -> torch.Tensor:
        buffer = self.buffers.get(path)
        if buffer is None or buffer.shape != tensor.shape or buffer.dtype != tensor.dtype:
            buffer = empty_pinned(tensor.shape, tensor.dtype)
            self.buffers[path] = buffer
        buffer.copy_(tensor, non_blocking=True)
        return buffer


def _copy_cpu_state_split_on_sharded_dim_to_gpu(
    model: nn.Module, optimizer_state_dict: dict[str, Any]
) -> dict[str, Any]:
    """Splitting a packed parameter's state along its sharded dim all-gathers it, so CPU-resident state is moved
    to GPU for that gather to run over NCCL (from #4331)."""
    state = dict(optimizer_state_dict["state"])
    for packed_info in get_model_packed_parameters(model):
        if packed_info.fqn not in state:
            continue
        sharded_on_packing_dim = Shard(packed_info.spec.dim)
        state[packed_info.fqn] = {
            key: DTensor(value._local_tensor.cuda(), value._spec, requires_grad=False)
            if isinstance(value, DTensor) and value._local_tensor.is_cpu and sharded_on_packing_dim in value.placements
            else value
            for key, value in state[packed_info.fqn].items()
        }
    return {**optimizer_state_dict, "state": state}


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

    def _build_state_dict(self, cpu_optimizer_state: bool = False) -> dict[str, Any]:
        # Automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        checkpoint_optimizers = self._get_checkpoint_optimizers()
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, checkpoint_optimizers)
        if cpu_optimizer_state:
            optimizer_state_dict = _copy_cpu_state_split_on_sharded_dim_to_gpu(self.model, optimizer_state_dict)
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
        return state_dict

    def async_state_dict(self) -> dict[str, Any]:
        """The state dict for an async save. It reads CPU-offloaded optimizer state where it lives, without the GPU
        round trip of ``prepare_checkpoint_save``; the async save copies it to pinned memory before returning."""
        return self._build_state_dict(cpu_optimizer_state=True)

    def state_dict(self) -> dict[str, Any]:
        for optimizer in self.optimizers:
            if isinstance(optimizer, OffloadOptimizer):
                optimizer.prepare_checkpoint_save()

        state_dict = self._build_state_dict()

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
    """Utility class to save and load trainer checkpoints to resume SFT and RL training."""

    def __init__(self, output_dir: Path, config: CheckpointConfig, resume: ResumeConfig | None = None):
        self.config = config
        self.skip_optimizer = config.skip_optimizer
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.logger = get_logger()
        self.world = get_world()

        self.stager: PinnedStager | None = None
        # Background writes run in their own thread, with their DCP collectives on their own gloo group
        self.writer: ThreadPoolExecutor | None = None
        self.write_group: torch.distributed.ProcessGroup | None = None
        self.pending: PendingSave | None = None

        # Only complete steps count toward keep_last: a step cut off mid-write before a resume must not displace one
        all_steps = [
            step
            for step in get_all_ckpt_steps(self.ckpt_dir)
            if is_ckpt_step_complete(get_step_path(self.ckpt_dir, step))
        ]
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
    ) -> tuple[Staging | None, Future] | None:
        """Save the trainer checkpoint to a given path. With ``async_save``, returns what ``_async_save`` does."""
        self.logger.debug(f"Saving training checkpoint to {path}")
        start_time = time.perf_counter()

        app_state = AppState(model, optimizers, scheduler, progress)

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
        if not self.config.async_save:
            dcp_save({"app": app_state}, checkpoint_id=path, process_group=get_dcp_group())
            self.logger.debug(f"Saved training checkpoint in {format_time(time.perf_counter() - start_time)}")
            return None
        return self._async_save(app_state, path, start_time)

    def _async_save(self, app_state: AppState, path: Path, start_time: float) -> tuple[Staging | None, Future]:
        """Copy the state into pinned host buffers and write them from a background thread. Returns the copy (None
        once it is done) and the write future.

        The copy of GPU-resident state overlaps the next step, until ``wait_for_staging`` before its optimizer step.
        Offloaded optimizer state is updated outside the optimizer step (``full`` offload updates it during backward),
        so with an offload optimizer the copy finishes before this returns."""
        if self.stager is None:
            self.stager = PinnedStager()
            # A thread rather than DCP's writer process: sending DTensors to that process crashes on torch 2.13/2.14,
            # and it needs shared memory the size of the shard. A new thread's current device is 0, so without the
            # initializer its first CUDA call would create a context on GPU 0 for every rank.
            self.writer = ThreadPoolExecutor(
                max_workers=1, initializer=torch.cuda.set_device, initargs=(torch.cuda.current_device(),)
            )
            self.write_group = torch.distributed.new_group(backend="gloo", timeout=DCP_TIMEOUT)
        staging = self.stager.stage({"app": app_state.async_state_dict()})
        write = self.writer.submit(self._write, staging, path)
        if any(isinstance(optimizer, OffloadOptimizer) for optimizer in app_state.optimizers):
            staging.wait()
            staging = None
        self.logger.info(
            f"{'Staged' if staging is None else 'Started staging'} training checkpoint in "
            f"{format_time(time.perf_counter() - start_time)}, writing it in the background"
        )
        return staging, write

    def _write(self, staging: Staging, path: Path) -> None:
        # The default writer's copy-ahead loader syncs the thread's current stream, which is the default
        # stream training runs on, about once per tensor. The staged tensors are already in pinned host memory.
        writer = FileSystemWriter(path, per_thread_copy_ahead=0)
        dcp_save(staging.wait(), storage_writer=writer, process_group=self.write_group)

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
        """Save the full checkpoint state for a specified step. With ``async_save``, this returns once the state is
        staged (without optimizer offload, once staging started) and the write finishes in the background: call
        ``wait_for_staging`` before the next optimizer step and ``wait`` before relying on the checkpoint."""
        # One save in flight: the previous write must finish before its host buffers are reused
        self.wait()
        start_time = time.perf_counter()
        ckpt_path = self.get_ckpt_path(step)
        # Master-only mkdir + barrier: concurrent mkdir from every rank can
        # re-raise FileExistsError on a parallel FS (see save_to_path).
        if self.world.is_master:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier(group=get_dcp_group())

        futures = self.save_to_path(ckpt_path, model, optimizers, scheduler, progress, dataloader)
        if futures is None:
            self._register(step)
        else:
            staging, write = futures
            self.pending = PendingSave(step=step, staging=staging, write=write, start_time=start_time)
            write.add_done_callback(self.pending.record_end_time)

    def _register(self, step: int) -> None:
        """Count a fully written checkpoint, making it visible to ``maybe_clean``."""
        if step not in self.ckpt_steps:
            bisect.insort(self.ckpt_steps, step)

    def wait_for_staging(self) -> None:
        """The mutation fence of an async save: call before anything mutates the checkpointed state (the
        optimizer step). Blocks until a pending save has copied the state to host memory; its write may still run.
        Also registers a write that has finished, and raises if it failed."""
        if self.pending is None:
            return
        if self.pending.staging is not None:
            start_time = time.perf_counter()
            self.pending.staging.wait()
            self.pending.staging = None
            self.logger.info(f"Waited {format_time(time.perf_counter() - start_time)} for checkpoint staging")
        if self.pending.write.done():
            self.wait()

    def wait(self) -> None:
        """Block until the pending save is written to storage. Raises if the write failed."""
        if self.pending is None:
            return
        pending, self.pending = self.pending, None
        wait_start_time = time.perf_counter()
        # Raises the background writer's exception, so a failed write stops training instead of being lost
        pending.write.result()
        # The done callback may still be running
        end_time = pending.end_time or time.perf_counter()
        self.logger.info(
            f"Wrote checkpoint for step {pending.step} in {format_time(end_time - pending.start_time)} "
            f"(waited {format_time(time.perf_counter() - wait_start_time)} for it now)"
        )
        self._register(pending.step)

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
