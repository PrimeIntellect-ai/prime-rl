import asyncio
import bisect
import gc
import json
import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import obstore
import torch
from obstore.exceptions import BaseError
from obstore.store import ObjectStore, from_url
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

# DCP writes it last, so a checkpoint is complete exactly when it exists
METADATA_FILE = ".metadata"
# R2 needs equal-size multipart parts. 64 MB parts with this many in flight reach the link rate
TRANSFER_PART_BYTES = 64 << 20
UPLOAD_PARTS_PER_FILE = 32
DOWNLOAD_PARTS_IN_FLIGHT = 256


@dataclass
class Progress:
    step: int = 1
    total_tokens: int = 0
    total_samples: int = 0


def get_remote_steps(remote: ObjectStore) -> tuple[set[int], set[int]]:
    """Every step in the remote, and the committed ones: those whose ``.metadata`` was uploaded."""
    keys = {meta["path"] for meta in obstore.list(remote).collect()}
    steps = {int(key.split("/")[0].removeprefix("step_")) for key in keys}
    return steps, {step for step in steps if f"{_get_remote_key(step)}/{METADATA_FILE}" in keys}


def _get_remote_key(step: int) -> str:
    return f"step_{step}/trainer"


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
    """Utility class to save and load trainer checkpoints to resume SFT and RL training."""

    def __init__(self, output_dir: Path, config: CheckpointConfig, resume: ResumeConfig | None = None):
        self.config = config
        self.skip_optimizer = config.skip_optimizer
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.logger = get_logger()
        self.world = get_world()
        self.remote = from_url(config.remote_dir) if config.remote_dir is not None else None
        self.uploader = ThreadPoolExecutor(max_workers=1)
        self._pending_save: tuple[int, Future] | None = None

        self.remote_steps = self._resolve_remote_steps()
        all_steps = sorted(set(get_all_ckpt_steps(self.ckpt_dir)) | self.remote_steps)
        if resume is not None and resume.step is not None:
            self.ckpt_steps = [s for s in all_steps if s <= resume.step]
        else:
            self.ckpt_steps = all_steps

    def get_ckpt_path(self, step: int) -> Path:
        """Get the path to write the trainer checkpoint for a given step."""
        return get_step_path(self.ckpt_dir, step) / "trainer"

    def resolve_latest_step(self, trainer_only: bool = False) -> int | None:
        """The latest step on disk or complete on the remote, None to start from scratch. ``trainer_only``
        skips steps on disk whose trainer checkpoint is incomplete (only the orchestrator's, or an interrupted save)."""
        steps = self.ckpt_steps
        if trainer_only:
            steps = [s for s in steps if s in self.remote_steps or (self.get_ckpt_path(s) / METADATA_FILE).exists()]
        return steps[-1] if steps else None

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
        """Load the trainer checkpoint for a given step (in-place), first downloading it from the remote
        when it is not complete on disk. ``path`` overrides where the checkpoint is read from (an external
        run's ``step_<N>/trainer``)."""
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        if path is None and self.remote is not None and not (ckpt_path / METADATA_FILE).exists():
            self._download(step, ckpt_path)
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
        """Save the full checkpoint state for a specified step. With a remote, the step is only added to
        ``ckpt_steps`` once ``wait`` has seen its background upload finish."""
        self.wait()
        ckpt_path = self.get_ckpt_path(step)
        # Master-only mkdir + barrier: concurrent mkdir from every rank can
        # re-raise FileExistsError on a parallel FS (see save_to_path).
        if self.world.is_master:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier(group=get_dcp_group())

        self.save_to_path(ckpt_path, model, optimizers, scheduler, progress, dataloader)
        if self.remote is None:
            bisect.insort(self.ckpt_steps, step)
        else:
            self._pending_save = (step, self.uploader.submit(self._upload, ckpt_path, step))

    def wait(self) -> None:
        """Block until the pending background upload is done, raising its error if it failed."""
        if self._pending_save is None:
            return
        step, future = self._pending_save
        self._pending_save = None
        future.result()
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
                if self.remote is not None:
                    self._remove_remote(ckpt_step)

        # Update checkpoint steps
        self.ckpt_steps = [step for step in self.ckpt_steps if step in steps_to_keep]

    def _download(self, step: int, path: Path) -> None:
        """Fetch a step from the remote into ``path``, shared by every rank as for any checkpoint, so each rank
        fetches its slice of the files. The master writes ``.metadata`` last and renames it into place, so a
        partial download never reads as complete."""
        key = _get_remote_key(step)
        metadata_key = f"{key}/{METADATA_FILE}"
        objects = sorted(
            (o for o in obstore.list(self.remote, prefix=key).collect() if o["path"] != metadata_key),
            key=lambda o: o["path"],
        )
        asyncio.run(self._get(key, path, objects[self.world.rank :: self.world.world_size]))
        torch.distributed.barrier(group=get_dcp_group())
        if self.world.is_master:
            partial = path / f"{METADATA_FILE}.partial"
            partial.write_bytes(obstore.get(self.remote, metadata_key).bytes())
            partial.replace(path / METADATA_FILE)
        torch.distributed.barrier(group=get_dcp_group())

    async def _get(self, key: str, path: Path, objects: list[dict]) -> None:
        """Download objects under ``key`` into ``path`` as concurrent byte ranges."""
        in_flight = asyncio.Semaphore(DOWNLOAD_PARTS_IN_FLIGHT)

        async def get_part(object_key: str, start: int, end: int, fd: int) -> None:
            async with in_flight:
                part = await obstore.get_range_async(self.remote, object_key, start=start, end=end)
            os.pwrite(fd, memoryview(part), start)

        async def get_object(meta: dict) -> None:
            file = path / meta["path"].removeprefix(f"{key}/")
            file.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            try:
                size = meta["size"]
                await asyncio.gather(
                    *[
                        get_part(meta["path"], start, min(start + TRANSFER_PART_BYTES, size), fd)
                        for start in range(0, size, TRANSFER_PART_BYTES)
                    ]
                )
            finally:
                os.close(fd)

        await asyncio.gather(*[get_object(meta) for meta in objects])

    def _resolve_remote_steps(self) -> set[int]:
        """Committed steps in the remote. The master removes uncommitted ones, uploads a crash cut short,
        since nothing else would ever read or remove them."""
        if self.remote is None:
            return set()
        steps, committed = get_remote_steps(self.remote)
        if self.world.is_master:
            for step in steps - committed:
                self._remove_remote(step)
        return committed

    async def _put(self, path: Path, step: int, files: list[Path]) -> None:
        """Upload files of the checkpoint at ``path`` concurrently, each in parallel parts."""
        key = _get_remote_key(step)
        await asyncio.gather(
            *[
                obstore.put_async(
                    self.remote,
                    f"{key}/{file.relative_to(path).as_posix()}",
                    file,
                    chunk_size=TRANSFER_PART_BYTES,
                    max_concurrency=UPLOAD_PARTS_PER_FILE,
                )
                for file in files
            ]
        )

    def _remove_remote(self, step: int) -> None:
        """``.metadata`` goes first, so a partly removed step never reads as complete. Logs and skips on
        failure, as ``_try_rmtree`` does: what is left is removed again after a restart."""
        try:
            keys = [meta["path"] for meta in obstore.list(self.remote, prefix=f"step_{step}").collect()]
            metadata_key = f"{_get_remote_key(step)}/{METADATA_FILE}"
            if metadata_key in keys:
                obstore.delete(self.remote, metadata_key)
            obstore.delete(self.remote, [key for key in keys if key != metadata_key])
        except BaseError as e:
            self.logger.warning(f"Failed to remove step {step} from {self.config.remote_dir}: {e}, skipping cleanup")

    def _upload(self, path: Path, step: int) -> None:
        """Each rank uploads the files it wrote (DCP names its shards ``__<rank>_<n>.distcp``). The master
        uploads ``mapping.json`` and ``.metadata`` last, once every rank succeeded."""
        rank = self.world.rank
        uploaded = False
        try:
            asyncio.run(
                self._put(path, step, [*path.glob(f"__{rank}_*.distcp"), *path.glob(f"dataloader/rank_{rank}.pt")])
            )
            uploaded = True
        finally:
            # Every rank reaches the collective, so a failed upload never leaves the others waiting
            failed_ranks = torch.tensor([int(not uploaded)])
            torch.distributed.all_reduce(failed_ranks, group=get_dcp_group())
        # Every rank raises, so none records the step as saved and cleans up the steps before it
        if failed_ranks.item() > 0:
            raise RuntimeError(f"Uploading step {step} failed on {failed_ranks.item()} rank(s)")
        if not self.world.is_master:
            return
        asyncio.run(self._put(path, step, [*path.glob("mapping.json")]))
        asyncio.run(self._put(path, step, [path / METADATA_FILE]))


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
