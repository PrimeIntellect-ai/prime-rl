import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer

from zeroband.training.world_info import get_world_info
from zeroband.utils.logger import get_logger
from zeroband.utils.models import ModelType


@dataclass
class TrainingProgress:
    total_tokens: int
    step: int
    total_samples: int


def _local_file_path(path: Path, local_rank: int) -> Path:
    return path / f"local_rank_{local_rank}.pt"


def _pathify(path: str | Path) -> Path:
    if isinstance(path, str):
        return Path(path)
    return path


def save_checkpoint_fsdp_state(
    model: ModelType,
    optimizers: list[torch.optim.Optimizer],
    training_progress: TrainingProgress,
    path_root: str | Path,
):
    """
    Checkpoint the model in a way that is compatible with FSDP.
    """
    path_root = _pathify(path_root) / f"step_{training_progress.step}"
    world_info = get_world_info()

    path_file = _local_file_path(path_root, world_info.local_rank)

    os.makedirs(path_root, exist_ok=True)

    with open(path_file, "wb") as f:
        state = {}
        state["model"] = model.state_dict()
        state["optimizers"] = [optimizer.state_dict() for optimizer in optimizers]
        state["training_progress"] = training_progress

        torch.save(state, f)


def load_checkpoint_fsdp_state(
    model: ModelType,
    optimizers: list[torch.optim.Optimizer],
    training_progress: TrainingProgress,
    path: str | Path,
):
    """
    Load the checkpoint state.
    """
    path = _pathify(path)
    world_info = get_world_info()

    path_file = _local_file_path(path, world_info.local_rank)

    if not os.path.exists(path_file):
        raise FileNotFoundError(f"Checkpoint step {training_progress.step} not found at {path_file}")

    with open(path_file, "rb") as f:
        state = torch.load(f, weights_only=False)

    model.load_state_dict(state["model"])

    for optimizer, optimizer_state in zip(optimizers, state["optimizers"]):
        optimizer.load_state_dict(optimizer_state)

    training_progress.total_tokens = state["training_progress"].total_tokens
    training_progress.step = state["training_progress"].step
    training_progress.total_samples = state["training_progress"].total_samples


@dataclass
class CkptJobStatus:
    path: Path
    success: bool
    error: Exception | None = None


class RolloutCkptManager:
    """
    This class is used to save the checkpoint for rollout in a separate process.
    Most of the checkpointing is async expect the gpu all gather.
    Moving tensor to cpu and to disk is async.
    Disk write is done in a separate process.
    """

    def __init__(self, tokenizer: AutoTokenizer, max_async_level: int, interval_rollout: int | None = None):
        self.max_async_level = max_async_level
        self.interval_rollout = interval_rollout

        self.tokenizer = tokenizer
        self.logger = get_logger()

        if get_world_info().rank == 0:
            ctx = mp.get_context("spawn")

            self.saving_queue = ctx.Queue()
            self.results_queue: mp.Queue = ctx.Queue()
            self.process = ctx.Process(target=self._save_loop, args=(self.saving_queue, self.results_queue))
            self.process_delete = ctx.Process(
                target=self._delete_ckpt_loop, args=(self.results_queue, self.max_async_level, self.interval_rollout)
            )
            self.process.start()
            self.process_delete.start()

    @staticmethod
    def _save_loop(ckpt_job_queue: mp.Queue, results_queue: mp.Queue):
        """Runs in its *own* Python process; handles disk I/O only."""

        while True:
            cpu_state, path, start_time = ckpt_job_queue.get()

            try:
                path_file = path / "model.safetensors"

                save_file(cpu_state, path_file, metadata={"format": "pt"})

                # model.config.save_pretrained(path)
                # model.generation_config.save_pretrained(path)
                # tokenizer.save_pretrained(path)

                stable_file = path / "stable"
                stable_file.touch()

                # logger.info(f"Full Rollout ckpt saved at {path} in {time.time() - start_time:.2f} seconds")
                results_queue.put(obj=CkptJobStatus(path=path, success=True))
            except Exception as e:
                # logger.error(f"Error saving rollout ckpt at {path}: {e}")
                results_queue.put(CkptJobStatus(path=path, success=False, error=e))
                break

    @staticmethod
    def _delete_ckpt_loop(results_queue: mp.Queue, max_async_level: int, interval_rollout: int | None):
        """Process to handle deletion of old checkpoints."""
        ckpt_to_delete = []
        while True:
            ckpt_job_status = results_queue.get()

            if ckpt_job_status.success:
                try:
                    ckpt_to_delete.append(ckpt_job_status.path)

                    if len(ckpt_to_delete) > max_async_level:
                        path_to_delete = ckpt_to_delete.pop(0)
                        ckpt_step = int(str(path_to_delete).split("_")[-1])

                        should_keep = interval_rollout is not None and ckpt_step % interval_rollout == 0

                        if path_to_delete.exists() and not should_keep:
                            shutil.rmtree(path_to_delete, ignore_errors=True)

                except Exception:
                    # Log error but don't break the loop
                    # logger.error(f"Error deleting rollout ckpt at {path_to_delete}: {e}")
                    pass

    def save_ckpt_for_rollout(self, model: ModelType, path: Path, dtype: torch.dtype = torch.bfloat16) -> Path:
        """
        Save the checkpoint for rollout as one unified safetensors file.

        Return:
            Path to the saved checkpoint safetensor
        """
        world_info = get_world_info()

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        self.logger.info(f"Saving rollout ckpt at {path}")

        copy_stream = torch.cuda.Stream()  # side stream for copies
        default_stream = torch.cuda.current_stream()  # usually the default stream

        cpu_state = {}
        for k, v in model.state_dict().items():
            # materialise and cast on the default stream
            if isinstance(v, DTensor):
                v = v.to(dtype).full_tensor()
            else:
                v = v.to(dtype)

            if world_info.rank == 0:
                host_buf = torch.empty_like(v, device="cpu", pin_memory=True)

                # ▸ 1. make the copy stream wait for all prior work on default
                copy_stream.wait_stream(default_stream)

                # ▸ 2. launch the async copy *inside* the copy stream context
                with torch.cuda.stream(copy_stream):
                    host_buf.copy_(v, non_blocking=True)
                    # ▸ 3. tell the allocator that v is still in use by copy_stream
                    v.record_stream(copy_stream)

                cpu_state[k] = host_buf

        # ensure all outstanding copies are finished before we serialise
        copy_stream.synchronize()

        if get_world_info().rank == 0:
            self.saving_queue.put((cpu_state, path, start_time))

        self.logger.info(f"Saving rollout ckpt at {path} scheduled in {time.time() - start_time:.2f} seconds")

        return path
