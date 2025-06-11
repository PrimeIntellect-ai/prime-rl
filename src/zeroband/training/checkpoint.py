import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
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
    scheduler: torch.optim.lr_scheduler.LRScheduler,
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
        state["scheduler"] = scheduler.state_dict()

        torch.save(state, f)


def load_checkpoint_fsdp_state(
    model: ModelType,
    optimizers: list[torch.optim.Optimizer],
    training_progress: TrainingProgress,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
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

    scheduler.load_state_dict(state["scheduler"])


async_ckpt_job = None


def save_ckpt_for_rollout(
    model: ModelType, tokenizer: AutoTokenizer, path: Path, dtype: torch.dtype = torch.bfloat16, async_save: bool = False
) -> Path:
    """
    Save the checkpoint for rollout as one unified safetensors file.

    Return:
        Path to the saved checkpoint safetensor
    """
    logger = get_logger("TRAIN")
    world_info = get_world_info()

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    path_file = path / "model.safetensors"

    start_time = time.time()
    logger.info(f"Saving rollout ckpt at {path}")

    cpu_state = {}

    for key, value in model.state_dict().items():
        if isinstance(value, DTensor):
            value: DTensor = value.to(dtype)
            # only gather after the downcast to dtype as it will be faster
            value = value.full_tensor()  # ideally would only be gathered on rank 0

        if world_info.rank == 0:
            key: set[str] = get_fqns(model, key)
            assert len(key) == 1
            key = next(iter(key))
            cpu_state[key] = value.to("cpu", non_blocking=False)
            # TODO(SAMI) keeping blocking here to avoid race condition, should be faster to make it non blocking tho

    torch.distributed.barrier()

    logger.info(f"gathering full tensor checkpointing in {time.time() - start_time:.2f} seconds")

    def _save():
        if world_info.rank == 0:
            save_file(cpu_state, path_file, metadata={"format": "pt"})

            model.config.save_pretrained(path)
            model.generation_config.save_pretrained(path)
            tokenizer.save_pretrained(path)

            stable_file = path / "stable"
            stable_file.touch()

            logger.info(f"Full Rollout ckpt saved at {path} in {time.time() - start_time:.2f} seconds")

    if async_save:
        logger.info(f"Rollout ckpt async saving  in {path} in {time.time() - start_time:.2f} seconds scheduled with async")
        async_ckpt_job = threading.Thread(target=_save)
        async_ckpt_job.start()
    else:
        _save()

    return path_file
