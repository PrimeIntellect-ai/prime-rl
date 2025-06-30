import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer
from transformers import AutoTokenizer

from zeroband.training.world import get_world
from zeroband.utils.logger import get_logger
from zeroband.utils.models import Model


@dataclass
class TrainingProgress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0


def save_full_checkpoint(
    model: Model,
    optimizers: list[Optimizer],
    progress: TrainingProgress,
    path: Path,
):
    # Get logger
    logger = get_logger()
    start_time = time.time()
    logger.info(f"Writing checkpoint to {path}")

    # Create checkpoint state
    ckpt_state = {
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "progress": progress,
    }

    # Create checkpoint directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)
    local_path = path / f"local_rank_{get_world().local_rank}"
    with open(local_path, "wb") as f:
        torch.save(ckpt_state, f)
    logger.info(f"Checkpoint saved at {path} in {time.time() - start_time:.2f} seconds")


def load_full_checkpoint(
    model: Model,
    optimizers: list[Optimizer],
    progress: TrainingProgress,
    path: Path,
):
    # Get logger
    logger = get_logger()
    start_time = time.time()
    logger.info(f"Loading checkpoint from {path}")

    # Check local step path exists
    local_path = path / f"local_rank_{get_world().local_rank}"
    if not local_path.exists():
        raise FileNotFoundError(f"Checkpoint step {progress.step} not found at {local_path}")

    # Load checkpoint state
    with open(local_path, "rb") as f:
        state = torch.load(f, weights_only=False)

    # Initialize model and optimizers
    model.load_state_dict(state["model"])
    for optimizer, optimizer_state in zip(optimizers, state["optimizers"]):
        optimizer.load_state_dict(optimizer_state)

    # Update progress
    progress.total_tokens = state["progress"].total_tokens
    progress.step = state["progress"].step
    progress.total_samples = state["progress"].total_samples

    logger.info(f"Checkpoint loaded in {time.time() - start_time:.2f} seconds")


def save_weight_checkpoint(
    model: Model,
    tokenizer: AutoTokenizer,
    path: Path,
    dtype: torch.dtype = torch.bfloat16,
    async_save: bool = False,
) -> Path:
    """
    Save a HF-compatible weight-only checkpoint to the specified path. Saves the
    model weights as `model.pt`, the model config, generation arguments and tokenizer
    using HF's `save_pretrained` method.
    """
    # Get logger and world info
    logger = get_logger()
    is_master = get_world().rank == 0

    # Create checkpoint directory if it doesn't exist
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    # Gather distributed weights for weight checkpoint
    start_time = time.time()
    logger.info("Gathering sharded weights")
    cpu_state = {}
    for key, value in model.state_dict().items():
        if isinstance(value, DTensor):
            value: DTensor = value.to(dtype)
            # only gather after the downcast to dtype as it will be faster
            value = value.full_tensor()  # ideally would only be gathered on rank 0

        if is_master:
            key: set[str] = get_fqns(model, key)
            assert len(key) == 1
            key = next(iter(key))
            # TODO(Sami) Blocking to avoid race condition, should make non-blocking long-term tho
            cpu_state[key] = value.to("cpu", non_blocking=False)

    torch.distributed.barrier()
    logger.info(f"Gathered sharded weights in {time.time() - start_time:.2f} seconds")

    model_path = path / "model.pt"

    def _save_weight_checkpoint():
        if is_master:
            start_time = time.time()
            logger.info(f"Saving weights to {path}")

            # Save model weights to temporary file to avoid race condition
            tmp_model_path = path / "model.pt.tmp"
            torch.save(cpu_state, tmp_model_path)

            # Rename temporary file to indicate checkpoint is complete
            tmp_model_path.rename(model_path)

            # Save model config, generation arguments and tokenizer
            model.config.save_pretrained(path)
            model.generation_config.save_pretrained(path)
            tokenizer.save_pretrained(path)

            logger.info(f"Saved weights to {path} in {time.time() - start_time:.2f} seconds")

    if async_save:
        thread = threading.Thread(target=_save_weight_checkpoint)
        thread.start()
    else:
        _save_weight_checkpoint()

    return model_path
