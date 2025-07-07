import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim.optimizer import Optimizer

from zeroband.training.model import Model
from zeroband.training.world import get_world
from zeroband.utils.logger import get_logger


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
    logger.debug(f"Writing checkpoint to {path}")

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
    logger.debug(f"Checkpoint saved at {path} in {time.time() - start_time:.2f} seconds")


def load_full_checkpoint(
    model: Model,
    optimizers: list[Optimizer],
    progress: TrainingProgress,
    path: Path,
):
    # Get logger
    logger = get_logger()
    start_time = time.time()
    logger.debug(f"Loading checkpoint from {path}")

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

    logger.debug(f"Checkpoint loaded in {time.time() - start_time:.2f} seconds")
