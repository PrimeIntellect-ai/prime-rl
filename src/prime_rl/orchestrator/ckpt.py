"""Checkpoint manager for ``Progress``. Layout:
``<output_dir>/checkpoints/step_N/orchestrator/progress.pt``."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.types import Progress
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


class CheckpointManager:
    def __init__(self, output_dir: Path, config: CheckpointConfig) -> None:
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def save(self, progress: Progress, step: int) -> None:
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with open(ckpt_path / "progress.pt", "wb") as f:
            torch.save({"progress": progress}, f)
        get_logger().debug(
            f"Orchestrator checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}"
        )

    def load(self, progress: Progress, step: int) -> None:
        ckpt_path = self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Orchestrator checkpoint not found at {state_file}")
        get_logger().debug(f"Loading checkpoint from {state_file}")
        start = time.perf_counter()
        if self.config.skip_progress:
            get_logger().info("Skipping progress loading from checkpoint")
        else:
            with open(state_file, "rb") as f:
                state = torch.load(f, weights_only=False)
            saved: Progress = state["progress"]
            for key, value in asdict(saved).items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
        get_logger().debug(f"Orchestrator checkpoint loaded in {format_time(time.perf_counter() - start)}")


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)
