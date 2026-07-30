"""Checkpoint manager for the orchestrator state (``Progress`` counters +
``TrainSource`` data position). Layout:
``<output_dir>/checkpoints/step_N/orchestrator/progress.pt``."""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Progress
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


class CheckpointManager:
    def __init__(self, output_dir: Path, config: CheckpointConfig) -> None:
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def save(self, progress: Progress, train_source: TrainSource, step: int) -> None:
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        # Save to a temporary file and do an atomic rename, to avoid corrupting the last
        # file if the process gets killed while writing
        fd, tmp_name = tempfile.mkstemp(dir=ckpt_path, prefix="progress.pt.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save({"progress": progress, "train_source": train_source.state_dict()}, f)
            os.replace(tmp_name, ckpt_path / "progress.pt")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        get_logger().debug(
            f"Orchestrator checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}"
        )

    def load(self, progress: Progress, train_source: TrainSource, step: int) -> None:
        ckpt_path = self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Orchestrator checkpoint not found at {state_file}")
        get_logger().debug(f"Loading checkpoint from {state_file}")
        start = time.perf_counter()
        if self.config.skip_progress:
            get_logger().info("Skipping progress and data position loading from checkpoint")
        else:
            with open(state_file, "rb") as f:
                state = torch.load(f, weights_only=False)
            saved: Progress = state["progress"]
            for key, value in asdict(saved).items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
            train_source.load_state_dict(state["train_source"])
            for name, position in state["train_source"].items():
                if name not in train_source.base_rows:
                    continue
                rows = train_source.base_rows[name]
                num_tasks = len(rows) if rows is not None else "infinite"
                get_logger().info(
                    f"Resumed data position for env {name} - epoch={position['epoch']}, "
                    f"cursor={position['cursor']}/{num_tasks}"
                )
        get_logger().debug(f"Orchestrator checkpoint loaded in {format_time(time.perf_counter() - start)}")


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)
