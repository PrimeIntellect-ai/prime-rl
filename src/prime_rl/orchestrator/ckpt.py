"""Checkpoint manager for orchestrator progress and train-source state. Layout:
``<output_dir>/checkpoints/step_N/orchestrator/progress.pt``."""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any

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

    def save(self, step: int, progress: Progress, train_source: dict[str, Any]) -> None:
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        # Save to a temporary file and do an atomic rename, to avoid corrupting the last
        # file if the process gets killed while writing
        fd, tmp_name = tempfile.mkstemp(dir=ckpt_path, prefix="progress.pt.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save({"progress": progress, "train_source": train_source}, f)
            os.replace(tmp_name, ckpt_path / "progress.pt")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        get_logger().debug(
            f"Orchestrator checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}"
        )

    def load(self, step: int, path: Path | None = None) -> tuple[Progress, dict[str, Any]] | None:
        """The saved progress and train-source state, or None when the config skips
        them. ``path`` overrides where the checkpoint is read from (an external run's
        ``step_<N>/orchestrator``)."""
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Orchestrator checkpoint not found at {state_file}")
        if self.config.skip_progress:
            get_logger().info("Skipping progress and train source loading from checkpoint")
            return None
        get_logger().debug(f"Loading checkpoint from {state_file}")
        start = time.perf_counter()
        with open(state_file, "rb") as f:
            state = torch.load(f, weights_only=False)
        get_logger().debug(f"Orchestrator checkpoint loaded in {format_time(time.perf_counter() - start)}")
        return state["progress"], state["train_source"]


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager:
    """The checkpoint manager always exists: ``resume`` decides whether it loads,
    ``ckpt`` whether it saves (a resume without ``ckpt`` loads but saves nothing)."""
    return CheckpointManager(output_dir, config or CheckpointConfig())
