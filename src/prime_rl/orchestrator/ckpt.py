"""Checkpoint manager for orchestrator progress, train-source state, and the
train sink's queued traces. Layout:
``<output_dir>/checkpoints/step_N/orchestrator/{progress,queue}.pt``."""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Progress
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


def _atomic_save(state: Any, path: Path) -> None:
    """Save to a temporary file and rename it, so a kill mid-write cannot
    corrupt the previous file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


class CheckpointManager:
    def __init__(self, output_dir: Path, config: CheckpointConfig) -> None:
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    async def save(self, progress: Progress, train_source: TrainSource, train_sink: TrainSink, step: int) -> None:
        """Progress and train-source state are small and written on the event
        loop, which keeps the dispatcher from mutating them mid-save. The
        sink's queued traces can be large: they are snapshotted on the loop
        and written to ``queue.pt`` in a thread."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        _atomic_save({"progress": progress, "train_source": train_source.state_dict()}, ckpt_path / "progress.pt")
        await asyncio.to_thread(_atomic_save, train_sink.state_dict(), ckpt_path / "queue.pt")
        get_logger().debug(
            f"Orchestrator checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}"
        )

    def load(
        self, progress: Progress, train_source: TrainSource, step: int, path: Path | None = None
    ) -> dict[str, Any] | None:
        """``path`` overrides where the checkpoint is read from (an external run's
        ``step_<N>/orchestrator``). Returns the train sink's queued traces for
        ``TrainSink.load_state_dict`` (the sink is built later in setup), or
        None when there are none to replay."""
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Orchestrator checkpoint not found at {state_file}")
        get_logger().debug(f"Loading checkpoint from {state_file}")
        start = time.perf_counter()
        if self.config.skip_progress:
            get_logger().info("Skipping progress and train source loading from checkpoint")
            return None
        with open(state_file, "rb") as f:
            state = torch.load(f, weights_only=False)
        saved: Progress = state["progress"]
        for key, value in asdict(saved).items():
            if hasattr(progress, key):
                setattr(progress, key, value)
        train_source.load_state_dict(state["train_source"])
        for name in state["train_source"]["envs"]:
            if name in train_source.curricula:
                get_logger().info(f"Resumed curriculum state for env {name}")
        sink_state = None
        queue_file = ckpt_path / "queue.pt"
        if queue_file.exists():
            # Replay only saves regeneration time: a queue that no longer
            # unpickles (e.g. after a dependency upgrade) must not block the resume.
            try:
                with open(queue_file, "rb") as f:
                    sink_state = torch.load(f, weights_only=False)
            except Exception as error:
                get_logger().warning(f"Resuming without replaying queued traces: cannot load {queue_file}: {error!r}")
        get_logger().debug(f"Orchestrator checkpoint loaded in {format_time(time.perf_counter() - start)}")
        return sink_state


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager:
    """The checkpoint manager always exists: ``resume`` decides whether it loads,
    ``ckpt`` whether it saves (a resume without ``ckpt`` loads but saves nothing)."""
    return CheckpointManager(output_dir, config or CheckpointConfig())
