"""Checkpoint manager for orchestrator progress, train-source state, and the
train sink's queued work. Layout: ``<output_dir>/checkpoints/step_N/orchestrator/progress.pt``."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Progress
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


def _atomic_save(state: dict[str, Any], path: Path) -> None:
    """Save to a temporary file and rename it, so a kill mid-write never
    corrupts the previous file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
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

    async def save(
        self,
        progress: Progress,
        train_source: TrainSource,
        train_sink: TrainSink,
        open_groups: dict[str, tuple[str, vf.Task]],
        step: int,
    ) -> None:
        """One file, so the queued work and the dataset cursor always restore together.
        The state is snapshotted on the event loop (the dispatcher cannot mutate it
        mid-snapshot) and pickled in a thread, since the queue can be large."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        state = {
            "progress": copy.copy(progress),
            "train_source": copy.deepcopy(train_source.state_dict()),
            "train_sink": train_sink.state_dict(open_groups),
        }
        await asyncio.to_thread(_atomic_save, state, ckpt_path / "progress.pt")
        get_logger().debug(
            f"Orchestrator checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}"
        )

    def load(
        self, progress: Progress, train_source: TrainSource, step: int, path: Path | None = None
    ) -> dict[str, Any] | None:
        """``path`` overrides where the checkpoint is read from (an external run's
        ``step_<N>/orchestrator``). Returns the train sink's saved work for
        ``TrainSink.load_state_dict`` (the sink is built later in setup); None for
        checkpoints written before the sink was saved."""
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"Orchestrator checkpoint not found at {state_file}")
        get_logger().debug(f"Loading checkpoint from {state_file}")
        start = time.perf_counter()
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
        get_logger().debug(f"Orchestrator checkpoint loaded in {format_time(time.perf_counter() - start)}")
        return state.get("train_sink")


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager:
    """The checkpoint manager always exists: ``resume`` decides whether it loads,
    ``ckpt`` whether it saves (a resume without ``ckpt`` loads but saves nothing)."""
    return CheckpointManager(output_dir, config or CheckpointConfig())
