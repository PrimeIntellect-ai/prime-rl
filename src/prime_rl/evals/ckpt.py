"""Checkpoint standalone eval progress cursor."""

from __future__ import annotations

import contextlib
import os
import pickle
import tempfile
import time
from pathlib import Path

from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_ckpt_dir, get_step_path


class CheckpointManager:
    def __init__(self, output_dir: Path) -> None:
        self.ckpt_dir = get_ckpt_dir(output_dir)

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "evals"

    def latest_step(self) -> int:
        steps = [
            step for step in get_all_ckpt_steps(self.ckpt_dir) if (self.get_ckpt_path(step) / "progress.pt").is_file()
        ]
        if not steps:
            raise FileNotFoundError(f"No evals checkpoints found in {self.ckpt_dir}")
        return steps[-1]

    def save(self, eval_source: EvalSource) -> None:
        cursor = eval_source.cursor
        ckpt_path = self.get_ckpt_path(cursor)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        # Keep the last complete checkpoint readable if the process dies mid-write.
        fd, tmp_name = tempfile.mkstemp(dir=ckpt_path, prefix="progress.pt.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(eval_source.state_dict(), f)
            os.replace(tmp_name, ckpt_path / "progress.pt")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        get_logger().debug(
            f"Evals checkpoint saved to {ckpt_path} (cursor={cursor}) in {format_time(time.perf_counter() - start)}"
        )

    def load(
        self,
        step: int,
        eval_source: EvalSource,
        path: Path | None = None,
    ) -> None:
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.is_file():
            raise FileNotFoundError(f"Evals checkpoint not found at {state_file}")
        get_logger().info(f"Loading evals checkpoint from {state_file}")
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        if state["cursor"] != step:
            raise ValueError(f"Evals checkpoint contains cursor {state['cursor']}, expected step {step}")
        eval_source.load_state_dict(state)
