"""Checkpoint standalone eval progress and completed task-group results."""

from __future__ import annotations

import contextlib
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from prime_rl.utils.logger import format_time, get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_ckpt_dir, get_step_path

if TYPE_CHECKING:
    from prime_rl.orchestrator.eval_sink import EvalSink
    from prime_rl.orchestrator.eval_source import EvalSource


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

    def save(self, step: int, eval_source: EvalSource, eval_sink: EvalSink) -> None:
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        fd, tmp_name = tempfile.mkstemp(dir=ckpt_path, prefix="progress.pt.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(
                    {
                        "completed_groups": step,
                        "eval_source": eval_source.state_dict(),
                        "eval_sink": eval_sink.state_dict(),
                    },
                    f,
                )
            os.replace(tmp_name, ckpt_path / "progress.pt")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        get_logger().debug(f"Evals checkpoint saved to {ckpt_path} in {format_time(time.perf_counter() - start)}")

    def load(
        self,
        step: int,
        eval_source: EvalSource,
        eval_sink: EvalSink,
        path: Path | None = None,
    ) -> None:
        ckpt_path = path if path is not None else self.get_ckpt_path(step)
        state_file = ckpt_path / "progress.pt"
        if not state_file.is_file():
            raise FileNotFoundError(f"Evals checkpoint not found at {state_file}")
        get_logger().info(f"Loading evals checkpoint from {state_file}")
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        if state["completed_groups"] != step:
            raise ValueError(
                f"Evals checkpoint contains {state['completed_groups']} completed groups, expected step {step}"
            )
        eval_source.load_state_dict(state["eval_source"])
        eval_sink.load_state_dict(state["eval_sink"])
