"""Lean checkpoint manager for orchestrator v2.

Persists ``Progress(step, totals…)`` only. The legacy orchestrator's buffer /
difficulty-pool persistence is intentionally dropped: v2 has no buffer (the
dispatcher iterates the dataset directly via the existing ``TrainEnvs``
abstraction) and no difficulty pools (replaced by ``pre_batch_filters``).

Layout — matches the legacy orchestrator's ``checkpoints/step_N/orchestrator/``
prefix so trainer weight discovery does not need to change.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator_v2.types import Progress
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_ckpt_dir, get_step_path


class CheckpointManager:
    """Saves/loads ``Progress`` under ``<output_dir>/checkpoints/step_N/orchestrator/state.pt``."""

    def __init__(self, output_dir: Path, config: CheckpointConfig) -> None:
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.logger = get_logger()

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def save(self, progress: Progress, step: int) -> None:
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with open(ckpt_path / "state.pt", "wb") as f:
            torch.save({"progress": progress}, f)
        self.logger.debug(f"V2 orchestrator checkpoint saved to {ckpt_path} in {time.perf_counter() - start:.2f}s")

    def load(self, progress: Progress, step: int) -> None:
        ckpt_path = self.get_ckpt_path(step)
        state_file = ckpt_path / "state.pt"
        if not state_file.exists():
            raise FileNotFoundError(f"V2 orchestrator checkpoint not found at {state_file}")
        self.logger.debug(f"Loading v2 checkpoint from {state_file}")
        start = time.perf_counter()
        if self.config.skip_progress:
            self.logger.info("Skipping progress loading from checkpoint")
        else:
            with open(state_file, "rb") as f:
                state = torch.load(f, weights_only=False)
            saved: Progress = state["progress"]
            for key, value in asdict(saved).items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
        self.logger.debug(f"V2 orchestrator checkpoint loaded in {time.perf_counter() - start:.2f}s")


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)
