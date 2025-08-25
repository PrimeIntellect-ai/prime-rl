import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import CheckpointConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_ckpt_dir


@dataclass
class RLProgress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


@dataclass
class SFTProgress(RLProgress):
    epoch: int = 0
    epoch_step: int = 0


class CheckpointManager:
    """Utility class to save and load orchestrator checkpoints to resume orchestrator."""

    def __init__(self, output_dir: Path, config: CheckpointConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self._logger = get_logger()
        self.ckpt_steps: list[int] = []

    def _get_step_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}"

    def _get_ckpt_path(self, step: int) -> Path:
        return self._get_step_path(step) / "orchestrator.pt"

    def get_ckpt_path(self, step: int) -> Path:
        return self._get_step_path(step)

    def _save_to_path(
        self,
        ckpt_path: Path,
        ckpt_step: int,
        buffer: Buffer,
        progress: RLProgress | SFTProgress,
    ):
        self._logger.debug(f"Saving orchestrator checkpoint to {ckpt_path}")
        start_time = time.time()

        # Create checkpoint state
        ckpt_state = {"progress": progress}

        # Save checkpoint state
        with open(ckpt_path, "wb") as f:
            torch.save(ckpt_state, f)

        buffer_path = ckpt_path.parent / "buffer"
        buffer.save(buffer_path)

        # Append to list of saved steps
        self.ckpt_steps.append(ckpt_step)

        self._logger.debug(f"Orchestrator checkpoint saved in {time.time() - start_time:.2f} seconds")

    def _load_from_path(self, ckpt_path: Path, progress: RLProgress | SFTProgress) -> None:
        """Loads a checkpoint from a given path in-place."""
        self._logger.debug(f"Loading checkpoint from {ckpt_path}")
        start_time = time.time()

        # Load checkpoint state
        with open(ckpt_path, "rb") as f:
            state = torch.load(f, weights_only=False)

        # Load checkpoint state in-place
        for key, value in asdict(state["progress"]).items():
            setattr(progress, key, value)

        self._logger.debug(f"Orchestrator checkpoint loaded in {time.time() - start_time:.2f} seconds")

    def load(self, progress: RLProgress | SFTProgress, step: int) -> None:
        """Loads a checkpoint from a given path."""
        ckpt_path = self._get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self._load_from_path(ckpt_path, progress)

    def save(
        self,
        progress: RLProgress | SFTProgress,
        buffer: Buffer,
        step: int,
    ) -> None:
        """Saves the full checkpoint state for a specified step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._get_ckpt_path(step)
        self._save_to_path(ckpt_path, step, buffer, progress)

    def maybe_clean(self) -> None:
        """Deletes past orchestrator checkpoints beyond the most recent `keep` steps. No-op if `keep` is None."""
        if self.config.keep is None:
            return

        # Get all the checkpoint steps to delete
        assert list(self.ckpt_steps) == sorted(self.ckpt_steps)
        ckpt_steps_to_keep = self.ckpt_steps[-self.config.keep :]
        ckpt_steps_to_delete = self.ckpt_steps[: -self.config.keep]
        for ckpt_step in ckpt_steps_to_delete:
            ckpt_path = self._get_ckpt_path(ckpt_step)
            if ckpt_path.exists():
                self._logger.debug(
                    f"Removing past orchestrator checkpoint for step {ckpt_step} ({ckpt_path}), because got checkpoints for {ckpt_steps_to_keep} ({len(self.ckpt_steps)} > {self.config.keep})"
                )
                ckpt_path.unlink(missing_ok=True)

        # Update checkpoint steps
        self.ckpt_steps = self.ckpt_steps[-self.config.keep :]


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)
