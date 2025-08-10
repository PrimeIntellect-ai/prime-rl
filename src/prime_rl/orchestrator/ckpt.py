import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from prime_rl.orchestrator.config import CheckpointConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_ckpt_dir


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


class CheckpointManager:
    """Utility class to save and load orchestrator checkpoints to resume orchestrator."""

    def __init__(self, outputs_dir: Path, config: CheckpointConfig):
        self.ckpt_dir = get_ckpt_dir(outputs_dir)
        self._logger = get_logger()
        self.keep = config.keep
        # Track saved checkpoint steps to avoid scanning directory
        import threading

        self._saved_steps: list[int] = []
        self._lock = threading.Lock()

    def _get_step_path(self, step: int) -> Path:
        return self.ckpt_dir / f"step_{step}"

    def _get_ckpt_path(self, step: int) -> Path:
        return self._get_step_path(step) / "orchestrator.pt"

    def _save_to_path(self, ckpt_path: Path, progress: Progress):
        self._logger.debug(f"Saving orchestrator checkpoint to {ckpt_path}")
        start_time = time.time()

        # Create checkpoint state
        ckpt_state = {"progress": progress}

        # Save checkpoint state
        with open(ckpt_path, "wb") as f:
            torch.save(ckpt_state, f)

        # Record saved step after successful write
        try:
            step_str = ckpt_path.parent.name.split("_")[-1]
            step_num = int(step_str)
            with self._lock:
                self._saved_steps.append(step_num)
                self._saved_steps.sort()
        except Exception:
            # Best-effort: ignore if parsing fails
            pass

        self._logger.debug(f"Orchestrator checkpoint saved in {time.time() - start_time:.2f} seconds")

    def _load_from_path(self, ckpt_path: Path, progress: Progress) -> None:
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

    def maybe_clean(self, step: int):
        """Delete past orchestrator checkpoints beyond the most recent `keep` steps. No-op if `keep` is None."""
        if self.keep is None:
            return
        try:
            with self._lock:
                # Ensure sorted ascending
                self._saved_steps.sort()
                num_to_delete = max(0, len(self._saved_steps) - self.keep)
                steps_to_delete = [self._saved_steps.pop(0) for _ in range(num_to_delete)]
            for old_step in steps_to_delete:
                path = self._get_ckpt_path(old_step)
                if path.exists():
                    self._logger.debug(f"Removing past orchestrator checkpoint {path}")
                    try:
                        path.unlink(missing_ok=True)
                    except TypeError:
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            pass
        except Exception as e:
            self._logger.warning(f"Failed to cleanup old orchestrator checkpoints: {e}")

    def load(self, progress: Progress, step: int) -> Path:
        """Loads a checkpoint from a given path."""
        ckpt_path = self._get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self._load_from_path(ckpt_path, progress)

    def save(
        self,
        progress: Progress,
        step: int,
    ):
        """Saves the full checkpoint state for a specified step."""
        step_path = self._get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = self._get_ckpt_path(step)
        self._save_to_path(ckpt_path, progress)
