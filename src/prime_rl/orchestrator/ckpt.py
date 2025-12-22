import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import CheckpointConfig
from prime_rl.utils.disk import format_bytes_binary, format_percent, get_disk_usage
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_ckpt_dir, get_step_path


@dataclass
class Progress:
    step: int = 0
    total_tokens: int = 0
    total_samples: int = 0
    total_problems: int = 0


class CheckpointManager:
    """Utility class to save and load orchestrator checkpoints to resume orchestrator."""

    def __init__(self, output_dir: Path, config: CheckpointConfig):
        self.config = config
        self.ckpt_dir = get_ckpt_dir(output_dir)
        self.logger = get_logger()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        usage = get_disk_usage(self.ckpt_dir)
        free_ratio = usage.free_bytes / usage.total_bytes if usage.total_bytes else 0.0
        self.logger.info(
            "Checkpoint disk space: "
            f"path={self.ckpt_dir} "
            f"free={format_bytes_binary(usage.free_bytes)} ({format_percent(free_ratio)}) "
            f"total={format_bytes_binary(usage.total_bytes)}"
        )

    def get_ckpt_path(self, step: int) -> Path:
        return get_step_path(self.ckpt_dir, step) / "orchestrator"

    def _save_to_path(
        self,
        ckpt_path: Path,
        ckpt_step: int,
        progress: Progress,
        buffer: Buffer,
    ):
        self.logger.debug(f"Saving orchestrator checkpoint to {ckpt_path}")
        start_time = time.perf_counter()

        # Save progress
        with open(ckpt_path / "progress.pt", "wb") as f:
            torch.save({"progress": progress}, f)

        # Save buffer
        buffer.save(ckpt_path / "buffer")

        self.logger.debug(f"Orchestrator checkpoint saved in {time.perf_counter() - start_time:.2f} seconds")

    def _load_from_path(self, ckpt_path: Path, progress: Progress, buffer: Buffer) -> None:
        """Loads a checkpoint from a given path in-place."""
        self.logger.debug(f"Loading checkpoint from {ckpt_path}")
        start_time = time.perf_counter()

        # Load progress
        if self.config.skip_progress:
            self.logger.info("Skipping progress loading from checkpoint")
        else:
            with open(ckpt_path / "progress.pt", "rb") as f:
                state = torch.load(f, weights_only=False)

            # Set progress in-place
            for key, value in asdict(state["progress"]).items():
                setattr(progress, key, value)

        # Load buffer
        if self.config.skip_buffer:
            self.logger.info("Skipping buffer loading from checkpoint")
        else:
            buffer.load(ckpt_path / "buffer")

        self.logger.debug(f"Orchestrator checkpoint loaded in {time.perf_counter() - start_time:.2f} seconds")

    def load(self, progress: Progress, buffer: Buffer, step: int) -> None:
        """Loads a checkpoint from a given path."""
        ckpt_path = self.get_ckpt_path(step)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        self._load_from_path(ckpt_path, progress, buffer)

    def save(
        self,
        progress: Progress,
        buffer: Buffer,
        step: int,
    ) -> None:
        """Saves the full checkpoint state for a specified step."""
        ckpt_path = self.get_ckpt_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        usage = get_disk_usage(self.ckpt_dir)
        free_ratio = usage.free_bytes / usage.total_bytes if usage.total_bytes else 0.0
        self.logger.info(
            "Checkpoint disk space (pre-save): "
            f"path={self.ckpt_dir} "
            f"free={format_bytes_binary(usage.free_bytes)} ({format_percent(free_ratio)}) "
            f"total={format_bytes_binary(usage.total_bytes)}"
        )
        self._save_to_path(ckpt_path, step, progress, buffer)


def setup_ckpt_manager(output_dir: Path, config: CheckpointConfig | None) -> CheckpointManager | None:
    if config is None:
        return None
    return CheckpointManager(output_dir, config)
