from pathlib import Path

import pytest

from prime_rl.orchestrator.buffer import SimpleBuffer
from prime_rl.orchestrator.ckpt import CheckpointManager, Progress
from prime_rl.orchestrator.config import CheckpointConfig, SimpleBufferConfig


@pytest.fixture
def tmp_ckpt_dir(tmp_path: Path) -> Path:
    """Create a temporary checkpoint directory."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


@pytest.fixture
def checkpoint_config() -> CheckpointConfig:
    """Create a checkpoint config with keep=1 for testing."""
    return CheckpointConfig(keep=1)


@pytest.fixture
def checkpoint_manager(tmp_ckpt_dir: Path, checkpoint_config: CheckpointConfig) -> CheckpointManager:
    """Create a checkpoint manager instance."""
    return CheckpointManager(tmp_ckpt_dir, checkpoint_config)


@pytest.fixture
def progress() -> Progress:
    """Create a progress instance."""
    return Progress(step=0, total_tokens=0, total_samples=0, total_problems=0)


@pytest.fixture
def buffer() -> SimpleBuffer:
    """Create a simple buffer instance."""
    from datasets import Dataset

    dataset = Dataset.from_list([{"problem": "test"}])
    return SimpleBuffer(dataset, SimpleBufferConfig())


# Note: maybe_clean() in orchestrator CheckpointManager now only cleans up for orchestrator-only workloads.
# For workloads with both orchestrator and trainer, cleanup is handled by the trainer CheckpointManager
# since it lags behind the orchestrator.
