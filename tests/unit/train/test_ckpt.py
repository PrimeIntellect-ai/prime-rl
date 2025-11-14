from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from torch import nn
from torch.optim import Adam

from prime_rl.trainer.ckpt import CheckpointManager, Progress
from prime_rl.trainer.config import CheckpointConfig


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
def mock_world():
    """Create a mock world for the checkpoint manager."""
    with patch("prime_rl.trainer.ckpt.get_world") as mock_get_world:
        mock_world = MagicMock()
        mock_world.is_master = True
        mock_world.rank = 0
        mock_get_world.return_value = mock_world
        yield mock_world


@pytest.fixture
def checkpoint_manager(tmp_ckpt_dir: Path, checkpoint_config: CheckpointConfig, mock_world) -> CheckpointManager:
    """Create a checkpoint manager instance."""
    return CheckpointManager(tmp_ckpt_dir, checkpoint_config)


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple model for testing."""
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))


@pytest.fixture
def progress() -> Progress:
    """Create a progress instance."""
    return Progress(step=0, total_tokens=0, total_samples=0)


def test_get_step_path(checkpoint_manager: CheckpointManager, tmp_ckpt_dir: Path):
    """Test that get_step_path returns the correct path."""
    step = 5
    expected_path = tmp_ckpt_dir / "checkpoints" / "step_5"
    assert checkpoint_manager.get_step_path(step) == expected_path


def test_is_complete_checkpoint_both_exist(checkpoint_manager: CheckpointManager, tmp_ckpt_dir: Path):
    """Test that _is_complete_checkpoint returns True when both orchestrator and trainer checkpoints exist."""
    step = 1
    step_path = checkpoint_manager.get_step_path(step)
    step_path.mkdir(parents=True, exist_ok=True)

    # Create both orchestrator and trainer checkpoints
    (step_path / "orchestrator").mkdir(exist_ok=True)
    (step_path / "trainer").mkdir(exist_ok=True)

    assert checkpoint_manager._is_complete_checkpoint(step) is True


def test_is_complete_checkpoint_only_trainer(checkpoint_manager: CheckpointManager, tmp_ckpt_dir: Path):
    """Test that _is_complete_checkpoint returns False when only trainer checkpoint exists."""
    step = 1
    step_path = checkpoint_manager.get_step_path(step)
    step_path.mkdir(parents=True, exist_ok=True)

    # Create only trainer checkpoint
    (step_path / "trainer").mkdir(exist_ok=True)

    assert checkpoint_manager._is_complete_checkpoint(step) is False


def test_is_complete_checkpoint_only_orchestrator(checkpoint_manager: CheckpointManager, tmp_ckpt_dir: Path):
    """Test that _is_complete_checkpoint returns False when only orchestrator checkpoint exists."""
    step = 1
    step_path = checkpoint_manager.get_step_path(step)
    step_path.mkdir(parents=True, exist_ok=True)

    # Create only orchestrator checkpoint
    (step_path / "orchestrator").mkdir(exist_ok=True)

    assert checkpoint_manager._is_complete_checkpoint(step) is False


def test_is_complete_checkpoint_neither(checkpoint_manager: CheckpointManager, tmp_ckpt_dir: Path):
    """Test that _is_complete_checkpoint returns False when neither checkpoint exists."""
    step = 1
    step_path = checkpoint_manager.get_step_path(step)
    step_path.mkdir(parents=True, exist_ok=True)

    assert checkpoint_manager._is_complete_checkpoint(step) is False


def test_maybe_clean_skips_when_incomplete(
    checkpoint_manager: CheckpointManager, simple_model: nn.Module, progress: Progress, tmp_ckpt_dir: Path
):
    """Test that maybe_clean skips deletion when the newest kept checkpoint is incomplete."""
    optimizer = Adam(simple_model.parameters())

    # Save checkpoints for steps 1 and 2
    checkpoint_manager.save(simple_model, [optimizer], None, progress, step=1)
    checkpoint_manager.save(simple_model, [optimizer], None, progress, step=2)

    # Only create trainer checkpoint for step 2 (incomplete)
    step2_path = checkpoint_manager.get_step_path(2)
    step2_path.mkdir(parents=True, exist_ok=True)
    (step2_path / "trainer").mkdir(exist_ok=True)

    # Create orchestrator checkpoint for step 1 (complete)
    step1_path = checkpoint_manager.get_step_path(1)
    (step1_path / "orchestrator").mkdir(exist_ok=True)

    # Try to clean - should skip because step 2 is incomplete
    checkpoint_manager.maybe_clean()

    # Both checkpoints should still exist
    assert checkpoint_manager.get_ckpt_path(1).exists()
    assert checkpoint_manager.get_ckpt_path(2).exists()


def test_maybe_clean_deletes_when_complete(
    checkpoint_manager: CheckpointManager, simple_model: nn.Module, progress: Progress, tmp_ckpt_dir: Path
):
    """Test that maybe_clean deletes old checkpoints when a complete checkpoint exists."""
    optimizer = Adam(simple_model.parameters())

    # Save checkpoints for steps 1, 2, and 3
    checkpoint_manager.save(simple_model, [optimizer], None, progress, step=1)
    checkpoint_manager.save(simple_model, [optimizer], None, progress, step=2)
    checkpoint_manager.save(simple_model, [optimizer], None, progress, step=3)

    # Create complete checkpoints for all steps
    for step in [1, 2, 3]:
        step_path = checkpoint_manager.get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        (step_path / "orchestrator").mkdir(exist_ok=True)
        (step_path / "trainer").mkdir(exist_ok=True)

    # Clean should delete step 1 (older than newest complete step 3, and we keep only 1)
    checkpoint_manager.maybe_clean()

    # Step 1 should be deleted, steps 2 and 3 should remain
    assert not checkpoint_manager.get_step_path(1).exists()
    assert checkpoint_manager.get_step_path(2).exists()
    assert checkpoint_manager.get_step_path(3).exists()


def test_maybe_clean_with_keep_1_ensures_complete_checkpoint(
    checkpoint_manager: CheckpointManager, simple_model: nn.Module, progress: Progress, tmp_ckpt_dir: Path
):
    """Test that with keep=1, at least one complete checkpoint always exists."""
    optimizer = Adam(simple_model.parameters())

    # Save multiple checkpoints
    for step in range(1, 5):
        checkpoint_manager.save(simple_model, [optimizer], None, progress, step=step)

    # Create incomplete checkpoint for step 4 (only trainer)
    step4_path = checkpoint_manager.get_step_path(4)
    step4_path.mkdir(parents=True, exist_ok=True)
    (step4_path / "trainer").mkdir(exist_ok=True)

    # Create complete checkpoint for step 3
    step3_path = checkpoint_manager.get_step_path(3)
    step3_path.mkdir(parents=True, exist_ok=True)
    (step3_path / "orchestrator").mkdir(exist_ok=True)
    (step3_path / "trainer").mkdir(exist_ok=True)

    # Clean should skip because no complete checkpoint in kept steps (step 4 is incomplete)
    checkpoint_manager.maybe_clean()

    # All steps should still exist (nothing deleted)
    for step in range(1, 5):
        assert checkpoint_manager.get_step_path(step).exists()

    # Now make step 4 complete
    (step4_path / "orchestrator").mkdir(exist_ok=True)

    # Clean should now work - newest complete is step 4, delete steps 1, 2, 3 (older than 4)
    # But keep last keep=1 of those older ones, so keep step 3, delete steps 1 and 2
    checkpoint_manager.maybe_clean()

    # Steps 1 and 2 should be deleted, steps 3 and 4 should remain
    assert not checkpoint_manager.get_step_path(1).exists()
    assert not checkpoint_manager.get_step_path(2).exists()
    assert checkpoint_manager.get_step_path(3).exists()
    assert checkpoint_manager.get_step_path(4).exists()
    assert checkpoint_manager._is_complete_checkpoint(4) is True


def test_maybe_clean_with_keep_none_does_nothing(
    tmp_ckpt_dir: Path, simple_model: nn.Module, progress: Progress, mock_world
):
    """Test that maybe_clean does nothing when keep is None."""
    config = CheckpointConfig(keep=None)
    manager = CheckpointManager(tmp_ckpt_dir, config)
    optimizer = Adam(simple_model.parameters())

    # Save multiple checkpoints
    for step in range(1, 4):
        manager.save(simple_model, [optimizer], None, progress, step=step)

    # Clean should do nothing
    manager.maybe_clean()

    # All checkpoints should still exist
    for step in range(1, 4):
        assert manager.get_ckpt_path(step).exists()


def test_maybe_clean_with_different_intervals(
    tmp_ckpt_dir: Path, simple_model: nn.Module, progress: Progress, mock_world
):
    """Test that cleanup works correctly with different checkpoint intervals."""
    config = CheckpointConfig(keep=2)
    manager = CheckpointManager(tmp_ckpt_dir, config)
    optimizer = Adam(simple_model.parameters())

    # Save checkpoints at different intervals (steps 1, 3, 5, 7)
    for step in [1, 3, 5, 7]:
        manager.save(simple_model, [optimizer], None, progress, step=step)
        step_path = manager.get_step_path(step)
        step_path.mkdir(parents=True, exist_ok=True)
        (step_path / "orchestrator").mkdir(exist_ok=True)
        (step_path / "trainer").mkdir(exist_ok=True)

    # Newest complete is step 7, steps older than 7 are [1, 3, 5]
    # Keep last keep=2 of those older ones, so keep [3, 5], delete [1]
    manager.maybe_clean()

    assert not manager.get_step_path(1).exists()
    assert manager.get_step_path(3).exists()
    assert manager.get_step_path(5).exists()
    assert manager.get_step_path(7).exists()


def test_maybe_clean_master_rank_only_logging(tmp_ckpt_dir: Path, simple_model: nn.Module, progress: Progress):
    """Test that logging only happens on master rank."""
    optimizer = Adam(simple_model.parameters())

    # Test with non-master rank
    with patch("prime_rl.trainer.ckpt.get_world") as mock_get_world:
        mock_world = MagicMock()
        mock_world.is_master = False
        mock_world.rank = 1
        mock_get_world.return_value = mock_world

        config = CheckpointConfig(keep=1)
        manager = CheckpointManager(tmp_ckpt_dir, config)

        # Save checkpoints
        manager.save(simple_model, [optimizer], None, progress, step=1)
        manager.save(simple_model, [optimizer], None, progress, step=2)

        # Create incomplete checkpoint for step 2
        step2_path = manager.get_step_path(2)
        step2_path.mkdir(parents=True, exist_ok=True)
        (step2_path / "trainer").mkdir(exist_ok=True)

        # Create complete checkpoint for step 1
        step1_path = manager.get_step_path(1)
        (step1_path / "orchestrator").mkdir(exist_ok=True)

        # Clean should skip (no logging on non-master)
        manager.maybe_clean()

        # Both should still exist
        assert manager.get_ckpt_path(1).exists()
        assert manager.get_ckpt_path(2).exists()


def test_maybe_clean_empty_checkpoint_steps(checkpoint_manager: CheckpointManager):
    """Test that maybe_clean handles empty checkpoint steps gracefully."""
    checkpoint_manager.ckpt_steps = []
    # Should not raise an error
    checkpoint_manager.maybe_clean()


def test_maybe_clean_no_checkpoints_to_delete(
    checkpoint_manager: CheckpointManager, simple_model: nn.Module, progress: Progress, tmp_ckpt_dir: Path
):
    """Test that maybe_clean handles the case when there are no checkpoints to delete."""
    optimizer = Adam(simple_model.parameters())

    # Save only one checkpoint
    checkpoint_manager.save(simple_model, [optimizer], None, progress, step=1)
    step_path = checkpoint_manager.get_step_path(1)
    step_path.mkdir(parents=True, exist_ok=True)
    (step_path / "orchestrator").mkdir(exist_ok=True)
    (step_path / "trainer").mkdir(exist_ok=True)

    # Clean should do nothing (keep=1, only 1 checkpoint)
    checkpoint_manager.maybe_clean()

    # Checkpoint should still exist
    assert checkpoint_manager.get_ckpt_path(1).exists()
