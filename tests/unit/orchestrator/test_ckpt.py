from unittest.mock import MagicMock

from prime_rl.orchestrator.ckpt import CheckpointManager, Progress
from prime_rl.orchestrator.config import CheckpointConfig


def test_checkpoint_round_trip_preserves_step(tmp_path):
    ckpt_manager = CheckpointManager(output_dir=tmp_path, config=CheckpointConfig())
    progress = Progress(step=5, total_tokens=10, total_samples=15, total_problems=20)
    buffer = MagicMock()

    ckpt_manager.save(progress, buffer, step=5)

    loaded_progress = Progress()
    loaded_buffer = MagicMock()
    ckpt_manager.load(loaded_progress, loaded_buffer, step=5)

    assert loaded_progress.step == 5
    assert loaded_progress.total_tokens == 10
    assert loaded_progress.total_samples == 15
    assert loaded_progress.total_problems == 20
