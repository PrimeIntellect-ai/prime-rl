"""Unit tests for filesystem transport, specifically FileSystemTrainingBatchReceiver."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from prime_rl.trainer.runs import Progress
from prime_rl.transport.filesystem import FileSystemTrainingBatchReceiver
from prime_rl.transport.types import TrainingBatch, TrainingSample

# Encoder for writing test batch files
_encoder = msgspec.msgpack.Encoder()


def make_sample(prompt_len: int = 2, completion_len: int = 2) -> TrainingSample:
    return TrainingSample(
        prompt_ids=list(range(prompt_len)),
        prompt_mask=[False] * prompt_len,
        completion_ids=list(range(100, 100 + completion_len)),
        completion_mask=[True] * completion_len,
        completion_logprobs=[-0.1] * completion_len,
        teacher_logprobs=None,
        advantage=1.0,
    )


def make_batch(num_examples: int = 3, step: int = 0) -> TrainingBatch:
    return TrainingBatch(
        examples=[make_sample() for _ in range(num_examples)],
        temperature=1.0,
        step=step,
        run_idx=None,  # Set by receiver
    )


@pytest.fixture
def mock_runs() -> MagicMock:
    runs = MagicMock()
    runs.used_idxs = [0]
    runs.ready_to_update = [False, False, False, False]
    runs.progress = {0: Progress(step=0)}
    runs.output_dir = Path("/tmp/test_output")

    def get_run_dir(idx: int) -> Path:
        return runs.output_dir / f"run_{idx}"

    runs.get_run_dir = get_run_dir
    return runs


@pytest.fixture
def receiver(mock_runs: MagicMock, tmp_path: Path) -> FileSystemTrainingBatchReceiver:
    mock_runs.output_dir = tmp_path

    with (
        patch("prime_rl.transport.filesystem.get_runs", return_value=mock_runs),
        patch("prime_rl.transport.base.get_logger", return_value=MagicMock()),
    ):
        recv = FileSystemTrainingBatchReceiver()
        yield recv


# =============================================================================
# Tests for _get_received_step - independent step tracking
# =============================================================================


def test_get_received_step_initializes_from_progress(receiver: FileSystemTrainingBatchReceiver) -> None:
    """First access to _get_received_step should initialize from runs.progress."""
    receiver.runs.progress[0].step = 5

    assert receiver._get_received_step(0) == 5
    assert 0 in receiver._received_steps
    assert receiver._received_steps[0] == 5


def test_get_received_step_returns_tracked_value(receiver: FileSystemTrainingBatchReceiver) -> None:
    """Subsequent accesses should return the tracked value, not runs.progress."""
    receiver.runs.progress[0].step = 0
    receiver._get_received_step(0)  # Initialize

    # Even if runs.progress changes, _received_steps should be independent
    receiver.runs.progress[0].step = 10
    assert receiver._get_received_step(0) == 0


def test_get_received_step_tracks_multiple_runs(receiver: FileSystemTrainingBatchReceiver) -> None:
    """Each run should have its own received step counter."""
    receiver.runs.used_idxs = [0, 1, 2]
    receiver.runs.progress[0] = Progress(step=0)
    receiver.runs.progress[1] = Progress(step=5)
    receiver.runs.progress[2] = Progress(step=10)

    assert receiver._get_received_step(0) == 0
    assert receiver._get_received_step(1) == 5
    assert receiver._get_received_step(2) == 10

    # Modify one, others unchanged
    receiver._received_steps[1] = 6
    assert receiver._get_received_step(0) == 0
    assert receiver._get_received_step(1) == 6
    assert receiver._get_received_step(2) == 10


# =============================================================================
# Tests for receive() - file reading and step incrementing
# =============================================================================


def test_receive_increments_received_step(receiver: FileSystemTrainingBatchReceiver, tmp_path: Path) -> None:
    """After successfully reading a batch, _received_steps should increment."""
    # Setup: create run directory and batch file at step 0
    run_dir = tmp_path / "run_0"
    rollout_dir = run_dir / "rollouts" / "step_0"
    rollout_dir.mkdir(parents=True)

    batch = make_batch(num_examples=2, step=0)
    batch_file = rollout_dir / "rollouts.bin"
    batch_file.write_bytes(_encoder.encode(batch))

    receiver.runs.progress[0].step = 0

    # First receive should read step 0
    batches = receiver.receive()
    assert len(batches) == 1
    assert receiver._received_steps[0] == 1  # Incremented after read


def test_receive_does_not_reread_same_file(receiver: FileSystemTrainingBatchReceiver, tmp_path: Path) -> None:
    """Receive should not re-read the same file on subsequent calls.

    This is the main bug fix test: when trainer step != orchestrator step,
    the receiver should track its own received step to avoid duplicates.
    """
    # Setup: create run directory and batch file at step 0
    run_dir = tmp_path / "run_0"
    rollout_dir = run_dir / "rollouts" / "step_0"
    rollout_dir.mkdir(parents=True)

    batch = make_batch(num_examples=2, step=0)
    batch_file = rollout_dir / "rollouts.bin"
    batch_file.write_bytes(_encoder.encode(batch))

    # Trainer step stays at 0 (simulating buffering scenario)
    receiver.runs.progress[0].step = 0

    # First receive reads the file
    batches1 = receiver.receive()
    assert len(batches1) == 1

    # Second receive should NOT re-read (even though runs.progress[0].step is still 0)
    batches2 = receiver.receive()
    assert len(batches2) == 0  # No new file at step 1


def test_receive_reads_next_step_file(receiver: FileSystemTrainingBatchReceiver, tmp_path: Path) -> None:
    """After reading step N, receiver should look for step N+1."""
    run_dir = tmp_path / "run_0"

    # Create batch files at step 0 and step 1
    for step in [0, 1]:
        rollout_dir = run_dir / "rollouts" / f"step_{step}"
        rollout_dir.mkdir(parents=True)
        batch = make_batch(num_examples=2, step=step)
        batch_file = rollout_dir / "rollouts.bin"
        batch_file.write_bytes(_encoder.encode(batch))

    receiver.runs.progress[0].step = 0

    # First receive reads step 0
    batches1 = receiver.receive()
    assert len(batches1) == 1

    # Second receive reads step 1
    batches2 = receiver.receive()
    assert len(batches2) == 1


def test_receive_skips_ready_to_update_runs(receiver: FileSystemTrainingBatchReceiver, tmp_path: Path) -> None:
    """Runs with ready_to_update=True should be skipped."""
    run_dir = tmp_path / "run_0"
    rollout_dir = run_dir / "rollouts" / "step_0"
    rollout_dir.mkdir(parents=True)

    batch = make_batch(num_examples=2, step=0)
    batch_file = rollout_dir / "rollouts.bin"
    batch_file.write_bytes(_encoder.encode(batch))

    receiver.runs.progress[0].step = 0
    receiver.runs.ready_to_update[0] = True

    batches = receiver.receive()
    assert len(batches) == 0  # Skipped because ready_to_update


def test_receive_multiple_runs_independent_steps(receiver: FileSystemTrainingBatchReceiver, tmp_path: Path) -> None:
    """Multiple runs should track received steps independently."""
    receiver.runs.used_idxs = [0, 1]
    receiver.runs.progress[0] = Progress(step=0)
    receiver.runs.progress[1] = Progress(step=0)

    # Create batch files: run_0 at step 0, run_1 at steps 0 and 1
    for run_idx in [0, 1]:
        run_dir = tmp_path / f"run_{run_idx}"
        for step in range(2):
            rollout_dir = run_dir / "rollouts" / f"step_{step}"
            rollout_dir.mkdir(parents=True)
            batch = make_batch(num_examples=1, step=step)
            batch_file = rollout_dir / "rollouts.bin"
            batch_file.write_bytes(_encoder.encode(batch))

    # First receive: both runs read step 0
    batches1 = receiver.receive()
    assert len(batches1) == 2
    assert receiver._received_steps[0] == 1
    assert receiver._received_steps[1] == 1

    # Second receive: both runs try step 1, only run_1 has it (run_0 doesn't)
    # Actually both have step 1, so both should read
    batches2 = receiver.receive()
    assert len(batches2) == 2
    assert receiver._received_steps[0] == 2
    assert receiver._received_steps[1] == 2


# =============================================================================
# Tests for checkpoint resume scenario
# =============================================================================


def test_receive_resumes_from_checkpoint_step(receiver: FileSystemTrainingBatchReceiver, tmp_path: Path) -> None:
    """On checkpoint resume, receiver should start from runs.progress step."""
    run_dir = tmp_path / "run_0"

    # Simulate checkpoint resume: trainer is at step 5
    receiver.runs.progress[0].step = 5

    # Create batch file at step 5
    rollout_dir = run_dir / "rollouts" / "step_5"
    rollout_dir.mkdir(parents=True)
    batch = make_batch(num_examples=2, step=5)
    batch_file = rollout_dir / "rollouts.bin"
    batch_file.write_bytes(_encoder.encode(batch))

    # Receiver should look for step 5, not step 0
    batches = receiver.receive()
    assert len(batches) == 1
    assert receiver._received_steps[0] == 6
