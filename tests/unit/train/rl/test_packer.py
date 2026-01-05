"""Unit tests for the Packer class.

Tests the Packer class with mocked dependencies to verify:
1. get_batch() correctly filters and returns batches by run_idx
2. has_enough_tokens() correctly calculates token thresholds
3. pack() updates progress, sets ready_to_update, and sends micro_batch_grid
4. Critical invariants are maintained through the packing process
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prime_rl.transport.types import TrainingBatch, TrainingSample


@pytest.fixture
def make_training_sample():
    """Factory for creating TrainingSample with configurable sizes."""

    def _make(
        prompt_len: int = 2,
        completion_len: int = 2,
        advantage: float = 1.0,
    ) -> TrainingSample:
        return TrainingSample(
            prompt_ids=list(range(1, prompt_len + 1)),
            prompt_mask=[False] * prompt_len,
            completion_ids=list(range(100, 100 + completion_len)),
            completion_mask=[True] * completion_len,
            completion_logprobs=[-0.1] * completion_len,
            teacher_logprobs=None,
            advantage=advantage,
        )

    return _make


@pytest.fixture
def make_training_batch(make_training_sample):
    """Factory for creating TrainingBatch."""

    def _make(
        num_examples: int = 3,
        temperature: float = 1.0,
        step: int = 1,
        run_idx: int | None = 0,
        prompt_len: int = 2,
        completion_len: int = 2,
    ) -> TrainingBatch:
        examples = [
            make_training_sample(prompt_len=prompt_len, completion_len=completion_len) for _ in range(num_examples)
        ]
        return TrainingBatch(
            examples=examples,
            temperature=temperature,
            step=step,
            run_idx=run_idx,
        )

    return _make


@pytest.fixture
def mock_progress():
    """Create a mock progress object."""
    progress = MagicMock()
    progress.step = 0
    progress.total_tokens = 0
    progress.total_samples = 0
    return progress


@pytest.fixture
def mock_runs(mock_progress):
    """Mock the Runs singleton for Packer tests."""
    runs = MagicMock()
    runs.used_idxs = [0]
    runs.max_runs = 4
    runs.ready_to_update = [False, False, False, False]
    runs.progress = {0: mock_progress}
    runs.output_dir = Path("/tmp/test_output")
    runs.check_for_changes = MagicMock()
    return runs


@pytest.fixture
def mock_receiver():
    """Mock TrainingBatchReceiver."""
    receiver = MagicMock()
    receiver.receive = MagicMock(return_value=[])
    return receiver


@pytest.fixture
def mock_sender():
    """Mock MicroBatchSender."""
    sender = MagicMock()
    sender.send = MagicMock()
    return sender


@pytest.fixture
def mock_tokenizer():
    """Mock PreTrainedTokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_config():
    """Mock TransportConfigType."""
    return MagicMock()


@pytest.fixture
def packer_with_mocks(mock_runs, mock_receiver, mock_sender, mock_tokenizer, mock_config, tmp_path):
    """Create Packer with all dependencies mocked."""
    from prime_rl.trainer.rl.packer import Packer

    mock_runs.output_dir = tmp_path

    with (
        patch("prime_rl.trainer.rl.packer.get_logger", return_value=MagicMock()),
        patch("prime_rl.trainer.rl.packer.get_runs", return_value=mock_runs),
        patch("prime_rl.trainer.rl.packer.setup_training_batch_receiver", return_value=mock_receiver),
        patch("prime_rl.trainer.rl.packer.setup_micro_batch_sender", return_value=mock_sender),
        patch("prime_rl.trainer.rl.packer.get_rollout_dir", return_value=tmp_path / "rollouts"),
        patch("shutil.rmtree"),
    ):
        packer = Packer(
            dp_world_size=2,
            seq_len=100,
            pad_to_multiple_of=8,
            tokenizer=mock_tokenizer,
            config=mock_config,
            start_step=0,
        )
        packer._mock_runs = mock_runs
        packer._mock_receiver = mock_receiver
        packer._mock_sender = mock_sender
        yield packer


# =============================================================================
# Tests for Packer.get_batch()
# =============================================================================


def test_get_batch_returns_dict_keyed_by_run_idx(packer_with_mocks, make_training_batch):
    """Verify get_batch returns dict keyed by run_idx."""
    batch = make_training_batch(run_idx=0)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]

    result = packer_with_mocks.get_batch()

    assert isinstance(result, dict)
    assert 0 in result
    assert result[0] == batch


def test_get_batch_filters_none_run_idx(packer_with_mocks, make_training_batch):
    """Verify batches with None run_idx are filtered out."""
    batch_valid = make_training_batch(run_idx=0)
    batch_invalid = make_training_batch(run_idx=None)
    packer_with_mocks._mock_receiver.receive.return_value = [batch_valid, batch_invalid]

    result = packer_with_mocks.get_batch()

    assert len(result) == 1
    assert 0 in result
    assert None not in result


def test_get_batch_calls_check_for_changes(packer_with_mocks):
    """Verify check_for_changes is called."""
    packer_with_mocks._mock_receiver.receive.return_value = []

    packer_with_mocks.get_batch()

    packer_with_mocks._mock_runs.check_for_changes.assert_called_once()


def test_get_batch_multiple_run_idxs(packer_with_mocks, make_training_batch):
    """Verify handling of multiple run_idx values."""
    batch0 = make_training_batch(run_idx=0)
    batch1 = make_training_batch(run_idx=1)
    batch2 = make_training_batch(run_idx=2)
    packer_with_mocks._mock_receiver.receive.return_value = [batch0, batch1, batch2]

    result = packer_with_mocks.get_batch()

    assert len(result) == 3
    assert 0 in result
    assert 1 in result
    assert 2 in result


def test_get_batch_empty_receiver(packer_with_mocks):
    """Verify handling of empty receiver results."""
    packer_with_mocks._mock_receiver.receive.return_value = []

    result = packer_with_mocks.get_batch()

    assert result == {}


# =============================================================================
# Tests for Packer.has_enough_tokens()
# =============================================================================


def test_has_enough_tokens_returns_false_when_empty(packer_with_mocks):
    """Verify empty dict returns False."""
    result = packer_with_mocks.has_enough_tokens({})

    assert result is False


def test_has_enough_tokens_below_threshold(packer_with_mocks, make_training_batch):
    """Verify returns False when below threshold."""
    batch = make_training_batch(num_examples=3, prompt_len=2, completion_len=2)
    rollouts = {0: batch}

    result = packer_with_mocks.has_enough_tokens(rollouts)

    assert result is False


def test_has_enough_tokens_above_threshold(packer_with_mocks, make_training_batch):
    """Verify returns True when above threshold."""
    batch = make_training_batch(num_examples=60, prompt_len=2, completion_len=2)
    rollouts = {0: batch}

    result = packer_with_mocks.has_enough_tokens(rollouts)

    assert result is True


def test_has_enough_tokens_threshold_calculation(packer_with_mocks, make_training_batch):
    """Verify threshold is seq_len * dp_world_size."""
    batch = make_training_batch(num_examples=26, prompt_len=2, completion_len=2)
    rollouts = {0: batch}

    result = packer_with_mocks.has_enough_tokens(rollouts)

    assert result is True


def test_has_enough_tokens_multiple_batches(packer_with_mocks, make_training_batch):
    """Verify token counting across multiple batches."""
    batch0 = make_training_batch(num_examples=10, prompt_len=2, completion_len=2)
    batch1 = make_training_batch(num_examples=10, prompt_len=2, completion_len=2)
    rollouts = {0: batch0, 1: batch1}

    result = packer_with_mocks.has_enough_tokens(rollouts)

    assert result is False


# =============================================================================
# Tests for Packer.pack()
# =============================================================================


def test_pack_updates_progress_step(packer_with_mocks, make_training_batch):
    """Verify progress step is incremented."""
    batch = make_training_batch(num_examples=60, prompt_len=2, completion_len=2)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]
    initial_step = packer_with_mocks._mock_runs.progress[0].step

    packer_with_mocks.pack()

    assert packer_with_mocks._mock_runs.progress[0].step == initial_step + 1


def test_pack_updates_progress_total_tokens(packer_with_mocks, make_training_batch):
    """Verify progress total_tokens is updated."""
    batch = make_training_batch(num_examples=10, prompt_len=2, completion_len=2)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]
    packer_with_mocks._mock_runs.progress[0].total_tokens = 0

    packer_with_mocks.pack()

    assert packer_with_mocks._mock_runs.progress[0].total_tokens == 40


def test_pack_updates_progress_total_samples(packer_with_mocks, make_training_batch):
    """Verify progress total_samples is updated."""
    batch = make_training_batch(num_examples=10, prompt_len=2, completion_len=2)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]
    packer_with_mocks._mock_runs.progress[0].total_samples = 0

    packer_with_mocks.pack()

    assert packer_with_mocks._mock_runs.progress[0].total_samples == 10


def test_pack_sets_ready_to_update_flag(packer_with_mocks, make_training_batch):
    """Verify ready_to_update flag is set for the run."""
    batch = make_training_batch(num_examples=60, run_idx=0)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]
    packer_with_mocks._mock_runs.ready_to_update = [False, False, False, False]

    packer_with_mocks.pack()

    assert packer_with_mocks._mock_runs.ready_to_update[0] is True


def test_pack_calls_sender_with_micro_batch_grid(packer_with_mocks, make_training_batch):
    """Verify sender.send is called with micro_batch_grid."""
    batch = make_training_batch(num_examples=60, prompt_len=2, completion_len=2)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]

    packer_with_mocks.pack()

    packer_with_mocks._mock_sender.send.assert_called_once()
    sent_grid = packer_with_mocks._mock_sender.send.call_args[0][0]
    assert isinstance(sent_grid, list)
    assert len(sent_grid) == 2


def test_pack_asserts_no_ready_runs_at_start(packer_with_mocks, make_training_batch):
    """Verify assertion fails if any run is already ready."""
    batch = make_training_batch(num_examples=60)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]
    packer_with_mocks._mock_runs.ready_to_update = [True, False, False, False]

    with pytest.raises(AssertionError, match="No runs should be ready to update"):
        packer_with_mocks.pack()


def test_pack_handles_multiple_runs(packer_with_mocks, make_training_batch, mock_progress):
    """Verify handling of multiple concurrent runs."""
    batch0 = make_training_batch(num_examples=30, run_idx=0)
    batch1 = make_training_batch(num_examples=30, run_idx=1)
    packer_with_mocks._mock_receiver.receive.return_value = [batch0, batch1]

    progress1 = MagicMock()
    progress1.step = 0
    progress1.total_tokens = 0
    progress1.total_samples = 0
    packer_with_mocks._mock_runs.progress[1] = progress1

    packer_with_mocks.pack()

    assert packer_with_mocks._mock_runs.progress[0].step == 1
    assert packer_with_mocks._mock_runs.progress[1].step == 1


def test_pack_sends_grid_with_equal_batch_counts(packer_with_mocks, make_training_batch):
    """Critical: Verify all dp ranks receive equal batch counts."""
    batch = make_training_batch(num_examples=60, prompt_len=2, completion_len=2)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]

    packer_with_mocks.pack()

    sent_grid = packer_with_mocks._mock_sender.send.call_args[0][0]
    batch_counts = [len(rank_batches) for rank_batches in sent_grid]
    assert len(set(batch_counts)) == 1, f"Unequal batch counts: {batch_counts}"


def test_pack_sends_non_empty_batches_per_rank(packer_with_mocks, make_training_batch):
    """Critical: Verify no dp rank receives an empty batch list."""
    batch = make_training_batch(num_examples=60, prompt_len=2, completion_len=2)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]

    packer_with_mocks.pack()

    sent_grid = packer_with_mocks._mock_sender.send.call_args[0][0]
    assert all(len(rank_batches) > 0 for rank_batches in sent_grid), "Empty batch list found"


def test_pack_lora_num_tokens_invariant(packer_with_mocks, make_training_batch):
    """Critical: Verify lora_num_tokens sum equals input_ids length."""
    batch = make_training_batch(num_examples=60, prompt_len=2, completion_len=2)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]

    packer_with_mocks.pack()

    sent_grid = packer_with_mocks._mock_sender.send.call_args[0][0]
    for rank_batches in sent_grid:
        for mb in rank_batches:
            assert sum(mb.lora_num_tokens) == len(mb.input_ids)


# =============================================================================
# Tests for edge cases
# =============================================================================


def test_pack_with_timeout_warning(packer_with_mocks, make_training_batch):
    """Verify timeout warning is logged when tokens are insufficient."""
    batch = make_training_batch(num_examples=3, prompt_len=2, completion_len=2)

    call_count = 0

    def mock_receive():
        nonlocal call_count
        call_count += 1
        return [batch]

    packer_with_mocks._mock_receiver.receive = mock_receive

    with patch("prime_rl.trainer.rl.packer.time") as mock_time:
        times = [0.0, 0.0, 100.0]
        mock_time.time.side_effect = times
        mock_time.sleep = MagicMock()

        packer_with_mocks.pack()

    packer_with_mocks._mock_sender.send.assert_called_once()


def test_pack_with_sparse_run_idxs(packer_with_mocks, make_training_batch):
    """Verify handling of non-contiguous run_idx values."""
    batch0 = make_training_batch(num_examples=30, run_idx=0)
    batch3 = make_training_batch(num_examples=30, run_idx=3)
    packer_with_mocks._mock_receiver.receive.return_value = [batch0, batch3]

    progress3 = MagicMock()
    progress3.step = 0
    progress3.total_tokens = 0
    progress3.total_samples = 0
    packer_with_mocks._mock_runs.progress[3] = progress3

    packer_with_mocks.pack()

    assert packer_with_mocks._mock_runs.ready_to_update[0] is True
    assert packer_with_mocks._mock_runs.ready_to_update[3] is True


def test_pack_with_single_sample(packer_with_mocks, make_training_batch):
    """Verify handling of single sample batch."""
    batch = make_training_batch(num_examples=1, prompt_len=50, completion_len=50)
    packer_with_mocks._mock_receiver.receive.return_value = [batch]

    packer_with_mocks.pack()

    packer_with_mocks._mock_sender.send.assert_called_once()
    sent_grid = packer_with_mocks._mock_sender.send.call_args[0][0]
    assert all(len(rank_batches) > 0 for rank_batches in sent_grid)
