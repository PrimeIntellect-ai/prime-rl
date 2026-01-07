"""Unit tests for the Packer class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prime_rl.trainer.rl.packer import Packer
from prime_rl.trainer.runs import Progress
from prime_rl.transport.types import TrainingBatch, TrainingSample


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


def make_batch(
    num_examples: int = 3,
    run_idx: int | None = 0,
    prompt_len: int = 2,
    completion_len: int = 2,
) -> TrainingBatch:
    return TrainingBatch(
        examples=[make_sample(prompt_len, completion_len) for _ in range(num_examples)],
        temperature=1.0,
        step=1,
        run_idx=run_idx,
    )


@pytest.fixture
def mock_runs() -> MagicMock:
    runs = MagicMock()
    runs.used_idxs = [0]
    runs.max_runs = 4
    runs.ready_to_update = [False, False, False, False]
    runs.progress = {0: Progress()}
    runs.output_dir = Path("/tmp/test_output")
    return runs


@pytest.fixture
def packer(mock_runs: MagicMock, tmp_path: Path) -> Packer:
    mock_runs.output_dir = tmp_path
    mock_receiver = MagicMock()
    mock_receiver.receive.return_value = []
    mock_sender = MagicMock()

    with (
        # Avoid log noise in tests
        patch("prime_rl.trainer.rl.packer.get_logger", return_value=MagicMock()),
        # Runs is a singleton with global state; need isolated state per test
        patch("prime_rl.trainer.rl.packer.get_runs", return_value=mock_runs),
        # Avoid filesystem/network I/O; lets us inject specific TrainingBatch inputs
        patch("prime_rl.trainer.rl.packer.setup_training_batch_receiver", return_value=mock_receiver),
        # Avoid filesystem/network I/O; lets us inspect what micro_batch_grid was sent
        patch("prime_rl.trainer.rl.packer.setup_micro_batch_sender", return_value=mock_sender),
    ):
        p = Packer(
            dp_world_size=2,
            seq_len=100,
            pad_to_multiple_of=8,
            tokenizer=MagicMock(pad_token_id=0),  # Only pad_token_id is used
            config=MagicMock(),
            start_step=0,
        )
        p._runs = mock_runs
        p._receiver = mock_receiver
        p._sender = mock_sender
        yield p


# =============================================================================
# Tests for Packer.get_batch()
# =============================================================================


def test_get_batch_returns_dict_keyed_by_run_idx(packer: Packer) -> None:
    batch = make_batch(run_idx=0)
    packer._receiver.receive.return_value = [batch]

    result = packer.get_batch()

    assert isinstance(result, dict)
    assert 0 in result
    assert result[0] == batch


def test_get_batch_filters_none_run_idx(packer: Packer) -> None:
    batch_valid = make_batch(run_idx=0)
    batch_invalid = make_batch(run_idx=None)
    packer._receiver.receive.return_value = [batch_valid, batch_invalid]

    result = packer.get_batch()

    assert len(result) == 1
    assert 0 in result
    assert None not in result


def test_get_batch_calls_check_for_changes(packer: Packer) -> None:
    packer._receiver.receive.return_value = []
    packer.get_batch()
    packer._runs.check_for_changes.assert_called_once()


def test_get_batch_multiple_run_idxs(packer: Packer) -> None:
    packer._receiver.receive.return_value = [
        make_batch(run_idx=0),
        make_batch(run_idx=1),
        make_batch(run_idx=2),
    ]

    result = packer.get_batch()

    assert len(result) == 3
    assert 0 in result
    assert 1 in result
    assert 2 in result


def test_get_batch_empty_receiver(packer: Packer) -> None:
    packer._receiver.receive.return_value = []

    result = packer.get_batch()

    assert result == {}


# =============================================================================
# Tests for Packer.has_enough_tokens()
# =============================================================================


def test_has_enough_tokens_returns_false_when_empty(packer: Packer) -> None:
    assert packer.has_enough_tokens({}) is False


def test_has_enough_tokens_below_threshold(packer: Packer) -> None:
    # 3 examples * 4 tokens = 12 tokens, threshold = 100 * 2 = 200
    rollouts = {0: make_batch(num_examples=3)}

    assert packer.has_enough_tokens(rollouts) is False


def test_has_enough_tokens_above_threshold(packer: Packer) -> None:
    # 60 examples * 4 tokens = 240 tokens, threshold = 200
    rollouts = {0: make_batch(num_examples=60)}

    assert packer.has_enough_tokens(rollouts) is True


def test_has_enough_tokens_threshold_calculation(packer: Packer) -> None:
    # 26 examples * 4 tokens = 104 tokens per batch
    # With estimation logic, this should exceed threshold
    rollouts = {0: make_batch(num_examples=26)}

    assert packer.has_enough_tokens(rollouts) is True


def test_has_enough_tokens_multiple_batches(packer: Packer) -> None:
    # 2 batches * 10 examples * 4 tokens = 80 tokens, threshold = 200
    rollouts = {
        0: make_batch(num_examples=10),
        1: make_batch(num_examples=10),
    }

    assert packer.has_enough_tokens(rollouts) is False


# =============================================================================
# Tests for Packer.pack()
# =============================================================================


def test_pack_updates_progress_step(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]
    initial_step = packer._runs.progress[0].step

    packer.pack()

    assert packer._runs.progress[0].step == initial_step + 1


def test_pack_updates_progress_total_tokens(packer: Packer) -> None:
    # 10 examples * 4 tokens = 40 tokens
    packer._receiver.receive.return_value = [make_batch(num_examples=10)]
    packer._runs.progress[0].total_tokens = 0

    packer.pack()

    assert packer._runs.progress[0].total_tokens == 40


def test_pack_updates_progress_total_samples(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=10)]
    packer._runs.progress[0].total_samples = 0

    packer.pack()

    assert packer._runs.progress[0].total_samples == 10


def test_pack_sets_ready_to_update_flag(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=60, run_idx=0)]
    packer._runs.ready_to_update = [False, False, False, False]

    packer.pack()

    assert packer._runs.ready_to_update[0] is True


def test_pack_calls_sender_with_micro_batch_grid(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]

    packer.pack()

    packer._sender.send.assert_called_once()
    sent_grid = packer._sender.send.call_args[0][0]
    assert isinstance(sent_grid, list)
    assert len(sent_grid) == 2  # dp_world_size


def test_pack_asserts_no_ready_runs_at_start(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]
    packer._runs.ready_to_update = [True, False, False, False]

    with pytest.raises(AssertionError, match="No runs should be ready to update"):
        packer.pack()


def test_pack_handles_multiple_runs(packer: Packer) -> None:
    packer._receiver.receive.return_value = [
        make_batch(num_examples=30, run_idx=0),
        make_batch(num_examples=30, run_idx=1),
    ]
    packer._runs.progress[1] = Progress()

    packer.pack()

    assert packer._runs.progress[0].step == 1
    assert packer._runs.progress[1].step == 1


def test_pack_sends_grid_with_equal_batch_counts(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]

    packer.pack()

    sent_grid = packer._sender.send.call_args[0][0]
    batch_counts = [len(rank_batches) for rank_batches in sent_grid]
    assert len(set(batch_counts)) == 1, f"Unequal batch counts: {batch_counts}"


def test_pack_sends_non_empty_batches_per_rank(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]

    packer.pack()

    sent_grid = packer._sender.send.call_args[0][0]
    assert all(len(rank_batches) > 0 for rank_batches in sent_grid)


def test_pack_lora_num_tokens_invariant(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]

    packer.pack()

    sent_grid = packer._sender.send.call_args[0][0]
    for rank_batches in sent_grid:
        for mb in rank_batches:
            assert sum(mb.lora_num_tokens) == len(mb.input_ids)


# =============================================================================
# Tests for edge cases
# =============================================================================


def test_pack_with_timeout_warning(packer: Packer) -> None:
    batch = make_batch(num_examples=3)

    call_count = 0

    def mock_receive():
        nonlocal call_count
        call_count += 1
        return [batch]

    packer._receiver.receive = mock_receive

    with patch("prime_rl.trainer.rl.packer.time") as mock_time:
        mock_time.time.side_effect = [0.0, 0.0, 100.0]
        mock_time.sleep = MagicMock()

        packer.pack()

    packer._sender.send.assert_called_once()


def test_pack_with_sparse_run_idxs(packer: Packer) -> None:
    packer._receiver.receive.return_value = [
        make_batch(num_examples=30, run_idx=0),
        make_batch(num_examples=30, run_idx=3),
    ]
    packer._runs.progress[3] = Progress()

    packer.pack()

    assert packer._runs.ready_to_update[0] is True
    assert packer._runs.ready_to_update[3] is True


def test_pack_with_single_sample(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=1, prompt_len=50, completion_len=50)]

    packer.pack()

    packer._sender.send.assert_called_once()
    sent_grid = packer._sender.send.call_args[0][0]
    assert all(len(rank_batches) > 0 for rank_batches in sent_grid)
