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
    temperature: float = 1.0,
) -> TrainingBatch:
    return TrainingBatch(
        examples=[make_sample(prompt_len, completion_len) for _ in range(num_examples)],
        temperature=temperature,
        step=1,
        run_idx=run_idx,
    )


def make_mock_config(batch_size: int = 10) -> MagicMock:
    """Create a mock OrchestratorConfig with batch_size."""
    config = MagicMock()
    config.batch_size = batch_size
    return config


@pytest.fixture
def mock_runs() -> MagicMock:
    runs = MagicMock()
    runs.used_idxs = [0]
    runs.max_runs = 4
    runs.ready_to_update = [False, False, False, False]
    runs.progress = {0: Progress()}
    runs.config = {0: make_mock_config(batch_size=10)}
    runs.output_dir = Path("/tmp/test_output")
    return runs


def _make_packer(mock_runs: MagicMock, tmp_path: Path, small_batch_granularity: bool) -> Packer:
    """Helper to create a Packer with specified granularity setting."""
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
            small_batch_granularity=small_batch_granularity,
        )
        p._runs = mock_runs
        p._receiver = mock_receiver
        p._sender = mock_sender
        return p


@pytest.fixture
def packer(mock_runs: MagicMock, tmp_path: Path) -> Packer:
    """Packer with small_batch_granularity=True for testing small batch behavior."""
    yield _make_packer(mock_runs, tmp_path, small_batch_granularity=True)


@pytest.fixture
def packer_full_batch(mock_runs: MagicMock, tmp_path: Path) -> Packer:
    """Packer with small_batch_granularity=False (default) for testing full batch behavior."""
    yield _make_packer(mock_runs, tmp_path, small_batch_granularity=False)


# =============================================================================
# Tests for Packer.get_batch() - now buffers samples instead of returning dict
# =============================================================================


def test_get_batch_buffers_samples(packer: Packer) -> None:
    batch = make_batch(num_examples=3, run_idx=0)
    packer._receiver.receive.return_value = [batch]

    packer.get_batch()

    assert len(packer.buffers[0]) == 3


def test_get_batch_filters_none_run_idx(packer: Packer) -> None:
    batch_valid = make_batch(num_examples=3, run_idx=0)
    batch_invalid = make_batch(num_examples=2, run_idx=None)
    packer._receiver.receive.return_value = [batch_valid, batch_invalid]

    packer.get_batch()

    assert len(packer.buffers[0]) == 3
    assert len(packer.buffers[None]) == 0  # None run_idx is filtered


def test_get_batch_calls_check_for_changes(packer: Packer) -> None:
    packer._receiver.receive.return_value = []
    packer.get_batch()
    packer._runs.check_for_changes.assert_called_once()


def test_get_batch_multiple_run_idxs(packer: Packer) -> None:
    packer._receiver.receive.return_value = [
        make_batch(num_examples=3, run_idx=0),
        make_batch(num_examples=4, run_idx=1),
        make_batch(num_examples=5, run_idx=2),
    ]

    packer.get_batch()

    assert len(packer.buffers[0]) == 3
    assert len(packer.buffers[1]) == 4
    assert len(packer.buffers[2]) == 5


def test_get_batch_empty_receiver(packer: Packer) -> None:
    packer._receiver.receive.return_value = []

    packer.get_batch()

    assert len(packer.buffers) == 0


def test_get_batch_preserves_temperature(packer: Packer) -> None:
    batch = make_batch(num_examples=2, run_idx=0, temperature=0.7)
    packer._receiver.receive.return_value = [batch]

    packer.get_batch()

    sample, temperature = packer.buffers[0][0]
    assert temperature == 0.7


# =============================================================================
# Tests for Packer.has_enough_tokens() - now checks internal buffers
# =============================================================================


def test_has_enough_tokens_returns_false_when_empty(packer: Packer) -> None:
    assert packer.has_enough_tokens() is False


def test_has_enough_tokens_below_threshold(packer: Packer) -> None:
    # 3 examples * 4 tokens = 12 tokens, threshold = 100 * 2 = 200
    batch = make_batch(num_examples=3)
    packer._receiver.receive.return_value = [batch]
    packer.get_batch()

    assert packer.has_enough_tokens() is False


def test_has_enough_tokens_above_threshold(packer: Packer) -> None:
    # 60 examples * 4 tokens = 240 tokens, threshold = 200
    batch = make_batch(num_examples=60)
    packer._receiver.receive.return_value = [batch]
    packer.get_batch()

    assert packer.has_enough_tokens() is True


def test_has_enough_tokens_threshold_calculation(packer: Packer) -> None:
    # 26 examples * 4 tokens = 104 tokens per batch
    # With estimation logic, this should exceed threshold
    batch = make_batch(num_examples=26)
    packer._receiver.receive.return_value = [batch]
    packer.get_batch()

    assert packer.has_enough_tokens() is True


def test_has_enough_tokens_multiple_runs(packer: Packer) -> None:
    # 2 runs * 10 examples * 4 tokens = 80 tokens, threshold = 200
    packer._receiver.receive.return_value = [
        make_batch(num_examples=10, run_idx=0),
        make_batch(num_examples=10, run_idx=1),
    ]
    packer.get_batch()

    assert packer.has_enough_tokens() is False


# =============================================================================
# Tests for Packer._update_run_progress() - step completion logic
# =============================================================================


def test_update_run_progress_increments_step_at_batch_size(packer: Packer) -> None:
    # batch_size = 10, so 10 samples should trigger step increment
    packer._runs.config[0].batch_size = 10

    packer._update_run_progress(run_idx=0, num_samples=10, num_tokens=40)

    assert packer._runs.progress[0].step == 1
    assert packer._runs.ready_to_update[0] is True


def test_update_run_progress_no_step_below_batch_size(packer: Packer) -> None:
    # batch_size = 10, so 5 samples should not trigger step
    packer._runs.config[0].batch_size = 10

    packer._update_run_progress(run_idx=0, num_samples=5, num_tokens=20)

    assert packer._runs.progress[0].step == 0
    assert packer._runs.ready_to_update[0] is False


def test_update_run_progress_accumulates_samples(packer: Packer) -> None:
    # batch_size = 10, two calls of 5 samples should trigger step
    packer._runs.config[0].batch_size = 10

    packer._update_run_progress(run_idx=0, num_samples=5, num_tokens=20)
    assert packer._runs.progress[0].step == 0

    packer._update_run_progress(run_idx=0, num_samples=5, num_tokens=20)
    assert packer._runs.progress[0].step == 1
    assert packer._runs.ready_to_update[0] is True


def test_update_run_progress_always_updates_totals(packer: Packer) -> None:
    packer._runs.config[0].batch_size = 100  # Large so step won't increment

    packer._update_run_progress(run_idx=0, num_samples=5, num_tokens=20)

    assert packer._runs.progress[0].total_samples == 5
    assert packer._runs.progress[0].total_tokens == 20


def test_update_run_progress_carries_over_excess(packer: Packer) -> None:
    # batch_size = 10, sending 15 samples should increment step and carry over 5
    packer._runs.config[0].batch_size = 10

    packer._update_run_progress(run_idx=0, num_samples=15, num_tokens=60)

    assert packer._runs.progress[0].step == 1
    assert packer.samples_consumed_this_step[0] == 5  # 15 - 10 = 5 carried over


# =============================================================================
# Tests for Packer._select_samples_round_robin()
# =============================================================================


def test_select_samples_round_robin_single_run(packer: Packer) -> None:
    # Add 5 samples to run 0
    for _ in range(5):
        packer.buffers[0].append((make_sample(), 1.0))

    selected = packer._select_samples_round_robin(token_budget=20)  # 5 samples * 4 tokens

    assert len(selected) == 5
    assert all(run_idx == 0 for run_idx, _, _ in selected)


def test_select_samples_round_robin_multiple_runs(packer: Packer) -> None:
    # Add samples to two runs
    packer._runs.used_idxs = [0, 1]
    for _ in range(4):
        packer.buffers[0].append((make_sample(), 1.0))
        packer.buffers[1].append((make_sample(), 1.0))

    # Budget for 6 samples (24 tokens), should take 3 from each run
    selected = packer._select_samples_round_robin(token_budget=24)

    run_counts = {0: 0, 1: 0}
    for run_idx, _, _ in selected:
        run_counts[run_idx] += 1

    assert run_counts[0] == 3
    assert run_counts[1] == 3


def test_select_samples_round_robin_skips_empty_buffer(packer: Packer) -> None:
    # Run 0 is empty, run 1 has samples
    packer._runs.used_idxs = [0, 1]
    for _ in range(4):
        packer.buffers[1].append((make_sample(), 1.0))

    selected = packer._select_samples_round_robin(token_budget=16)

    assert len(selected) == 4
    assert all(run_idx == 1 for run_idx, _, _ in selected)


def test_select_samples_round_robin_respects_token_budget(packer: Packer) -> None:
    # Add many samples
    for _ in range(100):
        packer.buffers[0].append((make_sample(), 1.0))

    # Budget for 10 tokens (2-3 samples at 4 tokens each)
    selected = packer._select_samples_round_robin(token_budget=10)

    total_tokens = sum(len(sample.prompt_ids) + len(sample.completion_ids) for _, sample, _ in selected)
    assert total_tokens >= 10  # At least budget
    assert total_tokens <= 14  # No more than one extra sample


# =============================================================================
# Tests for Packer.pack()
# =============================================================================


def test_pack_updates_progress_step_when_batch_size_reached(packer: Packer) -> None:
    # batch_size = 10, sending 60 samples (240 tokens) exceeds threshold (200)
    # Token budget = 200, samples have 4 tokens each, so ~50 samples selected
    packer._runs.config[0].batch_size = 10
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]

    packer.pack()

    # With token budget of 200 and 4 tokens/sample, we select ~50 samples
    # batch_size = 10, so step should be 5
    assert packer._runs.progress[0].step == 5


def test_pack_no_step_increment_below_batch_size(packer: Packer) -> None:
    # batch_size = 100, sending 60 samples should not increment step
    packer._runs.config[0].batch_size = 100
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]

    packer.pack()

    assert packer._runs.progress[0].step == 0
    assert packer._runs.ready_to_update[0] is False


def test_pack_updates_progress_total_tokens(packer: Packer) -> None:
    # 60 examples * 4 tokens = 240 tokens, exceeds threshold (200)
    # Token budget = 200, so ~50 samples (200 tokens) selected
    packer._runs.config[0].batch_size = 100  # High so step doesn't increment
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]
    packer._runs.progress[0].total_tokens = 0

    packer.pack()

    # With token budget of 200 and 4 tokens/sample, we select ~50 samples = 200 tokens
    assert packer._runs.progress[0].total_tokens == 200


def test_pack_updates_progress_total_samples(packer: Packer) -> None:
    # 60 examples * 4 tokens = 240 tokens, exceeds threshold (200)
    packer._runs.config[0].batch_size = 100  # High so step doesn't increment
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]
    packer._runs.progress[0].total_samples = 0

    packer.pack()

    # With token budget of 200 and 4 tokens/sample, we select ~50 samples
    assert packer._runs.progress[0].total_samples == 50


def test_pack_sets_ready_to_update_flag_on_step_completion(packer: Packer) -> None:
    packer._runs.config[0].batch_size = 10
    packer._receiver.receive.return_value = [make_batch(num_examples=60, run_idx=0)]
    packer._runs.ready_to_update = [False, False, False, False]

    packer.pack()

    assert packer._runs.ready_to_update[0] is True


def test_pack_calls_sender_with_micro_batch_grid(packer: Packer) -> None:
    packer._runs.config[0].batch_size = 10
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]

    packer.pack()

    packer._sender.send.assert_called_once()
    sent_grid = packer._sender.send.call_args[0][0]
    assert isinstance(sent_grid, list)
    assert len(sent_grid) == 2  # dp_world_size


def test_pack_handles_multiple_runs(packer: Packer) -> None:
    # 60 samples total = 240 tokens, exceeds threshold (200)
    # With round-robin, we get ~25 samples from each run
    packer._runs.used_idxs = [0, 1]
    packer._runs.config[0].batch_size = 25
    packer._runs.config[1] = make_mock_config(batch_size=25)
    packer._runs.progress[1] = Progress()
    packer._receiver.receive.return_value = [
        make_batch(num_examples=30, run_idx=0),
        make_batch(num_examples=30, run_idx=1),
    ]

    packer.pack()

    # Each run gets ~25 samples, batch_size = 25, so step = 1 for each
    assert packer._runs.progress[0].step == 1
    assert packer._runs.progress[1].step == 1


def test_pack_sends_grid_with_equal_batch_counts(packer: Packer) -> None:
    packer._runs.config[0].batch_size = 10
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]

    packer.pack()

    sent_grid = packer._sender.send.call_args[0][0]
    batch_counts = [len(rank_batches) for rank_batches in sent_grid]
    assert len(set(batch_counts)) == 1, f"Unequal batch counts: {batch_counts}"


def test_pack_sends_non_empty_batches_per_rank(packer: Packer) -> None:
    packer._runs.config[0].batch_size = 10
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]

    packer.pack()

    sent_grid = packer._sender.send.call_args[0][0]
    assert all(len(rank_batches) > 0 for rank_batches in sent_grid)


def test_pack_lora_num_tokens_invariant(packer: Packer) -> None:
    packer._runs.config[0].batch_size = 10
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
    packer._runs.config[0].batch_size = 3

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
    # 60 samples total = 240 tokens, exceeds threshold (200)
    # Round-robin gives ~25 samples to each run
    packer._runs.used_idxs = [0, 3]
    packer._runs.config[0].batch_size = 25
    packer._runs.config[3] = make_mock_config(batch_size=25)
    packer._runs.progress[3] = Progress()
    packer._receiver.receive.return_value = [
        make_batch(num_examples=30, run_idx=0),
        make_batch(num_examples=30, run_idx=3),
    ]

    packer.pack()

    # Each run gets ~25 samples, batch_size = 25, so ready_to_update = True
    assert packer._runs.ready_to_update[0] is True
    assert packer._runs.ready_to_update[3] is True


def test_pack_with_single_sample(packer: Packer) -> None:
    packer._runs.config[0].batch_size = 1
    packer._receiver.receive.return_value = [make_batch(num_examples=1, prompt_len=50, completion_len=50)]

    packer.pack()

    packer._sender.send.assert_called_once()
    sent_grid = packer._sender.send.call_args[0][0]
    assert all(len(rank_batches) > 0 for rank_batches in sent_grid)


def test_pack_with_timeout_sends_available_samples(packer: Packer) -> None:
    """When timeout occurs with insufficient tokens, pack what's available."""
    # Only 3 samples = 12 tokens, below threshold (200)
    packer._runs.config[0].batch_size = 3
    packer._receiver.receive.return_value = [make_batch(num_examples=3)]

    with patch("prime_rl.trainer.rl.packer.time") as mock_time:
        mock_time.time.side_effect = [0.0, 0.0, 100.0]  # First check, second check triggers timeout
        mock_time.sleep = MagicMock()

        packer.pack()

    # Should still pack and send the available samples after timeout
    packer._sender.send.assert_called_once()


# =============================================================================
# Tests for buffering behavior
# =============================================================================


def test_buffer_accumulates_across_get_batch_calls(packer: Packer) -> None:
    packer._receiver.receive.return_value = [make_batch(num_examples=3, run_idx=0)]

    packer.get_batch()
    packer.get_batch()

    assert len(packer.buffers[0]) == 6


def test_partial_step_persists_across_pack_calls(packer: Packer) -> None:
    # Token budget = 200, batch_size = 100
    # 60 samples = 240 tokens, ~50 samples selected
    packer._runs.config[0].batch_size = 100

    # First pack: 60 samples, ~50 selected, not enough for step
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]
    packer.pack()

    assert packer._runs.progress[0].step == 0
    assert packer.samples_consumed_this_step[0] == 50  # Token budget limits selection

    # Second pack: 60 more samples in buffer (10 leftover + 60 = 70)
    # After first pack, buffer has ~10 leftover, add 60 more = 70 samples
    # ~50 more selected, total consumed = 50 + 50 = 100, step increments
    packer._receiver.receive.return_value = [make_batch(num_examples=60)]
    packer.pack()

    assert packer._runs.progress[0].step == 1
    assert packer.samples_consumed_this_step[0] == 0  # 100 - 100 = 0


def test_round_robin_position_persists_across_pack_calls(packer: Packer) -> None:
    packer._runs.used_idxs = [0, 1]
    packer._runs.config[0].batch_size = 100
    packer._runs.config[1] = make_mock_config(batch_size=100)
    packer._runs.progress[1] = Progress()

    # Add samples to both runs
    for _ in range(30):
        packer.buffers[0].append((make_sample(), 1.0))
        packer.buffers[1].append((make_sample(), 1.0))

    # Pack should use round-robin
    packer.pack()

    # Both runs should have had samples taken
    assert packer._runs.progress[0].total_samples > 0
    assert packer._runs.progress[1].total_samples > 0


# =============================================================================
# Tests for full batch behavior (small_batch_granularity=False, the default)
# =============================================================================


def test_full_batch_has_enough_tokens_requires_batch_size(packer_full_batch: Packer) -> None:
    """When small_batch_granularity=False, has_enough_tokens requires batch_size samples."""
    packer = packer_full_batch
    packer._runs.config[0].batch_size = 10

    # Add 9 samples with enough tokens to meet threshold (200 tokens needed)
    # 9 samples * 25 tokens = 225 tokens (above threshold)
    for _ in range(9):
        packer.buffers[0].append((make_sample(prompt_len=10, completion_len=15), 1.0))

    # Even with enough tokens for threshold, should return False without batch_size samples
    assert not packer.has_enough_tokens()

    # Add one more to reach batch_size
    packer.buffers[0].append((make_sample(prompt_len=10, completion_len=15), 1.0))
    assert packer.has_enough_tokens()


def test_full_batch_select_only_from_full_batches(packer_full_batch: Packer) -> None:
    """When small_batch_granularity=False, only select from runs with batch_size samples."""
    packer = packer_full_batch
    packer._runs.used_idxs = [0, 1]
    packer._runs.config[0].batch_size = 10
    packer._runs.config[1] = make_mock_config(batch_size=10)
    packer._runs.progress[1] = Progress()

    # Run 0: 5 samples (below batch_size)
    for _ in range(5):
        packer.buffers[0].append((make_sample(), 1.0))

    # Run 1: 15 samples (above batch_size)
    for _ in range(15):
        packer.buffers[1].append((make_sample(), 1.0))

    # Select should only take from run 1
    selected = packer._select_samples_round_robin(token_budget=200)

    # All selected should be from run 1
    run_idxs = [run_idx for run_idx, _, _ in selected]
    assert all(idx == 1 for idx in run_idxs)
    assert len(selected) > 0


def test_full_batch_pack_waits_for_batch_size(packer_full_batch: Packer) -> None:
    """When small_batch_granularity=False, pack waits for batch_size samples."""
    packer = packer_full_batch
    packer._runs.config[0].batch_size = 10

    # Add 5 samples (below batch_size)
    for _ in range(5):
        packer.buffers[0].append((make_sample(), 1.0))

    # Pack should not send (timeout with no full batch)
    packer.pack()

    # No samples should be consumed (waiting for full batch)
    assert packer._runs.progress[0].total_samples == 0
    packer._sender.send.assert_not_called()


def test_full_batch_pack_processes_full_batch(packer_full_batch: Packer) -> None:
    """When small_batch_granularity=False, pack processes runs with batch_size samples."""
    packer = packer_full_batch
    packer._runs.config[0].batch_size = 10

    # Add 15 samples (above batch_size)
    for _ in range(15):
        packer.buffers[0].append((make_sample(), 1.0))

    packer.pack()

    # Should have processed samples
    assert packer._runs.progress[0].total_samples > 0
    packer._sender.send.assert_called_once()


def test_full_batch_multiple_runs_only_full_processed(packer_full_batch: Packer) -> None:
    """With multiple runs, only those with batch_size samples are processed."""
    packer = packer_full_batch
    packer._runs.used_idxs = [0, 1]
    packer._runs.config[0].batch_size = 20
    packer._runs.config[1] = make_mock_config(batch_size=10)
    packer._runs.progress[1] = Progress()

    # Run 0: 15 samples (below batch_size of 20)
    for _ in range(15):
        packer.buffers[0].append((make_sample(), 1.0))

    # Run 1: 15 samples (above batch_size of 10)
    for _ in range(15):
        packer.buffers[1].append((make_sample(), 1.0))

    packer.pack()

    # Only run 1 should have been processed
    assert packer._runs.progress[0].total_samples == 0
    assert packer._runs.progress[1].total_samples > 0


def test_full_batch_get_runs_with_full_batch(packer_full_batch: Packer) -> None:
    """Test _get_runs_with_full_batch helper method."""
    packer = packer_full_batch
    packer._runs.used_idxs = [0, 1, 2]
    packer._runs.config[0].batch_size = 10
    packer._runs.config[1] = make_mock_config(batch_size=5)
    packer._runs.config[2] = make_mock_config(batch_size=20)
    packer._runs.progress[1] = Progress()
    packer._runs.progress[2] = Progress()

    # Run 0: 10 samples (exactly batch_size)
    for _ in range(10):
        packer.buffers[0].append((make_sample(), 1.0))

    # Run 1: 3 samples (below batch_size)
    for _ in range(3):
        packer.buffers[1].append((make_sample(), 1.0))

    # Run 2: 25 samples (above batch_size)
    for _ in range(25):
        packer.buffers[2].append((make_sample(), 1.0))

    runs_with_full = packer._get_runs_with_full_batch()

    assert 0 in runs_with_full  # Exactly batch_size
    assert 1 not in runs_with_full  # Below batch_size
    assert 2 in runs_with_full  # Above batch_size


def test_full_batch_step_always_completes(packer_full_batch: Packer) -> None:
    """When small_batch_granularity=False, each pack should complete at least one step."""
    packer = packer_full_batch
    # batch_size=8 so that 8 samples (200 tokens) fit within token budget (200)
    packer._runs.config[0].batch_size = 8

    # Add batch_size samples with enough tokens to meet budget (200)
    # 8 samples * 25 tokens = 200 tokens
    for _ in range(8):
        packer.buffers[0].append((make_sample(prompt_len=10, completion_len=15), 1.0))

    packer.pack()

    # Step should have incremented (8 samples consumed, batch_size=8)
    assert packer._runs.progress[0].step == 1
    assert packer._runs.ready_to_update[0] is True


# =============================================================================
# Tests for run creation hook - resetting state when run is replaced
# =============================================================================


def test_on_run_created_clears_buffer(packer: Packer) -> None:
    """_on_run_created should clear buffered samples for the run index."""
    # Add samples to buffer
    for _ in range(5):
        packer.buffers[0].append((make_sample(), 1.0))
    packer.buffers[1].append((make_sample(), 1.0))

    # Simulate run 0 being replaced
    packer._on_run_created(0, "run_new")

    # Buffer for run 0 should be empty
    assert len(packer.buffers[0]) == 0
    # Buffer for run 1 should be unchanged
    assert len(packer.buffers[1]) == 1


def test_on_run_created_clears_samples_consumed(packer: Packer) -> None:
    """_on_run_created should clear partial step progress for the run index."""
    packer.samples_consumed_this_step[0] = 5
    packer.samples_consumed_this_step[1] = 3

    packer._on_run_created(0, "run_new")

    # Run 0 should be cleared
    assert 0 not in packer.samples_consumed_this_step
    # Run 1 should be unchanged
    assert packer.samples_consumed_this_step[1] == 3


def test_on_run_created_calls_receiver_reset(packer: Packer) -> None:
    """_on_run_created should call receiver.reset_run()."""
    packer._on_run_created(0, "run_new")

    packer._receiver.reset_run.assert_called_once_with(0)


def test_creation_hook_registered(mock_runs: MagicMock, tmp_path: Path) -> None:
    """Packer should register its creation hook with Runs."""
    mock_runs.output_dir = tmp_path
    mock_receiver = MagicMock()
    mock_receiver.receive.return_value = []
    mock_sender = MagicMock()

    with (
        patch("prime_rl.trainer.rl.packer.get_logger", return_value=MagicMock()),
        patch("prime_rl.trainer.rl.packer.get_runs", return_value=mock_runs),
        patch("prime_rl.trainer.rl.packer.setup_training_batch_receiver", return_value=mock_receiver),
        patch("prime_rl.trainer.rl.packer.setup_micro_batch_sender", return_value=mock_sender),
    ):
        packer = Packer(
            dp_world_size=2,
            seq_len=100,
            pad_to_multiple_of=8,
            tokenizer=MagicMock(pad_token_id=0),
            config=MagicMock(),
            start_step=0,
            small_batch_granularity=True,
        )

        # Verify the creation hook was registered
        mock_runs.register_creation_hook.assert_called_once_with(packer._on_run_created)
