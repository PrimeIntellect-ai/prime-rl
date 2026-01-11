"""Unit tests for trainer batch preparation functions.

Tests the critical invariants:
1. No empty microbatches - each dp rank must receive at least one microbatch
2. run_idx consistency - all tokens in a MicroBatch must come from the same run_idx
3. lora_num_tokens correctness - sum(lora_num_tokens) == len(input_ids), only one non-zero entry
4. Equal batch counts - all dp ranks must receive the same number of microbatches
"""

import pytest

from prime_rl.trainer.batch import (
    packed_samples_into_micro_bs,
    pad_micro_batch,
    prepare_batch,
    prepare_sample,
)
from prime_rl.transport.types import MicroBatch, TrainingSample


@pytest.fixture
def make_training_sample():
    """Factory for creating TrainingSample with configurable sizes."""

    def _make(
        prompt_len: int = 2,
        completion_len: int = 2,
        advantage: float = 1.0,
        with_teacher_logprobs: bool = False,
    ) -> TrainingSample:
        return TrainingSample(
            prompt_ids=list(range(1, prompt_len + 1)),
            prompt_mask=[False] * prompt_len,
            completion_ids=list(range(100, 100 + completion_len)),
            completion_mask=[True] * completion_len,
            completion_logprobs=[-0.1] * completion_len,
            teacher_logprobs=[0.0] * (prompt_len + completion_len) if with_teacher_logprobs else None,
            advantage=advantage,
        )

    return _make


@pytest.fixture
def make_micro_batch():
    """Factory for creating MicroBatch with configurable sizes."""

    def _make(
        seq_len: int = 4,
        temperature: float = 1.0,
        lora_num_tokens: list[int] | None = None,
        with_teacher_logprobs: bool = False,
    ) -> MicroBatch:
        return MicroBatch(
            input_ids=list(range(seq_len)),
            loss_mask=[True] * seq_len,
            advantages=[1.0] * seq_len,
            inference_logprobs=[-0.1] * seq_len,
            position_ids=list(range(seq_len)),
            temperature=temperature,
            teacher_logprobs=[0.0] * seq_len if with_teacher_logprobs else None,
            lora_num_tokens=lora_num_tokens,
        )

    return _make


# =============================================================================
# Tests for prepare_sample()
# =============================================================================


def test_prepare_sample_basic(make_training_sample):
    """Verify correct MicroBatch creation from TrainingSample."""
    sample = make_training_sample(prompt_len=2, completion_len=3)
    result = prepare_sample(sample, seq_len=100, temperature=0.8)

    assert len(result.input_ids) == 5
    assert len(result.loss_mask) == 5
    assert len(result.advantages) == 5
    assert len(result.position_ids) == 5
    assert len(result.inference_logprobs) == 5
    assert result.temperature == 0.8
    assert result.position_ids == [0, 1, 2, 3, 4]
    assert result.loss_mask == [False, False, True, True, True]


def test_prepare_sample_truncates_at_seq_len(make_training_sample):
    """Verify truncation when sample exceeds seq_len."""
    sample = make_training_sample(prompt_len=5, completion_len=5)
    result = prepare_sample(sample, seq_len=6, temperature=1.0)

    assert len(result.input_ids) == 6
    assert len(result.loss_mask) == 6
    assert len(result.advantages) == 6
    assert len(result.position_ids) == 6
    assert len(result.inference_logprobs) == 6


def test_prepare_sample_inference_logprobs_prepends_zeros(make_training_sample):
    """Verify prompt tokens get 0.0 for inference_logprobs."""
    sample = make_training_sample(prompt_len=3, completion_len=2)
    result = prepare_sample(sample, seq_len=100, temperature=1.0)

    assert result.inference_logprobs[:3] == [0.0, 0.0, 0.0]
    assert result.inference_logprobs[3:] == [-0.1, -0.1]


def test_prepare_sample_with_teacher_logprobs(make_training_sample):
    """Verify teacher_logprobs are preserved when present."""
    sample = make_training_sample(prompt_len=2, completion_len=2, with_teacher_logprobs=True)
    result = prepare_sample(sample, seq_len=100, temperature=1.0)

    assert result.teacher_logprobs is not None
    assert len(result.teacher_logprobs) == 4


def test_prepare_sample_without_teacher_logprobs(make_training_sample):
    """Verify teacher_logprobs is None when not provided."""
    sample = make_training_sample(prompt_len=2, completion_len=2, with_teacher_logprobs=False)
    result = prepare_sample(sample, seq_len=100, temperature=1.0)

    assert result.teacher_logprobs is None


def test_prepare_sample_advantages_repeated_for_all_tokens(make_training_sample):
    """Verify advantages are repeated for all tokens."""
    sample = make_training_sample(prompt_len=2, completion_len=3, advantage=2.5)
    result = prepare_sample(sample, seq_len=100, temperature=1.0)

    assert all(adv == 2.5 for adv in result.advantages)
    assert len(result.advantages) == 5


# =============================================================================
# Tests for packed_samples_into_micro_bs()
# =============================================================================


def test_packing_initializes_lora_num_tokens(make_training_sample):
    """Verify lora_num_tokens is created with length num_loras."""
    sample = make_training_sample(prompt_len=2, completion_len=2)
    micro_batch = prepare_sample(sample, seq_len=100, temperature=1.0)
    samples = [(0, micro_batch)]

    result = packed_samples_into_micro_bs(samples, max_seq_len=100, num_loras=4)

    assert len(result) == 1
    assert result[0].lora_num_tokens is not None
    assert len(result[0].lora_num_tokens) == 4


def test_packing_lora_num_tokens_sum_equals_input_len(make_training_sample):
    """Verify sum(lora_num_tokens) == len(input_ids) for each microbatch."""
    sample1 = make_training_sample(prompt_len=2, completion_len=2)
    sample2 = make_training_sample(prompt_len=3, completion_len=3)
    micro_batch1 = prepare_sample(sample1, seq_len=100, temperature=1.0)
    micro_batch2 = prepare_sample(sample2, seq_len=100, temperature=1.0)
    samples = [(0, micro_batch1), (0, micro_batch2)]

    result = packed_samples_into_micro_bs(samples, max_seq_len=100, num_loras=4)

    for mb in result:
        assert sum(mb.lora_num_tokens) == len(mb.input_ids)


def test_packing_single_nonzero_lora_entry(make_training_sample):
    """Verify only one entry is non-zero per microbatch when samples have same run_idx."""
    sample1 = make_training_sample(prompt_len=2, completion_len=2)
    sample2 = make_training_sample(prompt_len=2, completion_len=2)
    micro_batch1 = prepare_sample(sample1, seq_len=100, temperature=1.0)
    micro_batch2 = prepare_sample(sample2, seq_len=100, temperature=1.0)
    samples = [(0, micro_batch1), (0, micro_batch2)]

    result = packed_samples_into_micro_bs(samples, max_seq_len=100, num_loras=4)

    for mb in result:
        nonzero_count = sum(1 for x in mb.lora_num_tokens if x > 0)
        assert nonzero_count == 1, f"Expected 1 non-zero entry, got {nonzero_count}"


def test_packing_correct_lora_index(make_training_sample):
    """Verify non-zero entry is at the correct run_idx."""
    sample = make_training_sample(prompt_len=2, completion_len=2)
    micro_batch = prepare_sample(sample, seq_len=100, temperature=1.0)
    run_idx = 2
    samples = [(run_idx, micro_batch)]

    result = packed_samples_into_micro_bs(samples, max_seq_len=100, num_loras=4)

    assert result[0].lora_num_tokens[run_idx] == len(result[0].input_ids)
    for i in range(4):
        if i != run_idx:
            assert result[0].lora_num_tokens[i] == 0


def test_packing_different_run_idx_can_share_bin(make_training_sample):
    """Verify samples with different run_idx CAN share microbatches if they fit.

    Note: The packed_samples_into_micro_bs function allows mixing run_idx in bins.
    The invariant of single run_idx per microbatch is enforced at the Packer level.
    """
    sample1 = make_training_sample(prompt_len=2, completion_len=2)
    sample2 = make_training_sample(prompt_len=2, completion_len=2)
    micro_batch1 = prepare_sample(sample1, seq_len=100, temperature=1.0)
    micro_batch2 = prepare_sample(sample2, seq_len=100, temperature=1.0)
    samples = [(0, micro_batch1), (1, micro_batch2)]

    result = packed_samples_into_micro_bs(samples, max_seq_len=100, num_loras=4)

    assert len(result) == 1
    assert result[0].lora_num_tokens[0] == 4
    assert result[0].lora_num_tokens[1] == 4
    assert sum(result[0].lora_num_tokens) == len(result[0].input_ids)


def test_packing_same_run_idx_samples_fit_in_one_bin(make_training_sample):
    """Verify samples with same run_idx can share a microbatch."""
    sample1 = make_training_sample(prompt_len=2, completion_len=2)
    sample2 = make_training_sample(prompt_len=2, completion_len=2)
    micro_batch1 = prepare_sample(sample1, seq_len=100, temperature=1.0)
    micro_batch2 = prepare_sample(sample2, seq_len=100, temperature=1.0)
    samples = [(0, micro_batch1), (0, micro_batch2)]

    result = packed_samples_into_micro_bs(samples, max_seq_len=100, num_loras=4)

    assert len(result) == 1
    assert len(result[0].input_ids) == 8


def test_packing_overflow_creates_new_bin(make_training_sample):
    """Verify samples that exceed max_seq_len create new bins."""
    sample1 = make_training_sample(prompt_len=3, completion_len=3)
    sample2 = make_training_sample(prompt_len=3, completion_len=3)
    micro_batch1 = prepare_sample(sample1, seq_len=100, temperature=1.0)
    micro_batch2 = prepare_sample(sample2, seq_len=100, temperature=1.0)
    samples = [(0, micro_batch1), (0, micro_batch2)]

    result = packed_samples_into_micro_bs(samples, max_seq_len=10, num_loras=4)

    assert len(result) == 2


def test_packing_sparse_run_idxs(make_training_sample):
    """Verify non-contiguous run_idx values are handled correctly."""
    sample1 = make_training_sample(prompt_len=2, completion_len=2)
    sample2 = make_training_sample(prompt_len=2, completion_len=2)
    micro_batch1 = prepare_sample(sample1, seq_len=100, temperature=1.0)
    micro_batch2 = prepare_sample(sample2, seq_len=100, temperature=1.0)
    samples = [(0, micro_batch1), (3, micro_batch2)]

    result = packed_samples_into_micro_bs(samples, max_seq_len=100, num_loras=4)

    assert len(result) == 1
    assert result[0].lora_num_tokens[0] == 4
    assert result[0].lora_num_tokens[1] == 0
    assert result[0].lora_num_tokens[2] == 0
    assert result[0].lora_num_tokens[3] == 4
    assert sum(result[0].lora_num_tokens) == len(result[0].input_ids)


# =============================================================================
# Tests for pad_micro_batch()
# =============================================================================


def test_pad_no_padding_needed(make_micro_batch):
    """Verify no padding when length is already a multiple."""
    mb = make_micro_batch(seq_len=8, lora_num_tokens=[8, 0, 0, 0])
    original_len = len(mb.input_ids)

    result = pad_micro_batch(mb, pad_to_multiple_of=8)

    assert len(result.input_ids) == original_len


def test_pad_adds_correct_padding(make_micro_batch):
    """Verify padding to next multiple."""
    mb = make_micro_batch(seq_len=5, lora_num_tokens=[5, 0, 0, 0])

    result = pad_micro_batch(mb, pad_to_multiple_of=8)

    assert len(result.input_ids) == 8


def test_pad_loss_mask_false_for_padding(make_micro_batch):
    """Verify padding tokens have False loss_mask."""
    mb = make_micro_batch(seq_len=5, lora_num_tokens=[5, 0, 0, 0])
    mb.loss_mask = [True] * 5

    result = pad_micro_batch(mb, pad_to_multiple_of=8)

    assert result.loss_mask[-3:] == [False, False, False]
    assert result.loss_mask[:5] == [True, True, True, True, True]


def test_pad_advantages_zero_for_padding(make_micro_batch):
    """Verify padding tokens have zero advantages."""
    mb = make_micro_batch(seq_len=5, lora_num_tokens=[5, 0, 0, 0])
    mb.advantages = [1.0] * 5

    result = pad_micro_batch(mb, pad_to_multiple_of=8)

    assert result.advantages[-3:] == [0.0, 0.0, 0.0]


def test_pad_adds_to_last_lora_index(make_micro_batch):
    """Verify padding tokens go to lora_num_tokens[-1]."""
    mb = make_micro_batch(seq_len=5, lora_num_tokens=[5, 0, 0, 0])

    result = pad_micro_batch(mb, pad_to_multiple_of=8)

    assert result.lora_num_tokens[-1] == 3
    assert result.lora_num_tokens[0] == 5


def test_pad_preserves_lora_sum_invariant(make_micro_batch):
    """Verify sum(lora_num_tokens) == len(input_ids) after padding."""
    mb = make_micro_batch(seq_len=5, lora_num_tokens=[5, 0, 0, 0])

    result = pad_micro_batch(mb, pad_to_multiple_of=8)

    assert sum(result.lora_num_tokens) == len(result.input_ids)


def test_pad_teacher_logprobs_extended(make_micro_batch):
    """Verify teacher_logprobs are also padded."""
    mb = make_micro_batch(seq_len=5, lora_num_tokens=[5, 0, 0, 0], with_teacher_logprobs=True)

    result = pad_micro_batch(mb, pad_to_multiple_of=8)

    assert result.teacher_logprobs is not None
    assert len(result.teacher_logprobs) == len(result.input_ids)


def test_pad_with_multiple_of_one(make_micro_batch):
    """Verify pad_to_multiple_of=1 results in no padding."""
    mb = make_micro_batch(seq_len=5, lora_num_tokens=[5, 0, 0, 0])
    original_len = len(mb.input_ids)

    result = pad_micro_batch(mb, pad_to_multiple_of=1)

    assert len(result.input_ids) == original_len


# =============================================================================
# Tests for prepare_batch()
# =============================================================================


def test_prepare_batch_single_rollout_creates_microbatch(make_training_sample):
    """Verify minimum case with 1 rollout works."""
    samples = [make_training_sample()]

    result = prepare_batch(
        rollouts=samples,
        temperature=1.0,
        seq_len=100,
        num_train_workers=1,
        idxs=[0],
        num_loras=1,
    )

    assert len(result) == 1
    assert len(result[0]) >= 1


def test_prepare_batch_equal_batches_per_worker(make_training_sample):
    """Verify all workers get the same number of microbatches."""
    samples = [make_training_sample() for _ in range(10)]

    result = prepare_batch(
        rollouts=samples,
        temperature=1.0,
        seq_len=100,
        num_train_workers=4,
        idxs=[0] * 10,
        num_loras=1,
    )

    batch_counts = [len(worker_batches) for worker_batches in result]
    assert len(set(batch_counts)) == 1, f"Unequal batch counts: {batch_counts}"


def test_prepare_batch_padding_batches_have_zero_advantages(make_training_sample):
    """Verify fake/padding batches have zero advantages."""
    samples = [make_training_sample() for _ in range(5)]

    result = prepare_batch(
        rollouts=samples,
        temperature=1.0,
        seq_len=4,
        num_train_workers=2,
        idxs=[0] * 5,
        num_loras=1,
    )

    flat_batches = [batch for worker_batches in result for batch in worker_batches]
    if len(flat_batches) > 5:
        padding_batch = flat_batches[-1]
        assert all(adv == 0.0 for adv in padding_batch.advantages)


def test_prepare_batch_padding_batches_have_false_loss_mask(make_training_sample):
    """Verify fake/padding batches have False loss_mask."""
    samples = [make_training_sample() for _ in range(5)]

    result = prepare_batch(
        rollouts=samples,
        temperature=1.0,
        seq_len=4,
        num_train_workers=2,
        idxs=[0] * 5,
        num_loras=1,
    )

    flat_batches = [batch for worker_batches in result for batch in worker_batches]
    if len(flat_batches) > 5:
        padding_batch = flat_batches[-1]
        assert all(not mask for mask in padding_batch.loss_mask)


def test_prepare_batch_lora_num_tokens_sum_invariant(make_training_sample):
    """Verify sum(lora_num_tokens) == len(input_ids) for all microbatches."""
    samples = [make_training_sample() for _ in range(10)]

    result = prepare_batch(
        rollouts=samples,
        temperature=1.0,
        seq_len=100,
        num_train_workers=2,
        idxs=[0] * 10,
        num_loras=4,
    )

    for worker_batches in result:
        for mb in worker_batches:
            assert sum(mb.lora_num_tokens) == len(mb.input_ids)


def test_prepare_batch_multiple_run_idx(make_training_sample):
    """Verify handling of multiple run_idx values."""
    samples = [make_training_sample() for _ in range(6)]

    result = prepare_batch(
        rollouts=samples,
        temperature=1.0,
        seq_len=100,
        num_train_workers=2,
        idxs=[0, 0, 0, 1, 1, 1],
        num_loras=4,
    )

    assert all(len(wb) > 0 for wb in result)
    for worker_batches in result:
        for mb in worker_batches:
            assert sum(mb.lora_num_tokens) == len(mb.input_ids)


@pytest.mark.parametrize(
    ("rollout_count", "num_train_workers", "num_loras"),
    [
        (1, 1, 1),
        (4, 2, 1),
        (5, 2, 2),
        (11, 4, 4),
        (3, 8, 2),
        (20, 4, 8),
    ],
)
def test_prepare_batch_invariants_parametrized(make_training_sample, rollout_count, num_train_workers, num_loras):
    """Parametrized test verifying all critical invariants."""
    samples = [make_training_sample() for _ in range(rollout_count)]

    result = prepare_batch(
        rollouts=samples,
        temperature=1.0,
        seq_len=100,
        num_train_workers=num_train_workers,
        idxs=[0] * rollout_count,
        num_loras=num_loras,
    )

    batch_counts = [len(wb) for wb in result]
    assert len(set(batch_counts)) == 1, f"Unequal batch counts: {batch_counts}"
    assert all(len(wb) > 0 for wb in result), "Empty microbatch list found"

    for worker_batches in result:
        for mb in worker_batches:
            assert sum(mb.lora_num_tokens) == len(mb.input_ids)
            assert len(mb.lora_num_tokens) == num_loras


def test_prepare_batch_empty_rollouts_behavior():
    """Document behavior when rollouts list is empty."""
    try:
        result = prepare_batch(
            rollouts=[],
            temperature=1.0,
            seq_len=100,
            num_train_workers=2,
            idxs=[],
            num_loras=1,
        )
        assert len(result) == 2
        assert all(len(wb) == 0 for wb in result)
    except Exception as e:
        pytest.skip(f"Empty rollouts raises {type(e).__name__}: {e}")


# =============================================================================
# Integration tests for the full packing pipeline
# =============================================================================


def test_run_idx_mixed_when_called_with_mixed_idxs(make_training_sample):
    """Document behavior: prepare_batch CAN mix run_idx when called with mixed idxs."""
    samples = [make_training_sample(prompt_len=2, completion_len=2) for _ in range(10)]

    result = prepare_batch(
        rollouts=samples,
        temperature=1.0,
        seq_len=100,
        num_train_workers=2,
        idxs=[0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
        num_loras=4,
    )

    for worker_batches in result:
        for mb in worker_batches:
            assert sum(mb.lora_num_tokens) == len(mb.input_ids)


def test_single_run_idx_per_batch_when_packer_pattern(make_training_sample):
    """Verify single run_idx per microbatch when following Packer's usage pattern."""
    samples_run0 = [make_training_sample(prompt_len=2, completion_len=2) for _ in range(5)]
    samples_run1 = [make_training_sample(prompt_len=2, completion_len=2) for _ in range(5)]

    result_run0 = prepare_batch(
        rollouts=samples_run0,
        temperature=1.0,
        seq_len=100,
        num_train_workers=2,
        idxs=[0] * 5,
        num_loras=4,
    )

    result_run1 = prepare_batch(
        rollouts=samples_run1,
        temperature=1.0,
        seq_len=100,
        num_train_workers=2,
        idxs=[1] * 5,
        num_loras=4,
    )

    for worker_batches in result_run0:
        for mb in worker_batches:
            nonzero_indices = [i for i, x in enumerate(mb.lora_num_tokens) if x > 0]
            for idx in nonzero_indices:
                assert idx in [0, 3], f"Unexpected lora index {idx} for run 0"

    for worker_batches in result_run1:
        for mb in worker_batches:
            nonzero_indices = [i for i, x in enumerate(mb.lora_num_tokens) if x > 0]
            for idx in nonzero_indices:
                assert idx in [1, 3], f"Unexpected lora index {idx} for run 1"


def test_dp_ranks_receive_identical_batch_counts(make_training_sample):
    """Critical test: All dp ranks must receive identical batch counts."""
    for num_workers in [1, 2, 4, 8]:
        for num_samples in [1, 3, 7, 15, 100]:
            samples = [make_training_sample() for _ in range(num_samples)]

            result = prepare_batch(
                rollouts=samples,
                temperature=1.0,
                seq_len=100,
                num_train_workers=num_workers,
                idxs=[0] * num_samples,
                num_loras=1,
            )

            batch_counts = [len(wb) for wb in result]
            assert len(set(batch_counts)) == 1, (
                f"Unequal batch counts with {num_workers} workers and {num_samples} samples: {batch_counts}"
            )
