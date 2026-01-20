import pytest

from prime_rl.trainer.batch import prepare_batch
from prime_rl.transport.types import TrainingSample


@pytest.fixture
def make_training_example():
    def _make_training_example() -> TrainingSample:
        return TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[False, False],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            teacher_logprobs=[0.0, 0.0, 0.0, 0.0],
            advantage=1.0,
        )

    return _make_training_example


@pytest.mark.parametrize(
    ("rollout_count", "num_train_workers", "expected_batches_per_worker"), [(4, 2, 2), (5, 2, 3), (7, 1, 7), (11, 4, 3)]
)
def test_prepare_batch_balances_micro_batches_across_workers(
    make_training_example, rollout_count, num_train_workers, expected_batches_per_worker
):
    examples = [make_training_example() for i in range(rollout_count)]

    batches_per_gpu = prepare_batch(
        rollouts=examples,
        temperatures=[0.5] * rollout_count,
        seq_len=4,
        num_train_workers=num_train_workers,
        idxs=[0] * rollout_count,
        num_loras=1,
    )

    assert all(len(worker_batches) == expected_batches_per_worker for worker_batches in batches_per_gpu)

    flat_batches = [batch for worker_batches in batches_per_gpu for batch in worker_batches]
    assert len(examples) <= len(flat_batches) < len(examples) + num_train_workers
    print(flat_batches)

    # Verify real rollouts have expected non-zero advantages and loss mask
    for batch in flat_batches[: len(examples)]:
        print(batch)
        assert sum(1 for advantage in batch.advantages if advantage != 0.0) == 4
        assert sum(1 for loss_mask in batch.loss_mask if loss_mask) == 2

    # Verify padded batches have zero advantages and loss mask
    for batch in flat_batches[len(examples) :]:
        assert sum(1 for advantage in batch.advantages if advantage != 0.0) == 0
        assert sum(1 for loss_mask in batch.loss_mask if loss_mask) == 0


def test_prepare_batch_splits_by_temperature(make_training_example):
    examples = [make_training_example() for _ in range(2)]
    temps = [0.7, 1.1]

    batches_per_gpu = prepare_batch(
        rollouts=examples,
        temperatures=temps,
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
    )

    flat_batches = [batch for worker_batches in batches_per_gpu for batch in worker_batches]
    assert len(flat_batches) == 2
    assert {batch.temperature for batch in flat_batches} == set(temps)
