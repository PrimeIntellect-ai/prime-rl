import pytest
import torch

from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.orchestrator.types import TrainingExample


@pytest.fixture
def make_training_example():
    def _make_training_example() -> TrainingExample:
        return TrainingExample(
            prompt_ids=[1, 2],
            prompt_mask=[0, 0],
            completion_ids=[3, 4],
            completion_mask=[1, 1],
            completion_logprobs=[-0.1, -0.2],
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
        temperature=0.5,
        seq_len=4,
        num_train_workers=num_train_workers,
    )

    assert all(len(worker_batches) == expected_batches_per_worker for worker_batches in batches_per_gpu)

    flat_batches = [batch for worker_batches in batches_per_gpu for batch in worker_batches]
    assert len(examples) <= len(flat_batches) < len(examples) + num_train_workers
    print(flat_batches)

    # Verify real rollouts have expected non-zero advantages and loss mask
    for batch in flat_batches[: len(examples)]:
        print(batch)
        assert torch.count_nonzero(batch["advantages"]) == 4
        assert torch.count_nonzero(batch["loss_mask"]) == 2

    # Verify padded batches have zero advantages and loss mask
    for batch in flat_batches[len(examples) :]:
        assert torch.count_nonzero(batch["advantages"]) == 0
        assert torch.count_nonzero(batch["loss_mask"]) == 0
