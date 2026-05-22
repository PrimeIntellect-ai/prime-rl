import pytest

from prime_rl.trainer.batch import prepare_batch, prepare_sample
from prime_rl.transport.types import TrainingSample


@pytest.fixture
def make_training_example():
    def _make_training_example(
        temperature: float = 1.0,
        training_mode: str = "rl",
        env_name: str = "test-env",
    ) -> TrainingSample:
        return TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[False, False],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperature=temperature,
            teacher_logprobs=[0.0, 0.0, 0.0, 0.0],
            advantage=1.0,
            env_name=env_name,
            training_mode=training_mode,
        )

    return _make_training_example


def test_training_sample_requires_env_name():
    with pytest.raises(TypeError, match="env_name"):
        TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[False, False],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperature=1.0,
            advantage=1.0,
        )


@pytest.mark.parametrize(
    ("rollout_count", "num_train_workers", "expected_batches_per_worker"), [(4, 2, 2), (5, 2, 3), (7, 1, 7), (11, 4, 3)]
)
def test_prepare_batch_balances_micro_batches_across_workers(
    make_training_example, rollout_count, num_train_workers, expected_batches_per_worker
):
    examples = [make_training_example() for i in range(rollout_count)]

    batches_per_gpu = prepare_batch(
        rollouts=examples,
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


def test_prepare_batch_does_not_pack_different_temperatures(make_training_example):
    """Temperature is a scalar per micro batch, so samples with different
    temperatures land in separate micro batches even when they would fit."""
    example1 = make_training_example(temperature=0.7, env_name="env-a")
    example2 = make_training_example(temperature=1.1, env_name="env-b")

    batches_per_gpu = prepare_batch(
        rollouts=[example1, example2],
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
    )

    flat_batches = [batch for worker_batches in batches_per_gpu for batch in worker_batches]
    assert len(flat_batches) == 2
    assert {b.temperature for b in flat_batches} == {0.7, 1.1}
    # Each micro batch holds exactly one sample (4 tokens: 2 prompt + 2 completion)
    for batch in flat_batches:
        assert len(batch.input_ids) == 4


def test_prepare_sample_propagates_training_mode(make_training_example):
    example = make_training_example(training_mode="sft")

    micro_batch = prepare_sample(example, seq_len=16)

    assert micro_batch.training_mode == "sft"


def test_prepare_batch_does_not_pack_mixed_training_mode(make_training_example):
    rl_example = make_training_example(training_mode="rl")
    sft_example = make_training_example(training_mode="sft")

    batches_per_gpu = prepare_batch(
        rollouts=[rl_example, sft_example],
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
    )

    flat_batches = [batch for worker_batches in batches_per_gpu for batch in worker_batches]
    assert len(flat_batches) == 2
    assert {batch.training_mode for batch in flat_batches} == {"rl", "sft"}


def test_prepare_sample_with_routed_experts():
    """Routed experts are passed through prepare_sample and match input_ids length."""
    # 2 prompt + 2 completion = 4 tokens, 2 layers, topk=2
    routed_experts = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 2], [1, 3]], [[1, 0], [3, 2]]]
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperature=1.0,
        advantage=1.0,
        env_name="test-env",
        routed_experts=routed_experts,
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is not None
    assert len(micro_batch.routed_experts) == 4
    assert micro_batch.routed_experts == routed_experts


def test_prepare_sample_truncates_routed_experts():
    """Routed experts are truncated to seq_len when input exceeds it."""
    routed_experts = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]]
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperature=1.0,
        advantage=1.0,
        env_name="test-env",
        routed_experts=routed_experts,
    )

    micro_batch = prepare_sample(sample, seq_len=3)
    assert micro_batch.routed_experts is not None
    assert len(micro_batch.routed_experts) == 3
    assert micro_batch.routed_experts == routed_experts[:3]
    assert micro_batch.env_names == ["test-env"] * 3


def test_prepare_sample_none_routed_experts():
    """When routed_experts is None, micro_batch.routed_experts is None."""
    sample = TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperature=1.0,
        advantage=1.0,
        env_name="test-env",
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is None
