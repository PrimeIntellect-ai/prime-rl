import pytest

from prime_rl.trainer.batch import prepare_batch, prepare_sample
from prime_rl.transport.types import TrainingSample


@pytest.fixture
def make_training_example():
    def _make_training_example(
        temperature: float = 1.0,
        sft_loss: bool = False,
        env_name: str = "test-env",
    ) -> TrainingSample:
        return TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[False, False],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[temperature, temperature],  # Per-token temperatures
            teacher_logprobs=[0.0, 0.0, 0.0, 0.0],
            advantage=1.0,
            env_name=env_name,
            sft_loss=sft_loss,
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
            completion_temperatures=[1.0, 1.0],
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


def test_prepare_batch_packs_different_temperatures(make_training_example):
    """With per-token temperatures, samples can be packed together regardless of their temperature values."""
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
    # With per-token temperatures, samples can now be packed together
    assert len(flat_batches) == 1
    # Each sample has 4 tokens (2 prompt + 2 completion), so 8 total tokens
    assert len(flat_batches[0].temperatures) == 8
    # First sample (4 tokens): all get temp 0.7
    assert flat_batches[0].temperatures[:4] == [0.7, 0.7, 0.7, 0.7]
    # Second sample (4 tokens): all get temp 1.1
    assert flat_batches[0].temperatures[4:8] == [1.1, 1.1, 1.1, 1.1]
    assert flat_batches[0].env_names == ["env-a"] * 4 + ["env-b"] * 4


def test_prepare_sample_propagates_sft_loss(make_training_example):
    example = make_training_example(sft_loss=True)

    micro_batch = prepare_sample(example, seq_len=16)

    assert micro_batch.sft_loss is True


def test_prepare_sample_overlays_sft_advantage_length_normalized(make_training_example):
    """Length-normalized (ECHO) overlay: each sft_mask position gets
    ``alpha / n_sft_tokens`` on the advantages tensor, and its loss_mask
    flips to True so the SFT gradient reaches it."""
    # 2 prompt + 2 completion = 4 tokens; mark both prompt positions as SFT.
    example = make_training_example()
    example.sft_mask = [True, True, False, False]
    example.sft_alpha = 0.5

    micro_batch = prepare_sample(example, seq_len=16)

    # n_sft = 2, alpha = 0.5 → weight = 0.25 on mask positions.
    assert micro_batch.advantages[0] == 0.25
    assert micro_batch.advantages[1] == 0.25
    # Non-SFT positions keep the rollout's scalar advantage (1.0 from the fixture).
    assert micro_batch.advantages[2] == 1.0
    assert micro_batch.advantages[3] == 1.0
    # SFT prompt positions are now loss-trainable; completion mask preserved.
    assert micro_batch.loss_mask == [True, True, True, True]
    # The mask itself rides through unchanged.
    assert micro_batch.sft_mask == [True, True, False, False]


def test_prepare_sample_overlays_sft_advantage_disable_echo(make_training_example):
    """When ``disable_echo=True`` the weight is a constant ``alpha``,
    not ``alpha / n_sft_tokens``. Useful for the ablation cell on the
    ECHO normalization."""
    example = make_training_example()
    example.sft_mask = [True, True, False, False]
    example.sft_alpha = 0.5

    micro_batch = prepare_sample(example, seq_len=16, disable_echo=True)

    # Constant alpha across all SFT positions.
    assert micro_batch.advantages[0] == 0.5
    assert micro_batch.advantages[1] == 0.5
    assert micro_batch.advantages[2] == 1.0
    assert micro_batch.advantages[3] == 1.0


def test_prepare_sample_skips_sft_overlay_without_alpha(make_training_example):
    """Carrying ``sft_mask`` without ``sft_alpha`` is a defensive
    no-op — the overlay only fires when both are set. Lets the
    orchestrator emit the mask conditionally without forcing alpha."""
    example = make_training_example()
    example.sft_mask = [True, True, False, False]
    example.sft_alpha = None

    micro_batch = prepare_sample(example, seq_len=16)

    # No advantage rewrite; original scalar fills every position.
    assert all(adv == 1.0 for adv in micro_batch.advantages)
    # No loss_mask flip on the SFT-mask positions either.
    assert micro_batch.loss_mask == [False, False, True, True]


def test_prepare_sample_truncates_sft_mask_with_other_per_token_lists(make_training_example):
    """Truncation slices ``sft_mask`` in lockstep with ``input_ids``,
    keeping the length-equality assertion green."""
    example = make_training_example()
    example.sft_mask = [True, True, False, False]
    example.sft_alpha = 0.5

    micro_batch = prepare_sample(example, seq_len=2)

    assert len(micro_batch.input_ids) == 2
    assert len(micro_batch.sft_mask) == 2
    assert len(micro_batch.advantages) == 2
    assert len(micro_batch.loss_mask) == 2


def test_prepare_batch_does_not_pack_mixed_sft_loss(make_training_example):
    rl_example = make_training_example(sft_loss=False)
    sft_example = make_training_example(sft_loss=True)

    batches_per_gpu = prepare_batch(
        rollouts=[rl_example, sft_example],
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
    )

    flat_batches = [batch for worker_batches in batches_per_gpu for batch in worker_batches]
    assert len(flat_batches) == 2
    assert {batch.sft_loss for batch in flat_batches} == {False, True}


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
        completion_temperatures=[1.0, 1.0],
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
        completion_temperatures=[1.0, 1.0],
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
        completion_temperatures=[1.0, 1.0],
        advantage=1.0,
        env_name="test-env",
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is None
