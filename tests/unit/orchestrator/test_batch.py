from types import SimpleNamespace

import numpy as np
import pytest

from prime_rl.trainer.batch import pad_micro_batch, prepare_batch, prepare_sample
from prime_rl.trainer.utils import build_bin_cost
from prime_rl.transport.types import EncodedTensor, MicroBatch, RoutedExperts, TrainingAdvantage, TrainingSample


def _advantage(loss: str, values: list[float], mask: list[bool]) -> TrainingAdvantage:
    return TrainingAdvantage(loss=loss, values=values, mask=mask)


@pytest.fixture
def make_training_example():
    def _make_training_example(
        temperature: float = 1.0,
        env_name: str = "test-env",
        advantages: list[TrainingAdvantage] | None = None,
    ) -> TrainingSample:
        mask = [False, False, True, True]
        return TrainingSample(
            token_ids=[1, 2, 3, 4],
            mask=mask,
            logprobs=[0.0, 0.0, -0.1, -0.2],
            temperatures=[temperature] * 4,
            advantages=advantages or [_advantage("rl", [0.0, 0.0, 1.0, 1.0], mask)],
            env_name=env_name,
        )

    return _make_training_example


def make_sized_training_example(length: int, env_name: str = "test-env") -> TrainingSample:
    assert length >= 1
    mask = [False] * (length - 1) + [True]
    return TrainingSample(
        token_ids=[1] * (length - 1) + [2],
        mask=mask,
        logprobs=[0.0] * (length - 1) + [-0.1],
        temperatures=[1.0] * length,
        advantages=[_advantage("rl", [0.0] * (length - 1) + [1.0], mask)],
        env_name=env_name,
    )


def _flatten_batches(batches_per_gpu: list[list[MicroBatch]]) -> list[MicroBatch]:
    return [batch for worker_batches in batches_per_gpu for batch in worker_batches]


def _worker_token_sums(batches_per_gpu: list[list[MicroBatch]]) -> list[int]:
    return [sum(len(batch.input_ids) for batch in worker_batches) for worker_batches in batches_per_gpu]


def _has_loss_tokens(batch: MicroBatch) -> bool:
    return any(any(channel.mask) for channel in batch.advantages)


def _channel(batch: MicroBatch, loss: str) -> TrainingAdvantage:
    matches = [channel for channel in batch.advantages if channel.loss == loss]
    assert len(matches) == 1
    return matches[0]


def _routed_experts(data, dtype=np.uint8):
    routed_experts = np.asarray(data, dtype=dtype)
    return RoutedExperts(
        data=routed_experts.tobytes(),
        shape=list(routed_experts.shape),
        dtype=str(routed_experts.dtype),
    )


def _encoded(arr) -> EncodedTensor:
    a = np.asarray(arr)
    return EncodedTensor(data=a.tobytes(), shape=list(a.shape), dtype=str(a.dtype))


def make_flops_config():
    return SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        head_dim=8,
    )


def test_training_sample_requires_env_name():
    with pytest.raises(TypeError, match="env_name"):
        TrainingSample(
            token_ids=[1, 2, 3, 4],
            mask=[False, False, True, True],
            logprobs=[0.0, 0.0, -0.1, -0.2],
            temperatures=[1.0] * 4,
            advantages=[],
        )


@pytest.mark.parametrize(
    ("rollout_count", "num_train_workers", "expected_batches_per_worker"),
    [(4, 2, 2), (5, 2, 3), (7, 1, 7), (11, 4, 3)],
)
def test_prepare_batch_balances_micro_batches_across_workers(
    make_training_example, rollout_count, num_train_workers, expected_batches_per_worker
):
    examples = [make_training_example() for _ in range(rollout_count)]

    batches_per_gpu = prepare_batch(
        rollouts=examples,
        seq_len=4,
        num_train_workers=num_train_workers,
        idxs=[0] * rollout_count,
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )

    assert all(len(worker_batches) == expected_batches_per_worker for worker_batches in batches_per_gpu)
    flat_batches = _flatten_batches(batches_per_gpu)
    assert len(examples) <= len(flat_batches) < len(examples) + num_train_workers
    assert len([batch for batch in flat_batches if _has_loss_tokens(batch)]) == len(examples)
    for batch in [batch for batch in flat_batches if not _has_loss_tokens(batch)]:
        assert batch.advantages == []


def test_randomized_packing_invariants():
    rng = np.random.default_rng(0)
    for case_idx in range(40):
        seq_len = int(rng.choice([8, 16, 32, 64]))
        num_train_workers = int(rng.choice([1, 2, 4, 8]))
        num_samples = int(rng.integers(1, 65))
        lengths = [int(x) for x in rng.integers(1, seq_len + 1, size=num_samples)]
        examples = [make_sized_training_example(length, env_name=f"env-{case_idx}") for length in lengths]
        bin_cost = build_bin_cost(make_flops_config() if case_idx % 2 == 0 else None)

        batches_per_gpu = prepare_batch(
            rollouts=examples,
            seq_len=seq_len,
            num_train_workers=num_train_workers,
            idxs=[0] * len(examples),
            num_loras=1,
            bin_cost=bin_cost,
        )
        flat_batches = _flatten_batches(batches_per_gpu)
        real_batches = [batch for batch in flat_batches if _has_loss_tokens(batch)]

        assert all(len(worker_batches) == len(batches_per_gpu[0]) for worker_batches in batches_per_gpu)
        assert sorted(length for batch in real_batches for length in batch.sequence_lengths) == sorted(lengths)
        for batch in flat_batches:
            assert len(batch.input_ids) <= seq_len
            assert sum(batch.sequence_lengths) == len(batch.input_ids)
            assert sum(batch.lora_num_tokens) == len(batch.input_ids)
            assert len(batch.env_names) == len(batch.input_ids)
            for channel in batch.advantages:
                assert len(channel.values) == len(batch.input_ids)
                assert len(channel.mask) == len(batch.input_ids)


def test_prepare_batch_packs_different_temperatures(make_training_example):
    example1 = make_training_example(temperature=0.7, env_name="env-a")
    example2 = make_training_example(temperature=1.1, env_name="env-b")

    batches_per_gpu = prepare_batch(
        rollouts=[example1, example2],
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )

    batch = _flatten_batches(batches_per_gpu)[0]
    assert batch.temperatures == [0.7] * 4 + [1.1] * 4
    assert batch.env_names == ["env-a"] * 4 + ["env-b"] * 4
    assert batch.sequence_lengths == [4, 4]


def test_pad_micro_batch_preserves_explicit_sequence_lengths(make_training_example):
    micro_batch = prepare_sample(make_training_example(), seq_len=16)

    padded = pad_micro_batch(micro_batch, pad_to_multiple_of=6)

    assert len(padded.input_ids) == 6
    assert padded.sequence_lengths == [4, 2]
    assert sum(padded.sequence_lengths) == len(padded.input_ids)
    assert _channel(padded, "rl").mask[-2:] == [False, False]


def test_split_to_align_avoids_dummy_micro_batches():
    examples = [make_sized_training_example(length) for length in [6, 6, 5, 5, 4, 4]]

    batches_per_gpu = prepare_batch(
        rollouts=examples,
        seq_len=12,
        num_train_workers=4,
        idxs=[0] * len(examples),
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )

    assert all(_has_loss_tokens(batch) for batch in _flatten_batches(batches_per_gpu))
    assert len(_flatten_batches(batches_per_gpu)) == 4


def test_pack_first_then_balance_distributes_micro_batches_by_tokens_without_model_config():
    examples = [make_sized_training_example(length) for length in [100, 90, 80, 70]]

    balanced = prepare_batch(
        rollouts=examples,
        seq_len=100,
        num_train_workers=2,
        idxs=[0] * len(examples),
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )

    assert _worker_token_sums(balanced) == [170, 170]


def test_flop_aware_balancing_pairs_long_and_short_sequence_workloads():
    examples = [make_sized_training_example(length) for length in [32, 32, 16, 16, 16, 16]]
    bin_cost = build_bin_cost(make_flops_config())

    balanced = prepare_batch(
        rollouts=examples,
        seq_len=32,
        num_train_workers=2,
        idxs=[0] * len(examples),
        num_loras=1,
        bin_cost=bin_cost,
    )

    assert sorted([sorted(batch.sequence_lengths) for batch in balanced[0]]) == [[16, 16], [32]]
    assert sorted([sorted(batch.sequence_lengths) for batch in balanced[1]]) == [[16, 16], [32]]
    assert bin_cost([32]) > bin_cost([16, 16])


def test_prepare_sample_truncates_channels(make_training_example):
    sample = make_training_example(
        advantages=[
            _advantage("rl", [0.0, 0.0, 1.0, 1.0], [False, False, True, True]),
            _advantage("ce", [0.0, 0.5, 0.0, 0.0], [False, True, False, False]),
        ]
    )

    micro_batch = prepare_sample(sample, seq_len=3)

    assert micro_batch.input_ids == [1, 2, 3]
    assert _channel(micro_batch, "rl").values == [0.0, 0.0, 1.0]
    assert _channel(micro_batch, "rl").mask == [False, False, True]
    assert _channel(micro_batch, "ce").values == [0.0, 0.5, 0.0]
    assert _channel(micro_batch, "ce").mask == [False, True, False]


def test_prepare_batch_aligns_loss_channels_in_mixed_bins(make_training_example):
    longer = TrainingSample(
        token_ids=[1, 2, 3, 4, 5, 6],
        mask=[False, False, False, True, True, True],
        logprobs=[0.0, 0.0, 0.0, -0.1, -0.1, -0.1],
        temperatures=[1.0] * 6,
        advantages=[_advantage("ce", [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [False, False, False, True, True, True])],
        env_name="test-env",
    )
    shorter = make_training_example()

    batches_per_gpu = prepare_batch(
        rollouts=[longer, shorter],
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )

    batch = _flatten_batches(batches_per_gpu)[0]
    assert _channel(batch, "ce").values == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 4
    assert _channel(batch, "ce").mask == [False, False, False, True, True, True] + [False] * 4
    assert _channel(batch, "rl").values == [0.0] * 6 + [0.0, 0.0, 1.0, 1.0]
    assert _channel(batch, "rl").mask == [False] * 6 + [False, False, True, True]


def test_prepare_sample_with_routed_experts():
    routed_experts = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 2], [1, 3]], [[1, 0], [3, 2]]]
    routed_payload = _routed_experts(routed_experts)
    sample = TrainingSample(
        token_ids=[1, 2, 3, 4],
        mask=[False, False, True, True],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        temperatures=[1.0] * 4,
        advantages=[_advantage("rl", [0.0, 0.0, 1.0, 1.0], [False, False, True, True])],
        env_name="test-env",
        routed_experts=routed_payload,
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is not None
    assert micro_batch.routed_experts == routed_payload


def test_prepare_sample_truncates_routed_experts():
    routed_experts = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]]
    routed_payload = _routed_experts(routed_experts)
    expected_payload = _routed_experts(routed_experts[:3])
    sample = TrainingSample(
        token_ids=[1, 2, 3, 4],
        mask=[False, False, True, True],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        temperatures=[1.0] * 4,
        advantages=[_advantage("rl", [0.0, 0.0, 1.0, 1.0], [False, False, True, True])],
        env_name="test-env",
        routed_experts=routed_payload,
    )

    micro_batch = prepare_sample(sample, seq_len=3)
    assert micro_batch.routed_experts is not None
    assert micro_batch.routed_experts.data == expected_payload.data
    assert micro_batch.routed_experts.shape == expected_payload.shape


def test_prepare_sample_truncates_mm_at_image_boundary():
    mm_token_type_ids = [0, 1, 1, 0, 1, 1, 0]
    pixel_values = np.array([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32)
    grid = np.array([[1, 2, 1], [1, 2, 1]], dtype=np.int64)
    mask = [False, False, False, False, False, True, True]
    sample = TrainingSample(
        token_ids=[10, 11, 12, 13, 14, 15, 16],
        mask=mask,
        logprobs=[0.0] * 7,
        temperatures=[1.0] * 7,
        advantages=[_advantage("rl", [1.0] * 7, mask)],
        env_name="test-env",
        mm_token_type_ids=mm_token_type_ids,
        mm_kwargs={"pixel_values": _encoded(pixel_values), "image_grid_thw": _encoded(grid)},
    )

    micro_batch = prepare_sample(sample, seq_len=5)

    assert len(micro_batch.input_ids) == 4
    assert len(micro_batch.mm_token_type_ids) == len(micro_batch.input_ids)
    n_placeholders = sum(1 for token_type in micro_batch.mm_token_type_ids if token_type)
    assert n_placeholders == 2
    assert micro_batch.mm_kwargs["pixel_values"].shape == [2, 1]
    assert micro_batch.mm_kwargs["image_grid_thw"].shape == [1, 3]
    kept = np.frombuffer(bytearray(micro_batch.mm_kwargs["pixel_values"].data), dtype=np.float32)
    assert kept.tolist() == [1.0, 1.0]
    assert n_placeholders == micro_batch.mm_kwargs["pixel_values"].shape[0]
