import numpy as np
import pytest

from prime_rl.trainer.batch import prepare_batch, prepare_sample
from prime_rl.transport.types import EncodedTensor, RoutedExperts, TrainingSample


def _routed_experts(data, dtype=np.uint8):
    routed_experts = np.asarray(data, dtype=dtype)
    return RoutedExperts(
        data=routed_experts.tobytes(),
        shape=list(routed_experts.shape),
        dtype=str(routed_experts.dtype),
    )


@pytest.fixture
def make_training_example():
    def _make_training_example(
        temperature: float = 1.0,
        training_mode: str = "rl",
        env_name: str = "test-env",
    ) -> TrainingSample:
        return TrainingSample(
            token_ids=[1, 2, 3, 4],
            mask=[False, False, True, True],
            logprobs=[0.0, 0.0, -0.1, -0.2],
            temperatures=[temperature, temperature, temperature, temperature],
            advantages=[1.0, 1.0, 1.0, 1.0],
            env_name=env_name,
            training_mode=training_mode,
        )

    return _make_training_example


def test_training_sample_requires_env_name():
    with pytest.raises(TypeError, match="env_name"):
        TrainingSample(
            token_ids=[1, 2, 3, 4],
            mask=[False, False, True, True],
            logprobs=[0.0, 0.0, -0.1, -0.2],
            temperatures=[1.0, 1.0, 1.0, 1.0],
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
    routed_payload = _routed_experts(routed_experts)
    sample = TrainingSample(
        token_ids=[1, 2, 3, 4],
        mask=[False, False, True, True],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0, 1.0],
        advantages=[1.0, 1.0, 1.0, 1.0],
        env_name="test-env",
        routed_experts=routed_payload,
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is not None
    assert micro_batch.routed_experts == routed_payload


def test_prepare_sample_truncates_routed_experts():
    """Routed experts are truncated to seq_len when input exceeds it."""
    routed_experts = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]]
    routed_payload = _routed_experts(routed_experts)
    expected_payload = _routed_experts(routed_experts[:3])
    sample = TrainingSample(
        token_ids=[1, 2, 3, 4],
        mask=[False, False, True, True],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0, 1.0],
        advantages=[1.0, 1.0, 1.0, 1.0],
        env_name="test-env",
        routed_experts=routed_payload,
    )

    micro_batch = prepare_sample(sample, seq_len=3)
    assert micro_batch.routed_experts is not None
    assert micro_batch.routed_experts == expected_payload
    assert micro_batch.env_names == ["test-env"] * 3


def _encoded(arr) -> EncodedTensor:
    a = np.asarray(arr)
    return EncodedTensor(data=a.tobytes(), shape=list(a.shape), dtype=str(a.dtype))


def test_prepare_sample_truncates_mm_at_image_boundary():
    """Truncation never splits an image's placeholder block: it cuts to a whole-image boundary
    and slices mm_kwargs to match, so image-token count stays == image-embedding count."""
    # Two 2-token images (patches-per-token = 1): image-pad at indices 1,2 (img0) and 4,5 (img1).
    mm_token_type_ids = [0, 1, 1, 0, 1, 1, 0]
    pixel_values = np.array([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32)  # img0=1.0, img1=2.0
    grid = np.array([[1, 2, 1], [1, 2, 1]], dtype=np.int64)
    sample = TrainingSample(
        token_ids=[10, 11, 12, 13, 14, 15, 16],
        mask=[False, False, False, False, False, True, True],
        logprobs=[0.0] * 7,
        temperatures=[1.0] * 7,
        advantages=[1.0] * 7,
        env_name="test-env",
        mm_token_type_ids=mm_token_type_ids,
        mm_kwargs={"pixel_values": _encoded(pixel_values), "image_grid_thw": _encoded(grid)},
    )

    # seq_len=5 falls inside img1 (one of its two placeholders survives) -> drop img1 entirely.
    mb = prepare_sample(sample, seq_len=5)
    assert len(mb.input_ids) == 4  # cut back to img1's first placeholder (index 4)
    assert len(mb.mm_token_type_ids) == len(mb.input_ids)
    n_placeholders = sum(1 for t in mb.mm_token_type_ids if t)
    assert n_placeholders == 2  # only img0's two placeholders remain
    # No mismatch: placeholders == image embeddings, and only img0's pixels are kept.
    assert mb.mm_kwargs["pixel_values"].shape == [2, 1]
    assert mb.mm_kwargs["image_grid_thw"].shape == [1, 3]
    kept = np.frombuffer(bytearray(mb.mm_kwargs["pixel_values"].data), dtype=np.float32)
    assert kept.tolist() == [1.0, 1.0]
    assert n_placeholders == mb.mm_kwargs["pixel_values"].shape[0]  # ppt == 1 here


def test_prepare_sample_none_routed_experts():
    """When routed_experts is None, micro_batch.routed_experts is None."""
    sample = TrainingSample(
        token_ids=[1, 2, 3, 4],
        mask=[False, False, True, True],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0, 1.0],
        advantages=[1.0, 1.0, 1.0, 1.0],
        env_name="test-env",
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is None
