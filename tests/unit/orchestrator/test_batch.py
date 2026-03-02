import pytest

from prime_rl.trainer.batch import _is_multimodal_sample, prepare_batch, prepare_sample
from prime_rl.transport.types import TrainingSample


@pytest.fixture
def make_training_example():
    def _make_training_example(
        temperature: float = 1.0,
        pixel_values: list[list[float]] | None = None,
        image_grid_thw: list[list[int]] | None = None,
    ) -> TrainingSample:
        return TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[False, False],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[temperature, temperature],
            teacher_logprobs=[0.0, 0.0, 0.0, 0.0],
            advantage=1.0,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
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
    example1 = make_training_example(temperature=0.7)
    example2 = make_training_example(temperature=1.1)

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
        routed_experts=routed_experts,
    )

    micro_batch = prepare_sample(sample, seq_len=3)
    assert micro_batch.routed_experts is not None
    assert len(micro_batch.routed_experts) == 3
    assert micro_batch.routed_experts == routed_experts[:3]


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
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is None


# --- Multimodal type-alignment tests ---

DUMMY_PIXEL_VALUES = [[0.1, 0.2, 0.3]]
DUMMY_GRID_THW = [[1, 2, 2]]


def _make_mm_example() -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        teacher_logprobs=[0.0, 0.0, 0.0, 0.0],
        advantage=1.0,
        pixel_values=DUMMY_PIXEL_VALUES,
        image_grid_thw=DUMMY_GRID_THW,
    )


def _make_text_example() -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.1, -0.2],
        completion_temperatures=[1.0, 1.0],
        teacher_logprobs=[0.0, 0.0, 0.0, 0.0],
        advantage=1.0,
    )


def test_mixed_mm_text_type_aligned_across_workers():
    """At every micro_step, all GPUs must process the same type (MM or text)."""
    # 2 MM + 2 text with 2 workers — naive chunking would misalign
    rollouts = [_make_mm_example(), _make_text_example(), _make_text_example(), _make_mm_example()]
    batches_per_gpu = prepare_batch(rollouts=rollouts, seq_len=4, num_train_workers=2, idxs=[0] * 4, num_loras=1)

    assert len(batches_per_gpu) == 2
    # Every GPU must have the same number of micro batches
    assert len(batches_per_gpu[0]) == len(batches_per_gpu[1])

    for step in range(len(batches_per_gpu[0])):
        types = [_is_multimodal_sample(batches_per_gpu[gpu][step]) for gpu in range(2)]
        assert types[0] == types[1], f"Type mismatch at micro_step {step}: {types}"


def test_mm_padding_batches_preserve_pixel_values():
    """Padding batches for the MM group must retain pixel_values so the vision encoder runs."""
    # 3 MM samples with 2 workers -> needs 1 MM padding batch
    rollouts = [_make_mm_example() for _ in range(3)]
    batches_per_gpu = prepare_batch(rollouts=rollouts, seq_len=4, num_train_workers=2, idxs=[0] * 3, num_loras=1)

    flat = [b for gpu_batches in batches_per_gpu for b in gpu_batches]
    mm_batches = [b for b in flat if _is_multimodal_sample(b)]

    # 3 real + 1 padding = 4 MM batches
    assert len(mm_batches) == 4
    # All MM batches (including padding) must have pixel_values
    for b in mm_batches:
        assert b.pixel_values is not None
        assert b.image_grid_thw is not None

    # The padding batch should have zero advantages
    padding_batches = [b for b in mm_batches if all(a == 0.0 for a in b.advantages)]
    assert len(padding_batches) == 1
    assert all(not m for m in padding_batches[0].loss_mask)


def test_all_multimodal_batches():
    """All-multimodal edge case: round-robin distribution works correctly."""
    rollouts = [_make_mm_example() for _ in range(4)]
    batches_per_gpu = prepare_batch(rollouts=rollouts, seq_len=4, num_train_workers=2, idxs=[0] * 4, num_loras=1)

    assert len(batches_per_gpu) == 2
    assert len(batches_per_gpu[0]) == len(batches_per_gpu[1]) == 2

    for step in range(2):
        for gpu in range(2):
            assert _is_multimodal_sample(batches_per_gpu[gpu][step])


def test_pure_text_only_unchanged(make_training_example):
    """Pure text-only training is a no-op — same behavior as before."""
    examples = [make_training_example() for _ in range(5)]
    batches_per_gpu = prepare_batch(rollouts=examples, seq_len=4, num_train_workers=2, idxs=[0] * 5, num_loras=1)

    assert len(batches_per_gpu) == 2
    # 5 samples -> 5 micro batches (seq_len=4, each sample is 4 tokens) -> pad to 6 -> 3 each
    assert len(batches_per_gpu[0]) == 3
    assert len(batches_per_gpu[1]) == 3

    for gpu_batches in batches_per_gpu:
        for b in gpu_batches:
            assert not _is_multimodal_sample(b)
