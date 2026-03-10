import pytest

from prime_rl.trainer.batch import _IMAGE_PAD_TOKEN_ID, _trim_multimodal_to_match, prepare_batch, prepare_sample
from prime_rl.transport.types import TrainingSample


@pytest.fixture
def make_training_example():
    def _make_training_example(temperature: float = 1.0) -> TrainingSample:
        return TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[False, False],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[temperature, temperature],  # Per-token temperatures
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


IMG = _IMAGE_PAD_TOKEN_ID
GRID = [1, 48, 64]  # 3072 patches -> 768 tokens
TOKENS_PER_IMG = 768
PATCHES_PER_IMG = 3072
PATCH_DIM = 1176


def _make_pixel_values(n):
    patches = n * PATCHES_PER_IMG
    return b"\x00" * (patches * 4 * PATCH_DIM), [patches, PATCH_DIM], [GRID] * n


def _make_vlm_sample(n_images, text_per_image=50, completion=100):
    prompt_ids = []
    for _ in range(n_images):
        prompt_ids += [100] * text_per_image + [IMG] * TOKENS_PER_IMG
    prompt_ids += [100] * 50
    completion_ids = [101] * completion
    pv, shape, grids = _make_pixel_values(n_images)
    return TrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=[True] * len(completion_ids),
        completion_logprobs=[-0.5] * len(completion_ids),
        completion_temperatures=[0.7] * len(completion_ids),
        advantage=1.0,
        pixel_values=pv,
        pixel_values_shape=shape,
        image_grid_thw=grids,
    )


def test_trim_noop_when_tokens_match():
    input_ids = [100] * 50 + [IMG] * TOKENS_PER_IMG + [100] * 50
    pv, shape, grids = _make_pixel_values(1)
    new_ids, keep_mask, *_ = _trim_multimodal_to_match(input_ids, pv, shape, grids)
    assert keep_mask is None
    assert new_ids is input_ids


def test_trim_noop_for_text_only():
    input_ids = [100] * 100
    new_ids, keep_mask, *_ = _trim_multimodal_to_match(input_ids, None, None, None)
    assert keep_mask is None


def test_trim_drops_partial_image():
    pv, shape, grids = _make_pixel_values(5)
    partial = 516
    input_ids = []
    for _ in range(4):
        input_ids += [100] * 50 + [IMG] * TOKENS_PER_IMG
    input_ids += [100] * 50 + [IMG] * partial

    new_ids, keep_mask, _, _, new_grids = _trim_multimodal_to_match(input_ids, pv, shape, grids)

    assert keep_mask is not None
    assert sum(1 for t in new_ids if t == IMG) == 4 * TOKENS_PER_IMG
    assert len(new_grids) == 4
    assert len(new_ids) == len(input_ids) - partial


def test_trim_drops_all_images():
    pv, shape, grids = _make_pixel_values(1)
    input_ids = [100] * 50 + [IMG] * 100  # partial, no complete image

    new_ids, _, new_pv, _, new_grids = _trim_multimodal_to_match(input_ids, pv, shape, grids)

    assert new_pv is None
    assert new_grids is None
    assert sum(1 for t in new_ids if t == IMG) == 0


def test_prepare_sample_vlm_seq_len_truncation():
    sample = _make_vlm_sample(5, text_per_image=200, completion=500)
    assert len(sample.prompt_ids) + len(sample.completion_ids) > 4096

    mb = prepare_sample(sample, seq_len=4096)

    img_tokens = sum(1 for t in mb.input_ids if t == IMG)
    features = sum(g[0] * g[1] * g[2] // 4 for g in mb.image_grid_thw)
    assert img_tokens == features
    assert len(mb.input_ids) == len(mb.loss_mask) == len(mb.advantages)


def test_prepare_sample_vlm_already_truncated_by_vllm():
    """vLLM left-truncates prompt to max_model_len, losing image tokens,
    but pixel_values still has all images."""
    pv, shape, grids = _make_pixel_values(5)
    partial = 516
    prompt_ids = [100] * 50 + [IMG] * partial  # truncated first image
    for _ in range(3):
        prompt_ids += [100] * 50 + [IMG] * TOKENS_PER_IMG
    prompt_ids += [100] * 50
    completion_ids = [101] * (4096 - len(prompt_ids))

    sample = TrainingSample(
        prompt_ids=prompt_ids,
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=completion_ids,
        completion_mask=[True] * len(completion_ids),
        completion_logprobs=[-0.5] * len(completion_ids),
        completion_temperatures=[0.7] * len(completion_ids),
        advantage=1.0,
        pixel_values=pv,
        pixel_values_shape=shape,
        image_grid_thw=grids,
    )
    assert len(sample.prompt_ids) + len(sample.completion_ids) == 4096

    mb = prepare_sample(sample, seq_len=4096)

    img_tokens = sum(1 for t in mb.input_ids if t == IMG)
    features = sum(g[0] * g[1] * g[2] // 4 for g in mb.image_grid_thw)
    assert img_tokens == features
    assert len(mb.input_ids) == len(mb.loss_mask) == len(mb.advantages)
