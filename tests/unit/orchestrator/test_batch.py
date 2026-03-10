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


# --- VLM multimodal trimming tests ---

IMG = _IMAGE_PAD_TOKEN_ID
# Each image: grid [1, 48, 64] -> 3072 patches -> 768 tokens (3072 / merge_size^2)
GRID = [1, 48, 64]
TOKENS_PER_IMAGE = 768
PATCHES_PER_IMAGE = 3072
PATCH_DIM = 1176


def _make_pixel_values(num_images: int) -> tuple[bytes, list[int], list[list[int]]]:
    total_patches = num_images * PATCHES_PER_IMAGE
    pv = b"\x00" * (total_patches * 4 * PATCH_DIM)
    shape = [total_patches, PATCH_DIM]
    grids = [GRID] * num_images
    return pv, shape, grids


def _make_vlm_sample(
    num_images: int,
    text_tokens_per_image: int = 50,
    trailing_text: int = 50,
    extra_completion: int = 0,
) -> TrainingSample:
    """Build a VLM TrainingSample with interleaved text and image tokens."""
    prompt_ids = []
    for _ in range(num_images):
        prompt_ids.extend([100] * text_tokens_per_image)
        prompt_ids.extend([IMG] * TOKENS_PER_IMAGE)
    prompt_ids.extend([100] * trailing_text)

    completion_ids = [101] * max(extra_completion, 100)
    pv, shape, grids = _make_pixel_values(num_images)

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


def test_trim_multimodal_noop_when_tokens_match():
    """No trimming when image tokens already match image_grid_thw."""
    input_ids = [100] * 50 + [IMG] * TOKENS_PER_IMAGE + [100] * 50
    pv, shape, grids = _make_pixel_values(1)

    new_ids, keep_mask, new_pv, new_shape, new_grids = _trim_multimodal_to_match(input_ids, pv, shape, grids)

    assert keep_mask is None
    assert new_ids is input_ids
    assert new_pv is pv
    assert new_grids == grids


def test_trim_multimodal_noop_for_text_only():
    """No trimming for text-only samples (pixel_values=None)."""
    input_ids = [100] * 100

    new_ids, keep_mask, new_pv, new_shape, new_grids = _trim_multimodal_to_match(input_ids, None, None, None)

    assert keep_mask is None
    assert new_ids is input_ids
    assert new_pv is None


def test_trim_multimodal_drops_partial_image():
    """When truncation leaves a partial image, drop it and its orphaned tokens."""
    # 5 images but only 4 complete + 516 partial tokens remain
    pv, shape, grids = _make_pixel_values(5)
    partial = 516
    input_ids = (
        [100] * 50
        + [IMG] * TOKENS_PER_IMAGE  # image 1
        + [100] * 50
        + [IMG] * TOKENS_PER_IMAGE  # image 2
        + [100] * 50
        + [IMG] * TOKENS_PER_IMAGE  # image 3
        + [100] * 50
        + [IMG] * TOKENS_PER_IMAGE  # image 4
        + [100] * 50
        + [IMG] * partial  # image 5 (partial)
    )

    new_ids, keep_mask, new_pv, new_shape, new_grids = _trim_multimodal_to_match(input_ids, pv, shape, grids)

    assert keep_mask is not None
    new_img_tokens = sum(1 for t in new_ids if t == IMG)
    assert new_img_tokens == 4 * TOKENS_PER_IMAGE
    assert len(new_grids) == 4
    assert len(new_ids) == len(input_ids) - partial


def test_trim_multimodal_drops_all_images():
    """When no complete image fits, drop all pixel data."""
    pv, shape, grids = _make_pixel_values(1)
    # Only 100 of the 768 image tokens present
    input_ids = [100] * 50 + [IMG] * 100

    new_ids, keep_mask, new_pv, new_shape, new_grids = _trim_multimodal_to_match(input_ids, pv, shape, grids)

    assert new_pv is None
    assert new_grids is None
    assert sum(1 for t in new_ids if t == IMG) == 0
    assert len(new_ids) == 50  # only text tokens remain


def test_prepare_sample_vlm_seq_len_truncation():
    """prepare_sample truncates to seq_len and trims multimodal data to match."""
    sample = _make_vlm_sample(num_images=5, text_tokens_per_image=200, extra_completion=500)
    total = len(sample.prompt_ids) + len(sample.completion_ids)
    assert total > 4096

    mb = prepare_sample(sample, seq_len=4096)

    img_tokens = sum(1 for t in mb.input_ids if t == IMG)
    features = sum(g[0] * g[1] * g[2] // 4 for g in mb.image_grid_thw)
    assert img_tokens == features
    assert len(mb.input_ids) == len(mb.loss_mask) == len(mb.advantages)


def test_prepare_sample_vlm_vllm_truncation():
    """Simulate vLLM left-truncation: total == seq_len but fewer image tokens than grids."""
    # This is the actual bug: vLLM truncates prompt to max_model_len, losing some
    # image_pad tokens, but pixel_values still has all images.
    pv, shape, grids = _make_pixel_values(5)
    # Build prompt_ids that simulate vLLM having already truncated:
    # 4 complete images + 516 partial + some text, totaling exactly 4096
    partial = 516
    text_per_gap = 50
    prompt_ids = (
        [100] * text_per_gap
        + [IMG] * partial  # left-truncated 5th image (only partial remains at start)
        + [100] * text_per_gap
        + [IMG] * TOKENS_PER_IMAGE
        + [100] * text_per_gap
        + [IMG] * TOKENS_PER_IMAGE
        + [100] * text_per_gap
        + [IMG] * TOKENS_PER_IMAGE
        + [100] * text_per_gap
    )
    target = 4096
    completion_len = target - len(prompt_ids)
    completion_ids = [101] * completion_len

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
