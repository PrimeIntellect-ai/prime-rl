from types import SimpleNamespace

import numpy as np
import pytest

from prime_rl.trainer.batch import pad_micro_batch, prepare_batch, prepare_sample
from prime_rl.trainer.utils import build_bin_cost
from prime_rl.transport.types import EncodedTensor, MicroBatch, RoutedExperts, TrainingSample


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


@pytest.fixture
def make_training_example():
    def _make_training_example(
        temperature: float = 1.0,
        ce_weights: list[float] | None = None,
        rl_weights: list[float] | None = None,
        env_name: str = "test-env",
    ) -> TrainingSample:
        return TrainingSample(
            token_ids=[1, 2, 3, 4],
            mask=[False, False, True, True],
            logprobs=[0.0, 0.0, -0.1, -0.2],
            temperatures=[temperature, temperature, temperature, temperature],
            advantages=[0.0, 0.0, 1.0, 1.0],
            env_name=env_name,
            ce_weights=ce_weights,
            rl_weights=rl_weights,
        )

    return _make_training_example


def make_sized_training_example(length: int, env_name: str = "test-env") -> TrainingSample:
    assert length >= 1
    prompt_len = length - 1
    return TrainingSample(
        token_ids=[1] * prompt_len + [2],
        mask=[False] * prompt_len + [True],
        logprobs=[0.0] * prompt_len + [-0.1],
        temperatures=[1.0] * length,
        advantages=[0.0] * prompt_len + [1.0],
        env_name=env_name,
    )


def _flatten_batches(batches_per_gpu):
    return [batch for worker_batches in batches_per_gpu for batch in worker_batches]


def _worker_token_sums(batches_per_gpu) -> list[int]:
    return [sum(len(batch.input_ids) for batch in worker_batches) for worker_batches in batches_per_gpu]


def _has_loss_tokens(batch: MicroBatch) -> bool:
    return any(batch.loss_mask)


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
            temperatures=[1.0, 1.0, 1.0, 1.0],
            advantages=[0.0, 0.0, 1.0, 1.0],
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
        bin_cost=build_bin_cost(None),
    )

    assert all(len(worker_batches) == expected_batches_per_worker for worker_batches in batches_per_gpu)

    flat_batches = _flatten_batches(batches_per_gpu)
    assert len(examples) <= len(flat_batches) < len(examples) + num_train_workers

    # Identify real vs padding batches by content, not position — the packer
    # distributes by workload, so a dummy can land anywhere in the order.
    real_batches = [batch for batch in flat_batches if _has_loss_tokens(batch)]
    dummy_batches = [batch for batch in flat_batches if not _has_loss_tokens(batch)]
    assert len(real_batches) == len(examples)

    # Verify real rollouts have expected non-zero advantages and loss mask
    # (the advantage stream is 0.0 on prompt positions, the scalar on completion)
    for batch in real_batches:
        assert sum(1 for advantage in batch.advantages if advantage != 0.0) == 2
        assert sum(1 for loss_mask in batch.loss_mask if loss_mask) == 2

    # Verify padded batches have zero advantages and loss mask
    for batch in dummy_batches:
        assert sum(1 for advantage in batch.advantages if advantage != 0.0) == 0
        assert sum(1 for loss_mask in batch.loss_mask if loss_mask) == 0


def test_randomized_packing_invariants():
    rng = np.random.default_rng(0)

    for case_idx in range(80):
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
        dummy_batches = [batch for batch in flat_batches if not _has_loss_tokens(batch)]

        assert all(len(worker_batches) == len(batches_per_gpu[0]) for worker_batches in batches_per_gpu)
        assert sorted(length for batch in real_batches for length in batch.sequence_lengths) == sorted(lengths)

        for batch in flat_batches:
            assert len(batch.input_ids) <= seq_len
            assert sum(batch.sequence_lengths) == len(batch.input_ids)
            assert batch.seq_lens == batch.sequence_lengths
            assert sum(batch.lora_num_tokens) == len(batch.input_ids)
            assert len(batch.env_names) == len(batch.input_ids)

        for batch in dummy_batches:
            assert not any(batch.loss_mask)
            assert not any(batch.advantages)


def test_pad_micro_batch_preserves_explicit_sequence_lengths():
    micro_batch = prepare_sample(make_sized_training_example(4), seq_len=16)

    padded = pad_micro_batch(micro_batch, pad_to_multiple_of=6)

    assert len(padded.input_ids) == 6
    assert padded.sequence_lengths == [6]
    assert padded.seq_lens == [6]
    assert sum(padded.sequence_lengths) == len(padded.input_ids)
    assert padded.loss_mask[-2:] == [False, False]


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


def test_flop_aware_split_to_align_splits_heaviest_flop_bin():
    examples = [make_sized_training_example(length) for length in [20, 18, 9, 9, 8, 8, 8]]

    batches_per_gpu = prepare_batch(
        rollouts=examples,
        seq_len=64,
        num_train_workers=4,
        idxs=[0] * len(examples),
        num_loras=1,
        bin_cost=build_bin_cost(make_flops_config()),
    )

    real_batches = [batch for batch in _flatten_batches(batches_per_gpu) if _has_loss_tokens(batch)]
    assert len(real_batches) == 4
    assert sorted(length for batch in real_batches for length in batch.sequence_lengths) == [8, 8, 8, 9, 9, 18, 20]
    assert sum(len(batch.sequence_lengths) > 1 for batch in real_batches) == 3


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
        bin_cost=build_bin_cost(None),
    )

    flat_batches = _flatten_batches(batches_per_gpu)
    # With per-token temperatures, samples can now be packed together
    assert len(flat_batches) == 1
    # Each sample has 4 tokens, so 8 total tokens
    assert len(flat_batches[0].temperatures) == 8
    # First sample (4 tokens): all get temp 0.7
    assert flat_batches[0].temperatures[:4] == [0.7, 0.7, 0.7, 0.7]
    # Second sample (4 tokens): all get temp 1.1
    assert flat_batches[0].temperatures[4:8] == [1.1, 1.1, 1.1, 1.1]
    assert flat_batches[0].env_names == ["env-a"] * 4 + ["env-b"] * 4
    assert flat_batches[0].seq_lens == [4, 4]


def test_prepare_sample_propagates_weight_streams(make_training_example):
    example = make_training_example(ce_weights=[0.0, 0.0, 1.0, 1.0], rl_weights=[0.0, 0.0, 0.0, 0.0])

    micro_batch = prepare_sample(example, seq_len=16)

    assert micro_batch.ce_weights == [0.0, 0.0, 1.0, 1.0]
    assert micro_batch.rl_weights == [0.0, 0.0, 0.0, 0.0]


def test_prepare_sample_uniform_rl_keeps_streams_none(make_training_example):
    micro_batch = prepare_sample(make_training_example(), seq_len=16)

    assert micro_batch.rl_weights is None
    assert micro_batch.ce_weights is None
    assert micro_batch.ref_kl_weights is None


@pytest.mark.parametrize("streams_on_longer", [True, False])
def test_prepare_batch_packs_mixed_components(make_training_example, streams_on_longer):
    """Component membership is per token, so samples feeding different
    components pack together. The stream-less sample's positions must backfill
    with the stream defaults (rl 1.0, ce 0.0) on whichever side of the pack
    boundary it lands — a wrong-side backfill silently reroutes tokens between
    components while keeping every array length-aligned."""
    longer = TrainingSample(
        token_ids=[1, 2, 3, 4, 5, 6],
        mask=[False, False, False, True, True, True],
        logprobs=[0.0, 0.0, 0.0, -0.1, -0.1, -0.1],
        temperatures=[1.0] * 6,
        advantages=[0.0] * 3 + [1.0] * 3,
        env_name="test-env",
        ce_weights=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0] if streams_on_longer else None,
        rl_weights=[0.0] * 6 if streams_on_longer else None,
    )
    shorter = make_training_example(
        ce_weights=None if streams_on_longer else [0.0, 0.0, 1.0, 1.0],
        rl_weights=None if streams_on_longer else [0.0, 0.0, 0.0, 0.0],
    )

    batches_per_gpu = prepare_batch(
        rollouts=[longer, shorter],
        seq_len=16,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )

    flat_batches = _flatten_batches(batches_per_gpu)
    assert len(flat_batches) == 1
    batch = flat_batches[0]
    # FFD places the longer sample first; every stream value must sit at its
    # sample's offset, with the stream-less side backfilled.
    if streams_on_longer:
        assert batch.rl_weights == [0.0] * 6 + [1.0] * 4
        assert batch.ce_weights == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 4
    else:
        assert batch.rl_weights == [1.0] * 6 + [0.0] * 4
        assert batch.ce_weights == [0.0] * 6 + [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("refs_on_longer", [True, False])
def test_prepare_batch_aligns_ref_logprobs_in_mixed_bins(make_training_example, refs_on_longer):
    """Packing a ref-bearing sample (e.g. OPD) with a ref-less one (e.g. GRPO)
    must keep ``ref_logprobs`` position-aligned with ``input_ids`` — placeholder
    0.0s on the ref-less tokens, both when the bin gains refs after ref-less
    content (backfill) and when ref-less content lands in a ref-bearing bin."""
    longer = TrainingSample(
        token_ids=[1, 2, 3, 4, 5, 6],
        mask=[False, False, False, True, True, True],
        logprobs=[0.0, 0.0, 0.0, -0.1, -0.1, -0.1],
        temperatures=[1.0] * 6,
        ref_logprobs=[-1.5] * 6 if refs_on_longer else None,
        advantages=[0.0] * 3 + [1.0] * 3,
        env_name="test-env",
    )
    shorter = make_training_example()
    shorter.ref_logprobs = None if refs_on_longer else [-1.5] * 4

    batches_per_gpu = prepare_batch(
        rollouts=[longer, shorter],
        seq_len=16,
        pad_to_multiple_of=1,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )
    flat_batches = _flatten_batches(batches_per_gpu)
    assert len(flat_batches) == 1  # both samples share one bin
    bin_content = flat_batches[0]
    assert len(bin_content.ref_logprobs) == len(bin_content.input_ids)
    # FFD places the longer sample first; refs must sit at their sample's offset
    if refs_on_longer:
        assert bin_content.ref_logprobs == [-1.5] * 6 + [0.0] * 4
    else:
        assert bin_content.ref_logprobs == [0.0] * 6 + [-1.5] * 4


def test_prepare_sample_with_routed_experts():
    """Routed experts are passed through prepare_sample and match input_ids length."""
    # 4 tokens, 2 layers, topk=2
    routed_experts = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 2], [1, 3]], [[1, 0], [3, 2]]]
    routed_payload = _routed_experts(routed_experts)
    sample = TrainingSample(
        token_ids=[1, 2, 3, 4],
        mask=[False, False, True, True],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0, 1.0],
        advantages=[0.0, 0.0, 1.0, 1.0],
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
        advantages=[0.0, 0.0, 1.0, 1.0],
        env_name="test-env",
        routed_experts=routed_payload,
    )

    micro_batch = prepare_sample(sample, seq_len=3)
    assert micro_batch.routed_experts is not None
    assert micro_batch.routed_experts == expected_payload
    assert micro_batch.env_names == ["test-env"] * 3


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
        advantages=[0.0] * 6 + [1.0],
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


def test_prepare_batch_packs_multimodal_with_text():
    mm_sample = TrainingSample(
        token_ids=[10, 11, 12],
        mask=[False, True, True],
        logprobs=[0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0],
        advantages=[0.0, 1.0, 1.0],
        env_name="mm-env",
        mm_token_type_ids=[0, 1, 0],
        mm_kwargs={
            "pixel_values": _encoded(np.array([[1.0, 2.0]], dtype=np.float32)),
            "image_grid_thw": _encoded(np.array([[1, 2, 2]], dtype=np.int64)),
        },
    )
    text_sample = TrainingSample(
        token_ids=[20, 21],
        mask=[False, True],
        logprobs=[0.0, -0.3],
        temperatures=[0.7, 0.7],
        advantages=[0.0, 1.0],
        env_name="text-env",
    )

    batches_per_gpu = prepare_batch(
        rollouts=[mm_sample, text_sample],
        seq_len=8,
        num_train_workers=2,
        idxs=[0, 0],
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )

    real_batches = [batch for batch in _flatten_batches(batches_per_gpu) if _has_loss_tokens(batch)]
    assert len(real_batches) == 1
    batch = real_batches[0]
    assert batch.seq_lens == [3, 2]
    assert batch.sequence_lengths == [3, 2]
    assert batch.position_ids == [0, 1, 2, 0, 1]
    assert batch.mm_token_type_ids == [0, 1, 0, 0, 0]
    assert batch.mm_kwargs is not None
    assert batch.mm_kwargs["pixel_values"].shape == [1, 2]
    assert batch.mm_kwargs["image_grid_thw"].shape == [1, 3]
    assert batch.env_names == ["mm-env"] * 3 + ["text-env"] * 2


def test_prepare_sample_none_routed_experts():
    """When routed_experts is None, micro_batch.routed_experts is None."""
    sample = TrainingSample(
        token_ids=[1, 2, 3, 4],
        mask=[False, False, True, True],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0, 1.0],
        advantages=[0.0, 0.0, 1.0, 1.0],
        env_name="test-env",
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is None
