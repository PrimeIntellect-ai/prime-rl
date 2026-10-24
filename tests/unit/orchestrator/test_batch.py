import numpy as np
import pytest

from prime_rl.orchestrator.batch import (
    _is_multimodal_sample,
    pad_micro_batch,
    prepare_batch,
    prepare_sample,
)
from prime_rl.transports.batch.types import MicroBatch, MMImageRef, MMRefs, RoutedExperts, TrainingSample, encode_array

SAMPLE_DTYPES = {
    "token_ids": np.int32,
    "mask": np.bool_,
    "logprobs": np.float32,
    "temperatures": np.float32,
    "advantages": np.float32,
    "ref_logprobs": np.float32,
    "mm_token_type_ids": np.int32,
    "rl_weights": np.float32,
    "ce_weights": np.float32,
    "ref_kl_weights": np.float32,
}


def _sample(**fields) -> TrainingSample:
    """A sample with typed per-token streams, as `trace_to_samples` builds it."""
    return TrainingSample(
        **{
            name: np.asarray(value, SAMPLE_DTYPES[name]) if name in SAMPLE_DTYPES and value is not None else value
            for name, value in fields.items()
        }
    )


def test_top_logprobs_survive_truncation_mixed_packing_and_padding(make_training_example):
    import msgspec

    from prime_rl.trainer.rl.data import DataLoader
    from prime_rl.transports.batch.types import SamplingMask, TopLogprobs

    sample = make_training_example()
    sample.top_logprobs = TopLogprobs(
        ids=np.array([3, 7, 4], np.int32).tobytes(),
        logprobs=np.array([-0.3, -1.8, -0.2], np.float32).tobytes(),
        counts=np.array([0, 0, 2, 1], np.int32).tobytes(),
    )
    truncated = prepare_sample(sample, seq_len=3)
    np.testing.assert_array_equal(np.frombuffer(truncated.top_logprobs.counts, np.int32), [0, 0, 2])
    np.testing.assert_array_equal(np.frombuffer(truncated.top_logprobs.ids, np.int32), [3, 7])
    replayed = make_training_example()
    replayed.sampling_mask = SamplingMask(
        ids=np.array([5, 6, 8], np.int32).tobytes(),
        counts=np.array([0, 0, 1, 2], np.int32).tobytes(),
        logprobs=np.log(np.array([1.0, 0.75, 0.25], np.float32)).tobytes(),
    )
    truncated = prepare_sample(replayed, seq_len=3)
    np.testing.assert_array_equal(np.frombuffer(truncated.sampling_mask.ids, np.int32), [5])
    np.testing.assert_array_equal(np.frombuffer(truncated.sampling_mask.logprobs, np.float32), [0.0])
    plain = make_training_example(rl_weights=[0.0] * 4)
    plain.sampling_mask = SamplingMask(
        ids=np.array([9], np.int32).tobytes(), counts=np.array([0, 0, 0, 1], np.int32).tobytes()
    )
    batches = prepare_batch(
        [sample, replayed, plain], seq_len=16, num_train_workers=1, bin_cost=sum, pad_to_multiple_of=16
    )
    [[packed]] = batches
    packed = msgspec.msgpack.decode(msgspec.msgpack.encode(packed, enc_hook=encode_array), type=MicroBatch)
    assert len(packed.top_logprobs.ids) == 3 * 4
    tensor = DataLoader._micro_batch_to_tensor(None, packed)
    ids, values = tensor["top_logprobs_ids"][0], tensor["top_logprobs_values"][0]
    assert ids.shape == values.shape
    assert ids.shape[0] == sum(packed.sequence_lengths)
    rows = {
        tuple(row[row >= 0].tolist()): value[row >= 0].tolist() for row, value in zip(ids, values) if (row >= 0).any()
    }
    expected = {(3, 7): [-0.3, -1.8], (4,): [-0.2], (5,): [0.0], (6, 8): np.log([0.75, 0.25]), (9,): [np.nan]}
    assert rows.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_allclose(rows[key], value, rtol=1e-6)
    # The source evidence is immutable under truncation and packing.
    np.testing.assert_array_equal(np.frombuffer(sample.top_logprobs.counts, np.int32), [0, 0, 2, 1])
    np.testing.assert_array_equal(np.frombuffer(replayed.sampling_mask.counts, np.int32), [0, 0, 1, 2])


@pytest.mark.parametrize("support", ["complete", "mismatch"])
def test_sampler_evidence_moves_masked_top_logprobs_onto_the_mask(support):
    import verifiers.v1 as vf

    from prime_rl.orchestrator.trajectories import _encode_sampler_evidence

    mask = vf.SamplingMask(ids=np.array([5, 6, 8], np.int32), counts=np.array([0, 0, 1, 2], np.int32))
    head = vf.TopLogprobs(
        ids=np.array([5, 8, 6], np.int32),
        logprobs=np.array([0.0, -0.3, -1.4], np.float32),
        counts=np.array([0, 0, 1, 2] if support == "complete" else [0, 1, 0, 2], np.int32),
    )
    if support == "mismatch":
        with pytest.raises(ValueError):
            _encode_sampler_evidence(mask, head, 4)
        return
    sampling_mask, top_logprobs = _encode_sampler_evidence(mask, head, 4)
    assert top_logprobs is None
    np.testing.assert_array_equal(np.frombuffer(sampling_mask.ids, np.int32), [5, 8, 6])
    np.testing.assert_array_equal(np.frombuffer(sampling_mask.counts, np.int32), [0, 0, 1, 2])
    np.testing.assert_array_equal(np.frombuffer(sampling_mask.logprobs, np.float32), np.float32([0.0, -0.3, -1.4]))


@pytest.mark.parametrize("corruption", [None, "missing", "duplicate", "vocab", "mass", "support_mass", "sampled"])
def test_top_logprobs_validate_only_rl_tokens_and_complete_support(corruption):
    import torch

    from prime_rl.trainer.rl.data import validate_score_centering_data

    batch = {
        "input_ids": torch.tensor([[0, 3, 6]]),
        "loss_mask": torch.tensor([[False, True, True]]),
        "rl_weights": torch.tensor([[0.0, 1.0, 0.0]]),
        "top_logprobs_ids": torch.tensor([[[-1, -1], [3, 7], [-1, -1]]]),
        "top_logprobs_values": torch.tensor([[[1.0, 1.0], [0.75, 0.25], [1.0, 1.0]]]).log(),
        "inference_logprobs": torch.tensor([[1.0, 0.75, 1.0]]).log(),
        "sampling_mask": torch.tensor([[[-1, -1], [-1, -1], [6, -1]]]),
    }
    validate_score_centering_data(batch, vocab_size=9)
    if corruption is None:
        batch["sampling_mask"][0, 1] = torch.tensor([7, 3])
        validate_score_centering_data(batch, vocab_size=9)
        return
    match corruption:
        case "missing":
            batch["top_logprobs_ids"] = None
        case "duplicate":
            batch["top_logprobs_ids"][0, 1, 1] = 3
        case "vocab":
            batch["top_logprobs_ids"][0, 1, 1] = 9
        case "mass":
            batch["top_logprobs_values"][0, 1] = 0
        case "support_mass":
            batch["sampling_mask"][0, 1] = torch.tensor([7, 3])
            batch["top_logprobs_values"][0, 1] = torch.tensor([0.5, 0.25]).log()
        case "sampled":
            batch["sampling_mask"][0, 1] = torch.tensor([7, 3])
            batch["input_ids"][0, 1] = 4
    with pytest.raises(ValueError):
        validate_score_centering_data(batch, vocab_size=9)


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
        ce_weights: list[float] | None = None,
        rl_weights: list[float] | None = None,
        env_name: str = "test-env",
    ) -> TrainingSample:
        return _sample(
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
    return _sample(
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


def flop_cost(seq_lens: list[int]) -> int:
    """A FlopCounter-shaped cost: a per-token term plus a per-document attention term."""
    return sum(64 * n + n * n for n in seq_lens)


def test_training_sample_requires_env_name():
    with pytest.raises(TypeError, match="env_name"):
        _sample(
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
        bin_cost=sum,
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


def test_prepare_batch_preserves_branch_identity(make_training_example):
    examples = []
    for i in range(5):
        example = make_training_example()
        example.trace_id = f"trace-{i}"
        example.branch_index = i
        examples.append(example)
    examples.append(make_training_example())  # no identity -> sentinels

    batches_per_gpu = prepare_batch(
        rollouts=examples,
        seq_len=4,
        num_train_workers=2,
        bin_cost=sum,
        pad_to_multiple_of=8,
    )

    seen: set[tuple[str, int]] = set()
    for batch in _flatten_batches(batches_per_gpu):
        if not _has_loss_tokens(batch):
            # Dummies are copies of real batches; identity must not survive the copy.
            assert batch.trace_ids is None and batch.branch_indices is None
            continue
        assert len(batch.trace_ids) == len(batch.branch_indices) == len(batch.sequence_lengths)
        seen.update(zip(batch.trace_ids, batch.branch_indices))
    assert seen == {(f"trace-{i}", i) for i in range(5)} | {("", -1)}


def test_prepare_sample_truncation_keeps_identity():
    sample = make_sized_training_example(10)
    sample.trace_id = "trace-x"
    sample.branch_index = 2
    micro_batch = prepare_sample(sample, seq_len=4)
    assert len(micro_batch.input_ids) == 4
    assert micro_batch.trace_ids == ["trace-x"]
    assert micro_batch.branch_indices == [2]


def test_randomized_packing_invariants():
    rng = np.random.default_rng(0)

    for case_idx in range(80):
        seq_len = int(rng.choice([8, 16, 32, 64]))
        num_train_workers = int(rng.choice([1, 2, 4, 8]))
        num_samples = int(rng.integers(1, 65))
        lengths = [int(x) for x in rng.integers(1, seq_len + 1, size=num_samples)]
        examples = [make_sized_training_example(length, env_name=f"env-{case_idx}") for length in lengths]
        bin_cost = flop_cost if case_idx % 2 == 0 else sum

        batches_per_gpu = prepare_batch(
            rollouts=examples,
            seq_len=seq_len,
            num_train_workers=num_train_workers,
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
            assert len(batch.env_names) == len(batch.sequence_lengths)

        for batch in dummy_batches:
            assert not batch.loss_mask.any()
            assert not batch.advantages.any()


def test_pad_micro_batch_preserves_explicit_sequence_lengths():
    micro_batch = prepare_sample(make_sized_training_example(4), seq_len=16)

    padded = pad_micro_batch(micro_batch, pad_to_multiple_of=6)

    assert len(padded.input_ids) == 6
    assert padded.sequence_lengths == [6]
    assert padded.seq_lens == [6]
    assert sum(padded.sequence_lengths) == len(padded.input_ids)
    assert padded.loss_mask[-2:].tolist() == [False, False]


def test_split_to_align_avoids_dummy_micro_batches():
    examples = [make_sized_training_example(length) for length in [6, 6, 5, 5, 4, 4]]

    batches_per_gpu = prepare_batch(
        rollouts=examples,
        seq_len=12,
        num_train_workers=4,
        bin_cost=sum,
    )

    assert all(_has_loss_tokens(batch) for batch in _flatten_batches(batches_per_gpu))
    assert len(_flatten_batches(batches_per_gpu)) == 4


def test_pack_first_then_balance_distributes_micro_batches_by_tokens_without_model_config():
    examples = [make_sized_training_example(length) for length in [100, 90, 80, 70]]

    balanced = prepare_batch(
        rollouts=examples,
        seq_len=100,
        num_train_workers=2,
        bin_cost=sum,
    )

    assert _worker_token_sums(balanced) == [170, 170]


def test_flop_aware_balancing_pairs_long_and_short_sequence_workloads():
    examples = [make_sized_training_example(length) for length in [32, 32, 16, 16, 16, 16]]
    bin_cost = flop_cost

    balanced = prepare_batch(
        rollouts=examples,
        seq_len=32,
        num_train_workers=2,
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
        bin_cost=flop_cost,
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
        bin_cost=sum,
    )

    flat_batches = _flatten_batches(batches_per_gpu)
    # With per-token temperatures, samples can now be packed together
    assert len(flat_batches) == 1
    # Each sample has 4 tokens, so 8 total tokens
    assert len(flat_batches[0].temperatures) == 8
    # First sample (4 tokens): all get temp 0.7
    assert flat_batches[0].temperatures[:4] == pytest.approx([0.7, 0.7, 0.7, 0.7])
    # Second sample (4 tokens): all get temp 1.1
    assert flat_batches[0].temperatures[4:8] == pytest.approx([1.1, 1.1, 1.1, 1.1])
    assert flat_batches[0].env_names == ["env-a", "env-b"]
    assert flat_batches[0].seq_lens == [4, 4]


def test_prepare_sample_propagates_weight_streams(make_training_example):
    example = make_training_example(ce_weights=[0.0, 0.0, 1.0, 1.0], rl_weights=[0.0, 0.0, 0.0, 0.0])

    micro_batch = prepare_sample(example, seq_len=16)

    assert micro_batch.ce_weights == pytest.approx([0.0, 0.0, 1.0, 1.0])
    assert micro_batch.rl_weights == pytest.approx([0.0, 0.0, 0.0, 0.0])


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
    longer = _sample(
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
        bin_cost=sum,
    )

    flat_batches = _flatten_batches(batches_per_gpu)
    assert len(flat_batches) == 1
    batch = flat_batches[0]
    # FFD places the longer sample first; every stream value must sit at its
    # sample's offset, with the stream-less side backfilled.
    if streams_on_longer:
        assert batch.rl_weights == pytest.approx([0.0] * 6 + [1.0] * 4)
        assert batch.ce_weights == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0, 1.0] + [0.0] * 4)
    else:
        assert batch.rl_weights == pytest.approx([1.0] * 6 + [0.0] * 4)
        assert batch.ce_weights == pytest.approx([0.0] * 6 + [0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize("refs_on_longer", [True, False])
def test_prepare_batch_aligns_ref_logprobs_in_mixed_bins(make_training_example, refs_on_longer):
    """Packing a ref-bearing sample (e.g. OPD) with a ref-less one (e.g. GRPO)
    must keep ``ref_logprobs`` position-aligned with ``input_ids`` — placeholder
    0.0s on the ref-less tokens, both when the bin gains refs after ref-less
    content (backfill) and when ref-less content lands in a ref-bearing bin."""
    longer = _sample(
        token_ids=[1, 2, 3, 4, 5, 6],
        mask=[False, False, False, True, True, True],
        logprobs=[0.0, 0.0, 0.0, -0.1, -0.1, -0.1],
        temperatures=[1.0] * 6,
        ref_logprobs=[-1.5] * 6 if refs_on_longer else None,
        advantages=[0.0] * 3 + [1.0] * 3,
        env_name="test-env",
    )
    shorter = make_training_example()
    shorter.ref_logprobs = None if refs_on_longer else np.full(4, -1.5, np.float32)

    batches_per_gpu = prepare_batch(
        rollouts=[longer, shorter],
        seq_len=16,
        pad_to_multiple_of=1,
        num_train_workers=1,
        bin_cost=sum,
    )
    flat_batches = _flatten_batches(batches_per_gpu)
    assert len(flat_batches) == 1  # both samples share one bin
    bin_content = flat_batches[0]
    assert len(bin_content.ref_logprobs) == len(bin_content.input_ids)
    # FFD places the longer sample first; refs must sit at their sample's offset
    if refs_on_longer:
        assert bin_content.ref_logprobs == pytest.approx([-1.5] * 6 + [0.0] * 4)
    else:
        assert bin_content.ref_logprobs == pytest.approx([0.0] * 6 + [-1.5] * 4)


def test_prepare_sample_with_routed_experts():
    """Routed experts are passed through prepare_sample and match input_ids length."""
    # 4 tokens, 2 layers, topk=2
    routed_experts = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 2], [1, 3]], [[1, 0], [3, 2]]]
    routed_payload = _routed_experts(routed_experts)
    sample = _sample(
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
    sample = _sample(
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
    assert micro_batch.env_names == ["test-env"]


def test_prepare_sample_truncates_mm_at_image_boundary():
    """Truncation never splits an image's expanded placeholder block."""
    mm_token_type_ids = [0, 1, 1, 0, 1, 1, 0]
    sample = _sample(
        token_ids=[10, 11, 12, 13, 14, 15, 16],
        mask=[False, False, False, False, False, True, True],
        logprobs=[0.0] * 7,
        temperatures=[1.0] * 7,
        advantages=[0.0] * 6 + [1.0],
        env_name="test-env",
        mm_token_type_ids=mm_token_type_ids,
        mm_refs=MMRefs(
            images=[
                MMImageRef(url="data:image/png;base64,first", offset=1, length=2),
                MMImageRef(url="data:image/png;base64,second", offset=4, length=2),
            ]
        ),
    )

    # seq_len=5 falls inside img1 (one of its two placeholders survives) -> drop img1 entirely.
    mb = prepare_sample(sample, seq_len=5)
    assert len(mb.input_ids) == 4  # cut back to img1's first placeholder (index 4)
    assert len(mb.mm_token_type_ids) == len(mb.input_ids)
    n_placeholders = sum(1 for t in mb.mm_token_type_ids if t)
    assert n_placeholders == 2  # only img0's two placeholders remain
    assert mb.mm_refs is not None
    assert [(ref.url, ref.offset, ref.length) for ref in mb.mm_refs.images] == [("data:image/png;base64,first", 1, 2)]


def test_prepare_batch_packs_multimodal_with_text():
    mm_sample = _sample(
        token_ids=[10, 11, 12],
        mask=[False, True, True],
        logprobs=[0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0],
        advantages=[0.0, 1.0, 1.0],
        env_name="mm-env",
        mm_token_type_ids=[0, 1, 0],
        mm_refs=MMRefs(images=[MMImageRef(url="data:image/png;base64,image", offset=1, length=1)]),
    )
    text_sample = _sample(
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
        num_train_workers=1,
        bin_cost=sum,
    )

    real_batches = [batch for batch in _flatten_batches(batches_per_gpu) if _has_loss_tokens(batch)]
    assert len(real_batches) == 1
    batch = real_batches[0]
    assert batch.seq_lens == [3, 2]
    assert batch.sequence_lengths == [3, 2]
    assert batch.position_ids.tolist() == [0, 1, 2, 0, 1]
    assert batch.mm_token_type_ids.tolist() == [0, 1, 0, 0, 0]
    assert batch.mm_refs is not None
    assert [(ref.offset, ref.length) for ref in batch.mm_refs.images] == [(1, 1)]
    assert batch.env_names == ["mm-env", "text-env"]


def test_split_to_align_splits_multimodal_bins():
    def make_mm_sample(token: int) -> TrainingSample:
        return _sample(
            token_ids=[token, token + 1, token + 2],
            mask=[False, True, True],
            logprobs=[0.0, -0.1, -0.2],
            temperatures=[1.0, 1.0, 1.0],
            advantages=[0.0, 1.0, 1.0],
            env_name=f"mm-{token}",
            mm_token_type_ids=[0, 1, 0],
            mm_refs=MMRefs(images=[MMImageRef(url=f"data:image/png;base64,{token}", offset=1, length=1)]),
        )

    batches_per_gpu = prepare_batch(
        rollouts=[make_mm_sample(10), make_mm_sample(20)],
        seq_len=8,
        num_train_workers=2,
        bin_cost=sum,
    )

    batches = _flatten_batches(batches_per_gpu)
    assert len(batches) == 2
    assert all(_has_loss_tokens(batch) and _is_multimodal_sample(batch) for batch in batches)


def test_prepare_sample_none_routed_experts():
    """When routed_experts is None, micro_batch.routed_experts is None."""
    sample = _sample(
        token_ids=[1, 2, 3, 4],
        mask=[False, False, True, True],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        temperatures=[1.0, 1.0, 1.0, 1.0],
        advantages=[0.0, 0.0, 1.0, 1.0],
        env_name="test-env",
    )

    micro_batch = prepare_sample(sample, seq_len=8)
    assert micro_batch.routed_experts is None
