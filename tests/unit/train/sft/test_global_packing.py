import copy
import random
import time

import pytest
import torch
from datasets import Dataset

from prime_rl.configs.sft import GlobalPackingConfig, LossMaskConfig, SFTConfig, SFTDataConfig
from prime_rl.trainer.sft.broker import (
    GlobalDataLoader,
    GlobalPackedDataset,
    compatibility_key,
    materialize_rows,
    prepare_sample,
)
from prime_rl.trainer.sft.data import CatDataset, SFTDataset
from prime_rl.trainer.sft.packing import OnlinePacker, SampleDescriptor


def descriptors(lengths, **kwargs):
    return [SampleDescriptor((0, index), length, **kwargs) for index, length in enumerate(lengths)]


def sample(length, offset=0):
    return dict(
        input_ids=list(range(offset, offset + length)),
        target_ids=list(range(offset + 1, offset + length + 1)),
        position_ids=list(range(length)),
        loss_mask=[index % 3 != 0 for index in range(length)],
        seq_lens=[length],
        mm_kwargs=None,
        mm_token_type_ids=None,
    )


@pytest.mark.parametrize(
    "lengths,expected,pending",
    [
        ([7, 6, 3, 4], [[0, 2], [1, 3]], []),
        ([6, 6, 6, 4], [[0], [1]], [2]),
        ([5, 5, 5, 5, 1], [[0, 1], [2, 3]], []),
        ([10, 10, 1], [[0], [1]], []),
        ([5, 6, 4], [[0], [1, 2]], []),
        ([3], [[0], []], []),
        ([], None, []),
    ],
)
def test_online_best_fit(lengths, expected, pending):
    source = iter(descriptors(lengths))
    packer = OnlinePacker(source, capacity=10, num_rows=2)
    rows = packer.next_step()
    actual = None if rows is None else [[item.sample_id[1] for item in row] for row in rows]
    assert actual == expected
    assert [item.sample_id[1] for item in packer.pending] == pending
    if lengths == [10, 10, 1]:
        assert next(source).sample_id == (0, 2)


def test_best_fit_tie_and_updated_capacity():
    packer = OnlinePacker(iter(descriptors([5, 10, 2, 1, 4, 4])), capacity=10, num_rows=4)
    rows = packer.next_step()
    assert [[item.sample_id[1] for item in row] for row in rows] == [[0, 2, 3], [1], [4, 5], []]
    tied = OnlinePacker(iter(descriptors([6, 6, 4])), capacity=10, num_rows=2)
    assert [[item.sample_id[1] for item in row] for row in tied.next_step()] == [[0, 2], [1]]


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("lookahead", [0, 1, 4])
def test_no_loss_or_duplicates_and_prefix(seed, lookahead):
    generator = random.Random(seed)
    inputs = descriptors([generator.randint(1, 10) for _ in range(100)])
    packer = OnlinePacker(iter(inputs), capacity=10, num_rows=6, dp_size=2, lookahead_samples=lookahead)
    seen = []
    while (rows := packer.next_step()) is not None:
        selected = sorted(item.sample_id[1] for row in rows for item in row)
        if lookahead == 0:
            assert selected == list(range(len(seen), len(seen) + len(selected)))
        seen.extend(selected)
        assert all(sum(item.length for item in row) <= 10 for row in rows)
    assert sorted(seen) == list(range(100))


def test_lookahead_pending_first_and_restart():
    inputs = descriptors([6, 6, 7, 8, 4, 3, 2])
    source = iter(inputs)
    packer = OnlinePacker(source, 10, 2, lookahead_samples=3)
    first = packer.next_step()
    assert [[item.sample_id[1] for item in row] for row in first] == [[0, 4], [1, 5]]
    assert [item.sample_id[1] for item in packer.pending] == [2, 3]
    assert packer.metrics["skipped_ahead_samples"] == 2
    state = copy.deepcopy(packer.state_dict())
    remaining = list(source)
    packer.source = iter(remaining)
    resumed = OnlinePacker(iter(remaining), 10, 2, lookahead_samples=3)
    resumed.load_state_dict(state)
    assert resumed.next_step() == packer.next_step()
    assert [item.sample_id[1] for row in resumed.next_step() or [] for item in row] == []


def test_pending_bytes_and_invalid_lengths():
    inputs = descriptors([6, 6, 7, 8, 4], num_bytes=5)
    packer = OnlinePacker(iter(inputs), 10, 2, lookahead_samples=4, max_pending_bytes=5, max_sample_bytes=5)
    packer.next_step()
    assert [item.sample_id[1] for item in packer.pending] == [2]
    assert packer.metrics["pending_bytes"] == 5
    for length in [0, 11]:
        with pytest.raises(ValueError, match="unresolved length"):
            OnlinePacker(iter(descriptors([length])), 10, 2).next_step()
    with pytest.raises(ValueError, match="max_sample_bytes"):
        OnlinePacker(iter(descriptors([4], num_bytes=6)), 10, 2, max_sample_bytes=5).next_step()


def test_global_packing_config_defaults_and_guards():
    assert SFTConfig().data.global_packing is None
    config = SFTConfig.model_validate({"data": {"global_packing": {}}, "val": {"data": {}}})
    assert config.data.global_packing.lookahead_samples == 0
    assert config.val.data.global_packing == config.data.global_packing
    for kwargs in ({"lookahead_samples": -1}, {"lookahead_samples": 128}, {"max_pending_bytes": 1}):
        with pytest.raises(ValueError):
            GlobalPackingConfig(**kwargs)
    with pytest.raises(ValueError, match="text-only"):
        SFTConfig.model_validate(
            {
                "data": {"global_packing": {}},
                "model": {
                    "name": "Qwen/Qwen3.5-2B",
                    "impl": "custom",
                    "vlm": {"vision_encoder_attr": "model.visual", "language_model_attr": "model.language_model"},
                },
            }
        )


def test_text_truncation_matches_local_packing():
    original = sample(17)
    truncated = prepare_sample(original, 10)
    expected = next(iter(CatDataset([original], 10)))
    for key in ("input_ids", "target_ids", "position_ids", "loss_mask", "seq_lens"):
        assert truncated[key] == expected[key]
    assert len(original["input_ids"]) == 17


def test_materialization_preserves_samples_and_alignment():
    inputs = descriptors([7, 6, 3, 4])
    samples = {item.sample_id: sample(item.length, item.sample_id[1] * 20) for item in inputs}
    buckets = OnlinePacker(iter(inputs), 10, 4, dp_size=2).next_step()
    lanes = materialize_rows(buckets, samples, 10, 2)
    for lane in lanes:
        assert len(lane) == 2
        for row in lane:
            assert row["input_ids"].shape == (1, 10)
            assert row["seq_lens"].sum() == 10
            offset = 0
            for sample_id in row["sample_ids"]:
                original = samples[sample_id]
                length = len(original["input_ids"])
                for key in ("input_ids", "target_ids", "position_ids", "loss_mask"):
                    assert row[key][0, offset : offset + length].tolist() == original[key]
                offset += length
            assert not row["loss_mask"][0, offset:].any()
    assert sum(row["num_tokens"] for lane in lanes for row in lane) == 20


def test_multimodal_compatibility_and_zero_loss_alignment():
    samples = {(0, 0): sample(5), (0, 1): sample(6), (0, 2): sample(7)}
    for sample_id, shape in [((0, 1), (2, 4)), ((0, 2), (3, 5))]:
        samples[sample_id]["mm_kwargs"] = {"pixel_values": torch.ones(shape)}
        samples[sample_id]["mm_token_type_ids"] = [1] * len(samples[sample_id]["input_ids"])
    inputs = [
        SampleDescriptor(sample_id, len(value["input_ids"]), compatibility_key(value))
        for sample_id, value in samples.items()
    ]
    packer = OnlinePacker(iter(inputs), 10, 6, dp_size=2)
    lanes = materialize_rows(packer.next_step(), samples, 10, 2)
    for micro_step in range(3):
        left, right = (lane[micro_step] for lane in lanes)
        assert compatibility_key(left) == compatibility_key(right)
        alignment = next(row for row in (left, right) if not row["sample_ids"])
        assert alignment["num_tokens"] == 0
        assert not alignment["loss_mask"].any()
        if left["mm_kwargs"] is not None:
            torch.testing.assert_close(left["mm_kwargs"]["pixel_values"], right["mm_kwargs"]["pixel_values"])


class DelayedRenderer:
    def __init__(self, renderer):
        self.renderer = renderer

    def __call__(self, example):
        time.sleep((8 - len(example["completion"])) * 0.0005)
        return self.renderer


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("lookahead", [0, 2])
def test_ordered_rendering_finite_tail_and_epoch_resume(dummy_renderer, workers, lookahead):
    raw = Dataset.from_dict({"prompt": [""] * 5, "completion": ["abcdef", "abcde", "ab", "abcd", "abcdefg"]})
    config = SFTDataConfig(
        seq_len=10, batch_size=4, num_workers=workers, global_packing=GlobalPackingConfig(lookahead_samples=lookahead)
    )

    def make_dataset():
        dataset = SFTDataset(raw, DelayedRenderer(dummy_renderer), seq_len=10, shuffle=True, seed=3, max_epochs=3)
        return GlobalPackedDataset(dataset, config, dp_size=2)

    uninterrupted = make_dataset()
    iterator = iter(uninterrupted)
    first = next(iterator)
    state = copy.deepcopy(uninterrupted.state_dict())
    expected = list(iterator)
    resumed = make_dataset()
    resumed.load_state_dict(state)
    actual = list(resumed)
    assert len(actual) == len(expected)
    for actual_step, expected_step in zip(actual, expected, strict=True):
        for actual_lane, expected_lane in zip(actual_step["lanes"], expected_step["lanes"], strict=True):
            for actual_row, expected_row in zip(actual_lane, expected_lane, strict=True):
                assert actual_row["sample_ids"] == expected_row["sample_ids"]
                for key in ("input_ids", "target_ids", "position_ids", "loss_mask", "seq_lens"):
                    torch.testing.assert_close(actual_row[key], expected_row[key])
    seen = [
        sample_id
        for step in [first, *expected]
        for lane in step["lanes"]
        for row in lane
        for sample_id in row["sample_ids"]
    ]
    assert sorted(seen) == [(epoch, index) for epoch in range(3) for index in range(5)]


def test_filtered_rows_are_counted(dummy_renderer):
    raw = Dataset.from_dict({"prompt": [""], "completion": ["abcdef"]})
    dataset = SFTDataset(
        raw,
        DelayedRenderer(dummy_renderer),
        seq_len=1,
        shuffle=False,
        max_epochs=1,
        loss_mask_config=LossMaskConfig(assistant=False),
    )
    config = SFTDataConfig(seq_len=1, batch_size=2, global_packing=GlobalPackingConfig())
    broker = GlobalPackedDataset(dataset, config, dp_size=2)
    assert list(broker) == []
    assert broker.filtered == 1


def test_prefetched_loader_checkpoint_replays_exactly(dummy_renderer):
    raw = Dataset.from_dict({"prompt": [""] * 5, "completion": ["abcdef", "abcde", "ab", "abcd", "abcdefg"]})
    config = SFTDataConfig(
        seq_len=10, batch_size=2, num_workers=2, global_packing=GlobalPackingConfig(lookahead_samples=2)
    )

    def make_loader():
        dataset = SFTDataset(raw, DelayedRenderer(dummy_renderer), seq_len=10, shuffle=True, seed=9, max_epochs=4)
        return GlobalDataLoader(GlobalPackedDataset(dataset, config, dp_size=1), config, cp_size=1)

    loader = make_loader()
    next(loader)
    with pytest.raises(RuntimeError, match="completed optimizer step"):
        loader.state_dict()
    next(loader)
    state = copy.deepcopy(loader.state_dict())
    expected = list(loader)
    loader.close()
    resumed = make_loader()
    resumed.load_state_dict(state)
    actual = list(resumed)
    resumed.close()
    assert [row["sample_ids"] for row in actual] == [row["sample_ids"] for row in expected]
    for actual_row, expected_row in zip(actual, expected, strict=True):
        for key in ("input_ids", "target_ids", "position_ids", "loss_mask", "seq_lens"):
            torch.testing.assert_close(actual_row[key], expected_row[key])


def test_packed_attention_loss_and_gradients_match_independent_samples():
    torch.manual_seed(0)
    inputs = descriptors([7, 6, 3, 4])
    samples = {item.sample_id: sample(item.length, item.sample_id[1] * 10) for item in inputs}
    buckets = OnlinePacker(iter(inputs), 10, 6, dp_size=2).next_step()
    lanes = materialize_rows(buckets, samples, 10, 2)
    embedding = torch.randn(64, 8, dtype=torch.float64, requires_grad=True)
    projection = torch.randn(8, 64, dtype=torch.float64, requires_grad=True)

    def loss(row):
        inputs = torch.as_tensor(row["input_ids"]).flatten()
        targets = torch.as_tensor(row["target_ids"]).flatten()
        positions = torch.as_tensor(row["position_ids"]).flatten()
        mask = torch.as_tensor(row["loss_mask"]).flatten()
        hidden = embedding[inputs] + positions[:, None] * 0.01
        attention_mask = torch.block_diag(
            *[torch.ones(int(length), int(length), dtype=torch.bool).tril() for length in row["seq_lens"]]
        )
        scores = hidden @ hidden.T / 8**0.5
        attention = scores.masked_fill(~attention_mask, -torch.inf).softmax(dim=-1)
        logits = (attention @ hidden) @ projection
        return torch.nn.functional.cross_entropy(logits, targets, reduction="none")[mask].sum()

    count = sum(sum(row["loss_mask"]) for row in samples.values())
    reference = sum(loss(row) for row in samples.values()) / count
    expected_grads = torch.autograd.grad(reference, (embedding, projection))
    packed = sum(loss(row) for lane in lanes for row in lane) / count
    actual_grads = torch.autograd.grad(packed, (embedding, projection))
    torch.testing.assert_close(packed, reference)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)
