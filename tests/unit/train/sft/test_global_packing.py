import copy
import random

import pytest
import torch
from datasets import Dataset

from prime_rl.configs.sft import LossMaskConfig, PackingConfig, SFTConfig, SFTDataConfig
from prime_rl.trainer.sft.data import CatDataset, SFTDataset, cat_collate
from prime_rl.trainer.sft.data.broker import PackedDataLoader, materialize_rows
from prime_rl.trainer.sft.data.packing import OnlinePacker, SampleDescriptor, schedule_rows


class RendererFactory:
    def __init__(self, renderer):
        self.renderer = renderer

    def __call__(self, example):
        return self.renderer


def test_best_fit_consumes_contiguous_prefixes():
    for lengths, expected in [([7, 6, 3, 4], [[0, 2], [1, 3]]), ([6, 6, 7, 4], [[0], [1]])]:
        packer = OnlinePacker(10, 2)
        for position, length in enumerate(lengths):
            if not packer.add(SampleDescriptor(position, length)):
                break
        assert [[sample.position for sample in row] for row in packer.rows] == expected
    for seed in range(10):
        generator = random.Random(seed)
        lengths = [generator.randint(1, 10) for _ in range(100)]
        position = 0
        while position < len(lengths):
            start = position
            packer = OnlinePacker(10, 4)
            while position < len(lengths) and not packer.full:
                if not packer.add(SampleDescriptor(position, lengths[position])):
                    break
                position += 1
            assert sorted(sample.position for row in packer.rows for sample in row) == list(range(start, position))
            assert all(sum(sample.length for sample in row) <= 10 for row in packer.rows)
    with pytest.raises(ValueError):
        OnlinePacker(10, 2).add(SampleDescriptor(0, 11))


def test_rows_are_sorted_across_all_microsteps():
    rows = [[SampleDescriptor(position, length)] for position, length in enumerate([9, 1, 7, 3, 8, 2])]
    scheduled = schedule_rows(rows, 2)
    costs = [sum(sample.length**2 for sample in row) for microstep in zip(*scheduled) for row in microstep]
    assert costs == sorted(costs)
    assert [[row[0].position for row in rank] for rank in scheduled] == [[1, 3, 4], [5, 2, 0]]


def test_tensor_materialization_matches_sample_fields():
    samples = {}
    reference = []
    for position, length in enumerate([7, 3]):
        sample = dict(
            input_ids=list(range(position * 20, position * 20 + length)),
            target_ids=list(range(position * 20 + 1, position * 20 + length + 1)),
            position_ids=list(range(length)),
            loss_mask=[index % 3 != 0 for index in range(length)],
            seq_lens=[length],
            mm_kwargs=None,
            mm_token_type_ids=None,
        )
        reference.append(sample)
        samples[position] = torch.tensor(
            [sample[key] for key in ("input_ids", "target_ids", "position_ids", "loss_mask")]
        )
    descriptors = [SampleDescriptor(0, 7), SampleDescriptor(1, 3)]
    real, empty = materialize_rows([descriptors, []], samples, 16)
    expected = cat_collate([next(iter(CatDataset(reference, 16)))])
    for key in ("input_ids", "target_ids", "position_ids", "loss_mask", "seq_lens"):
        torch.testing.assert_close(real[key], expected[key], rtol=0, atol=0)
    assert real["num_tokens"] == 10
    assert real["sample_ids"] == [0, 1]
    assert empty["sample_ids"] == [] and empty["num_tokens"] == 0
    assert not empty["loss_mask"].any() and empty["seq_lens"].tolist() == [16]


@pytest.mark.parametrize("workers,chunk_size", [(1, 1), (2, 3)])
def test_prefetch_resume_is_cursor_only_and_chunk_independent(dummy_renderer, workers, chunk_size):
    raw = Dataset.from_dict({"prompt": [""] * 9, "completion": ["a" * size for size in [1, 6, 10, 2, 19, 3, 9, 7, 15]]})
    config = SFTDataConfig(
        seq_len=8, micro_batch_size=2, batch_size=8, num_workers=workers, packing=PackingConfig(chunk_size=chunk_size)
    )

    def make_loader(settings):
        dataset = SFTDataset(raw, RendererFactory(dummy_renderer), seq_len=8, shuffle=True, seed=9, max_epochs=3)
        return PackedDataLoader(dataset, settings)

    loader = make_loader(config)
    try:
        prefix = [next(loader)]
        with pytest.raises(RuntimeError, match="completed optimizer step"):
            loader.state_dict()
        prefix.extend(next(loader) for _ in range(3))
        state = copy.deepcopy(loader.state_dict())
        loader.future.result()
        assert state == loader.state_dict()
        assert set(state) == {"signature", "progress"}
        expected = list(loader)
    finally:
        loader.close()
    changed = config.model_copy(update={"num_workers": 1, "packing": PackingConfig(chunk_size=5)})
    resumed = make_loader(changed)
    try:
        resumed.load_state_dict(state)
        actual = list(resumed)
    finally:
        resumed.close()
    assert [row["sample_ids"] for row in actual] == [row["sample_ids"] for row in expected]
    for actual_row, expected_row in zip(actual, expected, strict=True):
        for key in ("input_ids", "target_ids", "position_ids", "loss_mask", "seq_lens"):
            torch.testing.assert_close(actual_row[key], expected_row[key], rtol=0, atol=0)
    assert sorted(position for row in prefix + actual for position in row["sample_ids"]) == list(range(27))


def test_filtered_epoch_and_empty_validation(dummy_renderer):
    raw = Dataset.from_dict({"prompt": [""], "completion": ["abc"]})
    config = SFTDataConfig(seq_len=8, batch_size=2)
    for finite in [True, False]:
        dataset = SFTDataset(
            raw,
            RendererFactory(dummy_renderer),
            seq_len=8,
            max_epochs=1 if finite else None,
            loss_mask_config=LossMaskConfig(assistant=False),
        )
        loader = PackedDataLoader(dataset, config)
        if finite:
            try:
                assert list(loader) == []
                assert loader.dataset_progress["step"] == 1
            finally:
                loader.close()
        else:
            with pytest.raises(ValueError, match="no trainable samples"):
                next(loader)
            with pytest.raises(ValueError, match="no trainable samples"):
                loader.close()


def test_packing_is_required_and_has_no_lookahead():
    assert SFTConfig().data.packing == PackingConfig()
    for values in [{"packing": None}, {"global_packing": {}}, {"packing": {"lookahead_samples": 1}}]:
        with pytest.raises(ValueError):
            SFTDataConfig.model_validate(values)
    with pytest.raises(ValueError):
        PackingConfig(chunk_size=0)
