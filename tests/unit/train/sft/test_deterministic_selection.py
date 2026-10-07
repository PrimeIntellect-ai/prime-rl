from collections import Counter

import pytest
from datasets import Dataset

from prime_rl.configs.sft import SFTDataConfig, deterministic_source_counts
from prime_rl.trainer.sft.data.dataset import load_sft_dataset
from prime_rl.trainer.sft.data.selection import DeterministicMixture


def sources(lengths):
    return [Dataset.from_dict({"source": [i] * length, "row": list(range(length))}) for i, length in enumerate(lengths)]


@pytest.mark.parametrize(
    "probabilities,counts",
    [
        (None, [1, 1]),
        ([0.999, 0.001], [999, 1]),
        ([0.6, 0.4], [3, 2]),
        ([1 / 3, 2 / 3], [1, 2]),
        ([1, 0], [1, 0]),
        ([0.5, 0.3, 0.2], [5, 3, 2]),
    ],
)
def test_exact_source_quotas(probabilities, counts):
    assert deterministic_source_counts(probabilities, len(counts)) == counts


@pytest.mark.parametrize(
    "probabilities",
    [[0, 0], [-0.1, 1.1], [float("nan"), 1], [float("inf"), 0], [0.9, 0.2], [0.9999999, 0.0000001], [1]],
)
def test_invalid_deterministic_probabilities(probabilities):
    with pytest.raises(ValueError):
        SFTDataConfig(splits=["train", "test"], probabilities=probabilities, deterministic_sampling=True)


def test_cycles_shuffle_and_random_access():
    raw = sources([4000, 20])
    mixture = DeterministicMixture(raw, [0.999, 0.001], "all_exhausted", seed=42)
    expected = [mixture[i] for i in range(3000)]
    for start in range(0, 3000, 1000):
        assert Counter(row["source"] for row in expected[start : start + 1000]) == {0: 999, 1: 1}
    assert len({i % 1000 for i, row in enumerate(expected) if row["source"] == 1}) > 1
    assert len({(row["source"], row["row"]) for row in expected}) == 3000
    # Reconstruct the global stream from independently created rank/worker shards.
    for workers in [2, 8, 32]:
        gathered = {}
        for rank in range(workers):
            shard = DeterministicMixture(raw, [0.999, 0.001], "all_exhausted", seed=42)
            gathered.update((i, shard[i]) for i in range(rank, 3000, workers))
        assert [gathered[i] for i in range(3000)] == expected
    shuffled = mixture.shuffle(seed=13, keep_in_memory=True)
    actual = [shuffled[i] for i in range(3000)]
    assert [row["source"] for row in actual] == [row["source"] for row in expected]
    assert actual != expected
    assert len({(row["source"], row["row"]) for row in actual}) == 3000
    next_epoch = mixture.shuffle(seed=14, keep_in_memory=True)
    assert [next_epoch[i] for i in range(1000)] != actual[:1000]
    assert Counter(next_epoch[i]["source"] for i in range(1000)) == {0: 999, 1: 1}
    assert [mixture[i] for i in range(2400, 1700, -1)] == expected[2400:1700:-1]
    other = DeterministicMixture(raw, [0.999, 0.001], "all_exhausted", seed=43)
    assert [other[i]["source"] for i in range(3000)] != [row["source"] for row in expected]


@pytest.mark.parametrize("strategy", ["first_exhausted", "all_exhausted"])
@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize(
    "lengths,quotas",
    [
        ([1, 1], [3, 2]),
        ([7, 11], [3, 2]),
        ([15, 10], [3, 2]),
        ([30, 1], [3, 2]),
        ([13], [1]),
        ([2, 13, 5], [5, 3, 2]),
        ([7, 11, 13], [3, 0, 2]),
    ],
)
def test_exhaustion_and_source_wrap(strategy, seed, lengths, quotas):
    cycle_size = sum(quotas)
    mixture = DeterministicMixture(sources(lengths), [quota / cycle_size for quota in quotas], strategy, seed=seed)
    counts = Counter()
    for index in range(len(mixture)):
        row = mixture[index]
        assert row["row"] == counts[row["source"]] % lengths[row["source"]]
        exhausted = [counts[source] >= size for source, size in enumerate(lengths) if quotas[source]]
        assert not (any(exhausted) if strategy == "first_exhausted" else all(exhausted))
        counts[row["source"]] += 1
    exhausted = [counts[source] >= size for source, size in enumerate(lengths) if quotas[source]]
    assert any(exhausted) if strategy == "first_exhausted" else all(exhausted)
    for start in range(0, len(mixture) - cycle_size + 1, cycle_size):
        assert Counter(mixture[i]["source"] for i in range(start, start + cycle_size)) == {
            source: quota for source, quota in enumerate(quotas) if quota
        }
    with pytest.raises(IndexError):
        mixture[len(mixture)]
    assert len(mixture.take(2)) == min(len(mixture), 2)


@pytest.mark.parametrize("count", [0, 1, 5, 7, 13, 100])
def test_take_preserves_selected_rows_when_shuffled(count):
    mixture = DeterministicMixture(sources([9, 5]), [0.6, 0.4], "all_exhausted", seed=7)
    selected = mixture.take(count)
    expected = [mixture[i] for i in range(min(count, len(mixture)))]
    assert list(selected) == expected
    shuffled = list(selected.shuffle(seed=13, keep_in_memory=True))
    assert [row["source"] for row in shuffled] == [row["source"] for row in expected]
    assert {(row["source"], row["row"]) for row in shuffled} == {(row["source"], row["row"]) for row in expected}
    with pytest.raises(ValueError, match="nonnegative"):
        mixture.take(-1)


def test_disabled_and_empty_sources():
    mixture = DeterministicMixture(sources([4, 0]), [1, 0], "all_exhausted", seed=0)
    assert [mixture[i] for i in range(len(mixture))] == [{"source": 0, "row": i} for i in range(4)]
    with pytest.raises(ValueError, match="nonempty"):
        DeterministicMixture(sources([4, 0]), None, "all_exhausted", seed=0)


def test_config_gates_dataset_loading(tmp_path):
    for name in ["train", "test"]:
        (tmp_path / f"{name}.jsonl").write_text("\n".join('{"completion": "example"}' for _ in range(10)))
    config = SFTDataConfig(name=str(tmp_path), splits=["train", "test"], probabilities=[0.6, 0.4], seed=27)
    assert isinstance(load_sft_dataset(config), Dataset)
    config.deterministic_sampling = True
    mixture = load_sft_dataset(config)
    assert isinstance(mixture, DeterministicMixture)
    assert mixture.seed == 27
    assert Counter(mixture[i]["__split"] for i in range(5)) == {"train": 3, "test": 2}
