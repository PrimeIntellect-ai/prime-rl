from collections import Counter

import pytest

from prime_rl.orchestrator.sampling import (
    ShuffledCursorSampler,
    WeightedRoundRobin,
)


def _rows(n: int) -> list[dict]:
    return [{"example_id": i, "env_name": "e"} for i in range(n)]


def test_shuffled_cursor_cycles_without_repeats_then_reshuffles():
    sampler = ShuffledCursorSampler(_rows(5), seed=42)
    cycle1 = [sampler.next()["example_id"] for _ in range(5)]
    cycle2 = [sampler.next()["example_id"] for _ in range(5)]
    # Each cycle visits every example exactly once (cursor), then reshuffles.
    assert sorted(cycle1) == list(range(5))
    assert sorted(cycle2) == list(range(5))


def test_shuffled_cursor_dataset_size():
    assert ShuffledCursorSampler(_rows(7), seed=0).dataset_size == 7


def test_shuffled_cursor_is_deterministic_per_seed():
    a = ShuffledCursorSampler(_rows(8), seed=123)
    b = ShuffledCursorSampler(_rows(8), seed=123)
    assert [a.next()["example_id"] for _ in range(8)] == [b.next()["example_id"] for _ in range(8)]


def test_shuffled_cursor_empty_raises():
    with pytest.raises(ValueError):
        ShuffledCursorSampler([], seed=0)


def test_observe_default_is_noop():
    sampler = ShuffledCursorSampler(_rows(3), seed=0)
    # Default observe accepts a (possibly empty) group and does nothing observable.
    assert sampler.observe([]) is None


def test_weighted_round_robin_honors_weights():
    mix = WeightedRoundRobin(["A", "B"], [1.0, 3.0], seed=0)
    counts = Counter(mix.pick() for _ in range(4000))
    # B weighted 3x A — expect roughly a 3:1 split.
    assert 2.5 < counts["B"] / counts["A"] < 3.5


def test_weighted_round_robin_is_deterministic_per_seed():
    a = WeightedRoundRobin(["A", "B", "C"], [1.0, 1.0, 1.0], seed=7)
    b = WeightedRoundRobin(["A", "B", "C"], [1.0, 1.0, 1.0], seed=7)
    assert [a.pick() for _ in range(20)] == [b.pick() for _ in range(20)]


def test_weighted_round_robin_empty_raises():
    with pytest.raises(ValueError):
        WeightedRoundRobin([], [], seed=0)
