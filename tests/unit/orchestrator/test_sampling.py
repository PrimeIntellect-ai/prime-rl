import random

import pytest

from prime_rl.orchestrator.sampling import WeightedRoundRobin


def test_weighted_round_robin_is_deterministic_per_rng():
    """Same seed → same pick sequence (env selection is reproducible)."""
    names = ["a", "b", "c"]
    weights = [1.0, 2.0, 3.0]
    a = WeightedRoundRobin(names, weights, rng=random.Random(0))
    b = WeightedRoundRobin(names, weights, rng=random.Random(0))
    assert [a.pick() for _ in range(100)] == [b.pick() for _ in range(100)]


def test_weighted_round_robin_respects_weights():
    """A heavily-weighted env dominates; a zero-weight env is never picked."""
    wrr = WeightedRoundRobin(["rare", "common", "never"], [1.0, 99.0, 0.0], rng=random.Random(0))
    picks = [wrr.pick() for _ in range(10_000)]
    assert "never" not in picks
    assert picks.count("common") > picks.count("rare")


def test_weighted_round_robin_rejects_empty():
    with pytest.raises(ValueError):
        WeightedRoundRobin([], [], rng=random.Random(0))
