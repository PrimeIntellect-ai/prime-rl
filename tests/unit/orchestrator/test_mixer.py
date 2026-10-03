from collections import deque

from prime_rl.orchestrator.train_source import inflight_caps, mixer_weight


def test_dispatch_corrects_for_acceptance() -> None:
    # 1:1 prompt share; math groups are half accepted, swe groups all accepted
    assert mixer_weight(16.0, 0.0, 0.5) == 2 * mixer_weight(16.0, 0.0, 1.0)


def test_deficit_term_vanishes_once_batch_target_is_queued() -> None:
    assert mixer_weight(8.0, 0.0, 0.5) == 16.0
    assert mixer_weight(8.0, 4.0, 0.5) == 12.0
    assert mixer_weight(8.0, 20.0, 0.5) == 8.0
    # an env that accepts nothing is dispatched like one that accepts everything
    assert mixer_weight(8.0, 0.0, 0.0) == mixer_weight(8.0, 0.0, 1.0) == 8.0


def test_inflight_caps_split_slots_by_littles_law_with_staleness_budget() -> None:
    durations = {"fast": deque([10.0]), "slow": deque([30.0])}
    bounds = {"fast": 1, "slow": 1}
    assert inflight_caps({"fast": 100.0, "slow": 100.0}, durations, 400, None, bounds) == {"fast": 125.0, "slow": 375.0}
    # 60 s steps: the slow env's 30 s episodes still fit 2 steps; at 10 s steps they need 3
    assert inflight_caps({"fast": 100.0, "slow": 100.0}, durations, 400, 60.0, bounds)["slow"] == 375.0
    assert inflight_caps({"fast": 100.0, "slow": 100.0}, durations, 400, 10.0, bounds)["slow"] == 250.0
