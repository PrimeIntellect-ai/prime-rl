from collections import Counter

from prime_rl.orchestrator.train_sink import select_by_quota
from prime_rl.orchestrator.train_source import inflight_caps, mixer_weight, smooth_round_robin, trace_quotas


def test_dispatch_follows_trace_share_over_group_size_and_acceptance() -> None:
    # 1:1 trace share; math groups are 16 traces and half accepted, swe groups 4 and all accepted
    targets = {"math": 0.5 * 512 / 16, "swe": 0.5 * 512 / 4}
    rates = {"math": 0.5, "swe": 1.0}
    weights = {env: mixer_weight(targets[env], 0.0, rates[env]) for env in targets}
    current = {env: 0.0 for env in targets}
    picks = Counter(smooth_round_robin(current, weights) for _ in range(600))
    accepted_traces = {"math": picks["math"] * 0.5 * 16, "swe": picks["swe"] * 1.0 * 4}
    assert accepted_traces["math"] == accepted_traces["swe"]
    assert max(current.values()) - min(current.values()) <= sum(weights.values())


def test_deficit_term_vanishes_once_batch_target_is_queued() -> None:
    assert mixer_weight(8.0, 0.0, 0.5) == 16.0
    assert mixer_weight(8.0, 4.0, 0.5) == 12.0
    assert mixer_weight(8.0, 20.0, 0.5) == 8.0
    # an env that accepts nothing is dispatched like one that accepts everything
    assert mixer_weight(8.0, 0.0, 0.0) == mixer_weight(8.0, 0.0, 1.0) == 8.0


def test_inflight_caps_split_slots_by_littles_law_with_staleness_clip() -> None:
    caps = inflight_caps({"fast": 100.0, "slow": 100.0}, {"fast": 10.0, "slow": 30.0}, 400, None)
    assert caps == {"fast": 125.0, "slow": 375.0}
    assert inflight_caps({"fast": 100.0, "slow": 10.0}, {"fast": 1.0, "slow": 1000.0}, 400, 1)["slow"] == 20.0


def test_quotas_and_selection() -> None:
    assert trace_quotas({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 512) == {"a": 171, "b": 171, "c": 170}
    quotas = {"a": 2, "b": 2}
    assert select_by_quota(["a", "a", "a", "b", "a", "b"], quotas) == ([0, 1, 3, 5], 0)
    # b is short by one: the oldest leftover trace fills its slot
    assert select_by_quota(["a", "a", "a", "b", "a"], quotas) == ([0, 1, 2, 3], 1)
    assert select_by_quota(["a"] * 5, {"a": 4}) == ([0, 1, 2, 3], 0)
