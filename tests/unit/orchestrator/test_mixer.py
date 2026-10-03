from collections import Counter

from prime_rl.orchestrator.train_sink import select_by_quota
from prime_rl.orchestrator.train_source import inflight_caps, mixer_weight, plan_quotas, smooth_round_robin


def test_dispatch_corrects_for_acceptance() -> None:
    # 1:1 prompt share; math groups are half accepted, swe groups all accepted
    targets = {"math": 16.0, "swe": 16.0}
    rates = {"math": 0.5, "swe": 1.0}
    weights = {env: mixer_weight(targets[env], 0.0, rates[env]) for env in targets}
    current = {env: 0.0 for env in targets}
    picks = Counter(smooth_round_robin(current, weights) for _ in range(600))
    assert picks["math"] * rates["math"] == picks["swe"] * rates["swe"]
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


def test_quotas_are_whole_groups_and_even_out_across_batches() -> None:
    sizes = {"a": 8, "b": 8, "c": 8}
    credit = dict.fromkeys(sizes, 0.0)
    targets = dict.fromkeys(sizes, 512 / 24)
    batches = [plan_quotas(credit, targets, sizes, 512) for _ in range(3)]
    assert all(sum(quotas.values()) == 512 and all(q % 8 == 0 for q in quotas.values()) for quotas in batches)
    assert {env: sum(quotas[env] for quotas in batches) for env in sizes} == dict.fromkeys(sizes, 512)
    # one-group batch, two envs: whole groups alternate instead of splitting both
    credit = {"a": 0.0, "b": 0.0}
    tiny = [plan_quotas(credit, {"a": 0.5, "b": 0.5}, {"a": 8, "b": 8}, 8) for _ in range(2)]
    assert sorted(tuple(quotas.values()) for quotas in tiny) == [(0, 8), (8, 0)]


def test_selection_ships_oldest_policy_first() -> None:
    quotas = {"a": 2, "b": 2}
    assert select_by_quota([("a", 3), ("a", 1), ("a", 2), ("b", 1), ("b", 2)], quotas) == ([1, 2, 3, 4], 0)
    # b is short by one: the oldest-policy leftover trace fills its slot
    assert select_by_quota([("a", 3), ("a", 2), ("a", 1), ("b", 1)], quotas) == ([0, 1, 2, 3], 1)
    assert select_by_quota([("a", 5), ("a", 4), ("b", 9), ("a", 6), ("a", 0)], {"a": 2, "b": 1}) == ([1, 2, 4], 0)
