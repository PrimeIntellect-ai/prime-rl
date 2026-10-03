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


def test_inflight_caps_split_slots_by_littles_law_with_staleness_clip() -> None:
    caps = inflight_caps({"fast": 100.0, "slow": 100.0}, {"fast": 10.0, "slow": 30.0}, 400, None)
    assert caps == {"fast": 125.0, "slow": 375.0}
    assert inflight_caps({"fast": 100.0, "slow": 10.0}, {"fast": 1.0, "slow": 1000.0}, 400, 1)["slow"] == 20.0
