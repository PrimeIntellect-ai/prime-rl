import dataclasses
import math

import numpy as np
import pytest

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.orchestrator import goodput
from prime_rl.orchestrator.concurrency import EngineLoadSample
from prime_rl.orchestrator.goodput import GoodputController, GroupDurationLaw, bin_of, bin_size


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(goodput, "time", c)
    return c


def sample(tokens: float, preemptions: int = 0, running: int = 0) -> EngineLoadSample:
    return EngineLoadSample(
        engine_id="0",
        role=None,
        kv_capacity_tokens=None,
        max_model_len=None,
        kv_usage=0.0,
        running=running,
        waiting=0,
        waiting_capacity=None,
        preemptions_delta=preemptions,
        generation_tokens_delta=tokens,
    )


def law_from(groups: list[list[float]]) -> GroupDurationLaw:
    """Groups whose members progress at the average pace: virtual finish == tokens."""
    law = GroupDurationLaw(window=10_000)
    for g in groups:
        law.add_complete(g, g)
    return law


def test_bins_roundtrip():
    for i in range(0, 60):
        assert bin_of(bin_size(i)) == i or bin_size(i) == bin_size(bin_of(bin_size(i)))


def test_law_terms_uncensored_match_empirical():
    rng = np.random.default_rng(0)
    groups = [list(rng.lognormal(7, 0.6, 8)) for _ in range(400)]
    law = law_from(groups)
    x = float(np.median([max(g) for g in groups]))
    S, A, C = law.terms(x, 8.0, None)
    fresh = [g for g in groups if max(g) <= x]
    assert S == pytest.approx(len(fresh) / len(groups), abs=1e-9)
    assert A == pytest.approx(sum(sum(g) for g in fresh) / len(groups), rel=1e-9)
    assert C == pytest.approx(sum(sum(min(l, x) for l in g) for g in groups) / len(groups), rel=1e-9)


def test_censoring_shifts_mass_to_tail():
    groups = [[100.0, 200.0]] * 50 + [[100.0, 1000.0]] * 50
    plain = law_from(groups)
    censored = law_from(groups)
    for _ in range(50):
        censored.add_censored(500.0, 2, [100.0], [100.0])  # dropped at virtual age 500
    S_plain, _, _ = plain.terms(300.0, 8.0, None)
    S_cens, _, _ = censored.terms(300.0, 8.0, None)
    # Censored groups are not events: the short-group mass is diluted, not inflated
    assert S_cens < S_plain
    assert S_plain == pytest.approx(0.5)


def make_ctl(clock, *, staleness: bool = True, **cfg):
    defaults = dict(initial_inflight=32, max_inflight=4096)
    defaults.update(cfg)
    if not staleness:  # throughput only, as in evals
        return GoodputController(ConcurrencyConfig(**defaults))
    return GoodputController(ConcurrencyConfig(**defaults), batch_size=1024, max_off_policy_steps=8)


def test_eta_decreasing_in_pool_and_group_size(clock):
    rng = np.random.default_rng(1)

    def eta_curve(group_size):
        ctl = make_ctl(clock)
        for _ in range(400):
            g = list(rng.lognormal(math.log(3000), 0.8, group_size))
            ctl.law.add_complete(g, g)
        return [ctl.eta(p, 1.0) for p in (256, 1024, 4096, 16384)]

    small, large = eta_curve(4), eta_curve(64)
    assert all(a >= b - 1e-12 for a, b in zip(small, small[1:]))
    assert all(a >= b - 1e-12 for a, b in zip(large, large[1:]))
    assert all(s >= l - 1e-9 for s, l in zip(small, large))
    assert small[0] == pytest.approx(1.0, abs=1e-3)
    assert large[-1] < 0.9


def test_eta_is_one_without_staleness_bound(clock):
    ctl = GoodputController(ConcurrencyConfig())
    for _ in range(50):
        ctl.law.add_complete([1000.0, 50000.0], [1000.0, 50000.0])
    assert ctl.eta(10_000, 1.0) == 1.0


def drive(ctl, clock, throughput, seconds, *, poll=5.0, pool=None, preempt=None):
    """Feed polls with T = throughput(inflight); the pool fills the cap unless ``pool`` caps it."""
    inflight = {"n": ctl.max_inflight}
    ctl.bind(set_limit=lambda n: inflight.__setitem__("n", n), get_inflight=lambda: inflight["n"])
    caps = []
    end = clock.t + seconds
    while clock.t < end:
        clock.t += poll
        n = min(ctl.max_inflight, pool) if pool is not None else ctl.max_inflight
        inflight["n"] = n
        p = preempt(n) if preempt is not None else 0
        ctl.observe([sample(throughput(n) * poll, preemptions=p, running=n)])
        caps.append(ctl.max_inflight)
    return caps


def test_climbs_to_saturation_knee(clock):
    ctl = make_ctl(clock, staleness=False)
    knee = 400.0
    caps = drive(ctl, clock, lambda p: 20_000 * p / (p + knee), 4 * 3600)
    late = np.median(caps[len(caps) // 2 :])
    assert late >= knee


def test_no_growth_without_staleness_evidence(clock):
    ctl = make_ctl(clock)
    caps = drive(ctl, clock, lambda p: 10.0 * p, 3600)
    assert max(caps) == 32


def test_backs_off_a_cliff(clock):
    ctl = make_ctl(clock, staleness=False)
    caps = drive(ctl, clock, lambda p: 30.0 * p if p <= 200 else 1500.0, 4 * 3600)
    late = caps[len(caps) // 2 :]
    assert np.mean([c <= 200 for c in late]) > 0.8
    assert max(late) <= bin_size(bin_of(200) + 2)


def test_fresh_fraction_bounds_growth_when_throughput_keeps_scaling(clock):
    rng = np.random.default_rng(2)
    ctl = make_ctl(clock)
    for _ in range(400):
        g = list(rng.lognormal(math.log(3000), 0.8, 64))
        ctl.law.add_complete(g, g)
    caps = drive(ctl, clock, lambda p: 15.0 * p, 6 * 3600)
    late = np.median(caps[len(caps) // 2 :])
    # Goodput keeps rising with linear throughput; the fresh-fraction bound stops it
    best = max(i for i in range(5, 60) if ctl.eta(bin_size(i), 1.0) >= ctl.min_fresh)
    assert abs(bin_of(late) - best) <= 1
    assert ctl.eta(late, 1.0) >= ctl.min_fresh - 0.02


def test_preemption_thrash_steps_down_without_cascading(clock):
    ctl = make_ctl(clock, initial_inflight=512)
    before = ctl.max_inflight
    drive(ctl, clock, lambda p: 10.0 * p, 20, preempt=lambda n: n // 20)
    assert ctl.max_inflight < before
    # The pool still holds the old cap's work: further thrash must not cut again
    inflight = {"n": before}
    ctl.bind(set_limit=lambda n: None, get_inflight=lambda: inflight["n"])
    cut = ctl.max_inflight
    for _ in range(20):
        clock.t += 5.0
        ctl.observe([sample(1000.0, preemptions=before // 20, running=before)])
    assert ctl.max_inflight == cut


def test_occasional_preemptions_are_tolerated(clock):
    ctl = make_ctl(clock, initial_inflight=512)
    before = ctl.max_inflight
    drive(ctl, clock, lambda p: 10.0 * p, 20, preempt=lambda n: 1)
    assert ctl.max_inflight >= before


def test_no_growth_while_pool_does_not_fill_cap(clock):
    ctl = make_ctl(clock, initial_inflight=64)
    caps = drive(ctl, clock, lambda p: 10.0 * p, 3 * 3600, pool=40)
    assert max(caps) <= 64


def test_group_events_feed_law(clock):
    ctl = make_ctl(clock)
    for g in range(20):
        ctl.on_group_start(g, 4)
        for _ in range(4):
            ctl.on_episode_done(g, 1000, wall_s=60.0)
    ctl.on_group_start("dropped", 4)
    ctl.on_episode_done("dropped", 500, wall_s=60.0)
    ctl.on_group_drop("dropped", "stale")
    ctl.on_group_start("superseded", 4)
    ctl.on_group_drop("superseded", "superseded")
    assert ctl.law.completed == 20
    assert len(ctl.law) == 21
    assert ctl.lifetime == pytest.approx(60.0)
    assert not ctl.groups


def test_schedule_mode_pins_cap_and_measures(clock):
    ctl = GoodputController(
        ConcurrencyConfig(mode="schedule", schedule=[(0, 100), (600, 300)], max_inflight=4096),
        batch_size=1024,
        max_off_policy_steps=8,
    )
    assert ctl.max_inflight == 100
    caps = drive(ctl, clock, lambda p: 10.0 * p, 1200)
    assert set(caps) == {100, 300}
    assert caps[-1] == 300
    assert bin_of(100) in ctl.bins and ctl.bins[bin_of(100)].mean == pytest.approx(1000.0)


def test_failed_episodes_stay_out_of_the_law(clock):
    ctl = make_ctl(clock)
    for g in range(20):
        ctl.on_group_start(g, 4)
        for _ in range(4):
            ctl.on_episode_done(g, 0, wall_s=5.0, ok=False)  # e.g. sandbox creation failed
    assert ctl.law.completed == 0 and ctl.errored_groups == 20 and not ctl.law_ready
    # A partly failed group still measures its surviving members
    ctl.on_group_start("partial", 4)
    for tokens in (900, 1000, 1100):
        ctl.on_episode_done("partial", tokens, wall_s=60.0)
    ctl.on_episode_done("partial", 0, wall_s=2.0, ok=False)
    assert ctl.law.completed == 1
    assert list(ctl.law.obs[-1][2]) == [900, 1000, 1100]
    assert ctl.lifetime == pytest.approx(60.0)


def test_bootstrap_counts_engines_not_metric_endpoints(clock):
    ctl = GoodputController(ConcurrencyConfig(max_inflight=4096), batch_size=1024, max_off_policy_steps=8)
    ctl.bind(set_limit=lambda n: None, get_inflight=lambda: 0)
    # One DP deployment of 4 engines behind 4 API servers: each also exposes engine "0"
    keys = ["h#0", "h#0", "h#1", "h#0", "h#2", "h#0", "h#3"]
    samples = [dataclasses.replace(sample(0.0), engine_id=f"server{i}", engine_key=k) for i, k in enumerate(keys)]
    ctl.observe(samples)
    assert ctl.max_inflight == bin_size(bin_of(64 * 4))


def test_trainer_step_time_from_busy_policy_updates(clock):
    ctl = make_ctl(clock)
    assert ctl.train_step_s is None
    for t, lead in [(0, 1), (90, 1), (180, 1), (270, 0), (500, 1), (590, 1)]:
        clock.t = t
        ctl.on_policy_update(lead)
    # 270 -> 500 followed an idle trainer (lead 0): not a step time
    assert ctl.train_step_s == pytest.approx(90.0)
