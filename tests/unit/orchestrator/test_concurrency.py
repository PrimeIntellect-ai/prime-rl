"""Unit tests for the adaptive concurrency control law (``decide_limit``)."""

from prime_rl.configs.orchestrator import AdaptiveConcurrencyConfig
from prime_rl.orchestrator.concurrency import decide_limit


def _decide(**overrides):
    config = overrides.pop("config", AdaptiveConcurrencyConfig())
    kwargs = dict(
        current_limit=100,
        inflight=95,  # saturating (>= 0.9 * 100)
        kv_usage=0.4,
        kv_delta=0.0,  # settled
        preemption_rate=0.0,
        max_inflight=1000,
        config=config,
    )
    kwargs.update(overrides)
    return decide_limit(**kwargs)


def test_grows_when_below_target_settled_and_saturating():
    new, decision = _decide(kv_usage=0.4)
    assert decision == "grow"
    assert new > 100  # 1 + (1.5-1)*((0.8-0.4)/0.8) = 1.25 -> 125
    assert new == 125


def test_growth_tapers_toward_target():
    far, _ = _decide(kv_usage=0.2)
    near, _ = _decide(kv_usage=0.7)
    assert far - 100 > near - 100  # bigger step when further below target


def test_holds_when_usage_still_rising():
    # Below target and saturating, but KV is climbing fast -> the in-flight set's
    # KV hasn't materialized yet, so don't admit more.
    new, decision = _decide(kv_delta=0.1)
    assert decision == "hold"
    assert new == 100


def test_holds_when_not_saturating():
    # Limit isn't the bottleneck (few in flight) -> raising it wouldn't help.
    new, decision = _decide(inflight=10)
    assert decision == "hold"
    assert new == 100


def test_holds_in_deadband_between_target_and_high_water():
    new, decision = _decide(kv_usage=0.85)
    assert decision == "hold"
    assert new == 100


def test_backs_off_above_high_water():
    new, decision = _decide(kv_usage=0.97)
    assert decision == "backoff"
    assert new == 76  # int(min(100, 95) * 0.8)


def test_backs_off_on_preemptions():
    new, decision = _decide(kv_usage=0.5, preemption_rate=2.0)
    assert decision == "backoff"
    assert new < 100


def test_grow_clamped_to_max_inflight():
    new, _ = _decide(current_limit=900, inflight=890, kv_usage=0.1, max_inflight=1000)
    assert new == 1000


def test_backoff_clamped_to_min_inflight():
    config = AdaptiveConcurrencyConfig(min_inflight=8)
    new, _ = _decide(current_limit=10, inflight=10, kv_usage=0.99, config=config)
    assert new == 8
