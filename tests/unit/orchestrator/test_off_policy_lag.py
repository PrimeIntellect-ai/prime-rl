"""Tests for unified off-policy lag (#2674)."""

from prime_rl.orchestrator.dispatcher import off_policy_lag


def test_off_policy_lag_inflight_during_version_bumps():
    assert off_policy_lag(8, 5) == 2


def test_off_policy_lag_sink_queue():
    assert off_policy_lag(10, 5) == 4


def test_off_policy_lag_never_negative():
    assert off_policy_lag(3, 5) == 0
