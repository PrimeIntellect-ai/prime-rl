"""The platform train monitor's wall-time snapshot merging.

The platform rejects step-less metric records (step-keyed storage; the Sep 7
queue incident), so the monitor caches wall-time (``step=None``) payloads per
metric family and merges the latest values into the next explicit-step upload
instead of dropping or uploading them as-is.
"""

import asyncio
from typing import Any

import pytest

from prime_rl.configs.monitors import PrimeTrainMonitorConfig
from prime_rl.monitors.prime import PrimeTrainMonitor


class FakeRun:
    """Records the metrics POSTs the SDK would queue."""

    def __init__(self):
        self.rows: list[tuple[dict[str, Any], int | None]] = []

    def log_metrics(self, values, *, step=None):
        self.rows.append((dict(values), step))


def make_monitor() -> tuple[PrimeTrainMonitor, FakeRun]:
    run = FakeRun()
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    monitor.run = run
    return monitor, run


def test_wall_time_rows_are_cached_not_uploaded():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/kv_cache_usage_perc": 0.5}, step=None))
    asyncio.run(monitor.log_metrics({"dispatcher/inflight/train": 3.0}, step=None))

    # No step-less record ever reaches the platform.
    assert run.rows == []


def test_latest_wall_time_values_merge_into_next_step_upload():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/num_requests_running": 4.0}, step=None))
    asyncio.run(monitor.log_metrics({"inference/server0.0/num_requests_running": 1.0}, step=None))
    asyncio.run(monitor.log_metrics({"train/agg/effective/reward/mean": 0.9, "step": 1}, step=1))

    (metrics, step), = run.rows
    assert step == 1
    assert metrics["inference/server0.0/num_requests_running"] == 1.0  # latest value wins
    assert metrics["train/agg/effective/reward/mean"] == 0.9


def test_explicit_step_metrics_win_over_cached_gauges():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"dispatcher/inflight/train": 3.0}, step=None))
    asyncio.run(monitor.log_metrics({"dispatcher/inflight/train": 7.0}, step=2))

    (metrics, step), = run.rows
    assert step == 2
    assert metrics["dispatcher/inflight/train"] == 7.0


def test_step_zero_is_an_explicit_step_upload():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"concurrency/max_inflight": 128.0}, step=None))
    asyncio.run(monitor.log_metrics({"train/agg/effective/reward/mean": 0.5}, step=0))

    (metrics, step), = run.rows
    assert step == 0  # falsy step is not a wall-time row
    assert metrics["concurrency/max_inflight"] == 128.0


def test_embedded_step_key_is_a_step_row_not_wall_time():
    """The SDK honors a payload-embedded ``step`` key (it only fills a missing
    one), so a ``step=None`` call carrying one must be uploaded as-is, not
    cached like a wall-time sample."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"step": 5, "train/agg/effective/reward/mean": 0.4}, step=None))

    (metrics, step), = run.rows
    assert step == 5
    assert metrics["step"] == 5
    assert metrics["train/agg/effective/reward/mean"] == 0.4


def test_same_step_follow_up_uploads_still_carry_latest_gauges():
    """The cache survives a step upload: a second upload at the same step (e.g.
    an eval epoch's row) also carries the latest known gauges."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/kv_cache_usage_perc": 0.5}, step=None))
    asyncio.run(monitor.log_metrics({"train/agg/effective/reward/mean": 0.9, "step": 1}, step=1))
    asyncio.run(monitor.log_metrics({"eval/env/reward/mean": 0.8}, step=1))

    train_metrics, _ = run.rows[0]
    eval_metrics, _ = run.rows[1]
    assert train_metrics["inference/server0.0/kv_cache_usage_perc"] == 0.5
    assert eval_metrics["inference/server0.0/kv_cache_usage_perc"] == 0.5


def test_wall_time_payload_replaces_its_families_wholesale():
    """A family's snapshot is replaced, not per-key updated: dead keys (e.g.
    a removed engine) drop out instead of lingering forever."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/gauge": 1.0, "inference/server0.1/gauge": 2.0}, step=None))
    asyncio.run(monitor.log_metrics({"inference/server1.0/gauge": 3.0}, step=None))
    asyncio.run(monitor.log_metrics({"train/agg/effective/reward/mean": 0.1, "step": 1}, step=1))

    (metrics, _), = run.rows
    assert "inference/server0.0/gauge" not in metrics
    assert "inference/server0.1/gauge" not in metrics
    assert metrics["inference/server1.0/gauge"] == 3.0


def test_nonfinite_wall_time_values_are_dropped_not_cached():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/gauge": float("nan")}, step=None))
    asyncio.run(monitor.log_metrics({"train/agg/effective/reward/mean": 0.2, "step": 1}, step=1))

    (metrics, _), = run.rows
    assert "inference/server0.0/gauge" not in metrics


def test_monitors_do_not_share_cache():
    monitor_a, run_a = make_monitor()
    monitor_b, run_b = make_monitor()

    asyncio.run(monitor_a.log_metrics({"inference/server0.0/gauge": 1.0}, step=None))
    asyncio.run(monitor_b.log_metrics({"train/agg/effective/reward/mean": 0.3, "step": 1}, step=1))

    assert run_b.rows[0][0] == {"train/agg/effective/reward/mean": 0.3, "step": 1}


def test_no_wall_time_cache_means_unchanged_step_upload():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"train/agg/effective/reward/mean": 0.6, "step": 3}, step=3))

    (metrics, step), = run.rows
    assert metrics == {"train/agg/effective/reward/mean": 0.6, "step": 3}
    assert step == 3
