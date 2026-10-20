"""The platform train monitor's wall-time snapshot merging.

The platform rejects step-less metric records (step-keyed storage), so the
monitor caches wall-time (``step=None``) payloads per metric family, sums the
drained dispatcher delta counters, and merges both into the next strictly
advancing train row. Everything else uploads bare.
"""

import asyncio
from typing import Any

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


def train_row(step: int) -> dict[str, Any]:
    """A minimal orchestrator train row: metrics under the ``train/`` prefix."""
    return {"train/agg/effective/reward/mean": 0.1, "step": step}


def test_wall_time_rows_are_cached_not_uploaded():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/kv_cache_usage_perc": 0.5}, step=None))
    asyncio.run(monitor.log_metrics({"dispatcher/inflight/train": 3.0}, step=None))

    # No step-less record ever reaches the platform.
    assert run.rows == []


def test_latest_wall_time_values_merge_into_next_train_step_upload():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/num_requests_running": 4.0}, step=None))
    asyncio.run(monitor.log_metrics({"inference/server0.0/num_requests_running": 1.0}, step=None))
    asyncio.run(monitor.log_metrics({**train_row(1), "train/agg/effective/reward/mean": 0.9}, step=1))

    (metrics, step), = run.rows
    assert step == 1
    assert metrics["inference/server0.0/num_requests_running"] == 1.0  # latest value wins
    assert metrics["train/agg/effective/reward/mean"] == 0.9


def test_explicit_step_metrics_win_over_cached_gauges():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"dispatcher/inflight/train": 3.0}, step=None))
    asyncio.run(monitor.log_metrics({**train_row(2), "dispatcher/inflight/train": 7.0}, step=2))

    (metrics, step), = run.rows
    assert step == 2
    assert metrics["dispatcher/inflight/train"] == 7.0


def test_step_zero_is_an_explicit_step_upload():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"concurrency/max_inflight": 128.0}, step=None))
    asyncio.run(monitor.log_metrics(train_row(0), step=0))

    (metrics, step), = run.rows
    assert step == 0  # falsy step is not a wall-time row
    assert metrics["concurrency/max_inflight"] == 128.0


def test_embedded_step_key_is_a_step_row_not_wall_time():
    """The SDK honors a payload-embedded ``step`` key (it only fills a missing
    one), so a ``step=None`` call carrying one must be uploaded as-is, not
    cached like a wall-time sample."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({**train_row(5), "step": 5}, step=None))

    (metrics, step), = run.rows
    assert step == 5
    assert metrics["step"] == 5
    assert metrics["train/agg/effective/reward/mean"] == 0.1


def test_eval_rows_upload_bare_same_step_and_late():
    """Gauges attach to the train row only. An eval epoch finishing late must
    not overwrite the step's already-stored gauges with newer samples."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/kv_cache_usage_perc": 0.5}, step=None))
    asyncio.run(monitor.log_metrics({"dispatcher/cancelled/train": 1.0}, step=None))
    asyncio.run(monitor.log_metrics(train_row(5), step=5))
    # Same-step eval row, then a late eval row for a historical step.
    asyncio.run(monitor.log_metrics({"eval/env/reward/mean": 0.8, "step": 5}, step=5))
    asyncio.run(monitor.log_metrics({"eval/env/reward/mean": 0.7, "step": 3}, step=3))

    train_metrics, _ = run.rows[0]
    assert train_metrics["inference/server0.0/kv_cache_usage_perc"] == 0.5
    assert train_metrics["dispatcher/cancelled/train"] == 1.0
    assert run.rows[1][0] == {"eval/env/reward/mean": 0.8, "step": 5}  # bare
    assert run.rows[2][0] == {"eval/env/reward/mean": 0.7, "step": 3}  # bare, no overwrite


def test_same_step_train_row_uploads_bare():
    """The attach gate is strictly advancing: a repeated train row at the same
    step does not re-stamp the step's gauges."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/gauge": 1.0}, step=None))
    asyncio.run(monitor.log_metrics(train_row(1), step=1))
    asyncio.run(monitor.log_metrics({"inference/server0.0/gauge": 9.0}, step=None))
    asyncio.run(monitor.log_metrics(train_row(1), step=1))

    assert run.rows[0][0]["inference/server0.0/gauge"] == 1.0
    assert "inference/server0.0/gauge" not in run.rows[1][0]


def test_row_without_train_metrics_never_attaches():
    """The train row is identified by its ``train/`` metrics; any other step
    row (e.g. a trainer-side perf row) uploads bare."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/gauge": 1.0}, step=None))
    asyncio.run(monitor.log_metrics({"optim/lr": 1e-4, "step": 2}, step=2))
    asyncio.run(monitor.log_metrics(train_row(3), step=3))

    assert "inference/server0.0/gauge" not in run.rows[0][0]
    assert run.rows[1][0]["inference/server0.0/gauge"] == 1.0


def test_drained_deltas_sum_across_ticks_into_step_total():
    """``dispatcher/cancelled|errored`` arrive as per-tick deltas; the platform
    row carries their sum over the step window, not the latest tick."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"dispatcher/cancelled/train": 1.0, "dispatcher/errored/env": 2.0}, step=None))
    asyncio.run(monitor.log_metrics({"dispatcher/cancelled/train": 2.0, "dispatcher/errored/env": 0.0}, step=None))
    asyncio.run(monitor.log_metrics({"inference/server0.0/gauge": 1.0}, step=None))
    asyncio.run(monitor.log_metrics(train_row(4), step=4))

    (metrics, _), = run.rows
    assert metrics["dispatcher/cancelled/train"] == 3.0
    assert metrics["dispatcher/errored/env"] == 2.0
    assert metrics["inference/server0.0/gauge"] == 1.0


def test_deltas_reset_only_on_attach_not_on_bare_rows():
    """Totals survive bare same-step and late-eval uploads; only an attached
    train row consumes them."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"dispatcher/cancelled/train": 1.0}, step=None))
    asyncio.run(monitor.log_metrics({"eval/env/reward/mean": 0.8, "step": 1}, step=1))  # bare
    asyncio.run(monitor.log_metrics(train_row(1), step=1))  # attaches, consumes
    asyncio.run(monitor.log_metrics({"dispatcher/cancelled/train": 5.0}, step=None))
    asyncio.run(monitor.log_metrics(train_row(1), step=1))  # same step: bare
    asyncio.run(monitor.log_metrics(train_row(2), step=2))  # attaches

    assert run.rows[0][0] == {"eval/env/reward/mean": 0.8, "step": 1}  # bare
    assert run.rows[1][0]["dispatcher/cancelled/train"] == 1.0
    assert "dispatcher/cancelled/train" not in run.rows[2][0]  # bare, totals survived
    assert run.rows[3][0]["dispatcher/cancelled/train"] == 5.0  # totals from after the attach


def test_wall_time_payload_replaces_its_families_wholesale():
    """A family's snapshot is replaced, not per-key updated: dead keys (e.g.
    a removed engine) drop out instead of lingering forever."""
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/gauge": 1.0, "inference/server0.1/gauge": 2.0}, step=None))
    asyncio.run(monitor.log_metrics({"inference/server1.0/gauge": 3.0}, step=None))
    asyncio.run(monitor.log_metrics(train_row(1), step=1))

    (metrics, _), = run.rows
    assert "inference/server0.0/gauge" not in metrics
    assert "inference/server0.1/gauge" not in metrics
    assert metrics["inference/server1.0/gauge"] == 3.0


def test_nonfinite_wall_time_values_are_dropped_not_cached():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({"inference/server0.0/gauge": float("nan")}, step=None))
    asyncio.run(monitor.log_metrics(train_row(1), step=1))

    (metrics, _), = run.rows
    assert "inference/server0.0/gauge" not in metrics


def test_monitors_do_not_share_cache():
    monitor_a, run_a = make_monitor()
    monitor_b, run_b = make_monitor()

    asyncio.run(monitor_a.log_metrics({"inference/server0.0/gauge": 1.0}, step=None))
    asyncio.run(monitor_b.log_metrics(train_row(1), step=1))

    assert run_b.rows[0][0] == train_row(1)


def test_no_wall_time_cache_means_unchanged_step_upload():
    monitor, run = make_monitor()

    asyncio.run(monitor.log_metrics({**train_row(3), "train/agg/effective/reward/mean": 0.6}, step=3))

    (metrics, step), = run.rows
    assert metrics == {**train_row(3), "train/agg/effective/reward/mean": 0.6}
    assert step == 3
