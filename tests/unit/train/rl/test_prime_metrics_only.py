"""The RL trainer's platform wiring: a metrics-only attach on a run it does not own."""

import asyncio
from types import SimpleNamespace

import pytest

from prime_rl.configs.monitors import PrimeTrainMonitorConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.monitors.prime import PrimeTrainMonitor, write_platform_record
from prime_rl.utils.pathing import get_platform_run_path


class FakeRun:
    """A ``prime_runs.Run`` recording the lifecycle calls a monitor can make."""

    def __init__(self):
        self.id, self.url = "run-123", "https://platform/training/run-123"
        self.attached = False
        self._atexit_hook = lambda: None
        self.metrics, self.episodes = [], []
        self.flushed, self.finishes = [], []

    def flush(self, timeout=None):
        self.flushed.append(timeout)

    def finish(self, *args, **kwargs):
        self.finishes.append((args, kwargs))

    def log_metrics(self, values, *, step=None):
        self.metrics.append((values, step))

    def log_episodes(self, episodes):
        self.episodes.append(list(episodes))


@pytest.fixture
def prime(monkeypatch):
    """Records ``pr.init`` kwargs and unregistered atexit hooks."""
    seen = SimpleNamespace(calls=[], unregistered=[])

    def fake_init(**kwargs):
        seen.calls.append(kwargs)
        return FakeRun()

    monkeypatch.setattr("prime_rl.monitors.prime.pr.init", fake_init)
    monkeypatch.setattr("prime_rl.monitors.prime.atexit.unregister", seen.unregistered.append)
    monkeypatch.delenv("RUN_ID", raising=False)
    return seen


def _init(monitor, tmp_path=None):
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))


def _train_record(tmp_path):
    write_platform_record(tmp_path, {"kind": "train", "id": "run-123", "url": "https://platform/training/run-123"})


@pytest.mark.parametrize("paused", [False, True])
def test_hosted_attach_owns_nothing(prime, monkeypatch, paused):
    monkeypatch.setenv("RUN_ID", "run-123")
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    _init(monitor)

    (call,) = prime.calls
    assert call["id"] == "run-123"
    assert not set(call) & {"name", "model", "environments", "training", "config"}
    assert prime.unregistered == [monitor.run._atexit_hook]

    asyncio.run(monitor.log_episodes([SimpleNamespace()], step=1, kind="train", subset="effective"))
    asyncio.run(monitor.finalize(paused=paused))
    assert monitor.run.episodes == []
    assert monitor.run.flushed == [60.0]
    assert monitor.run.finishes == []


def test_selfhost_defers_until_record_appears(prime, monkeypatch, tmp_path):
    monkeypatch.setattr("prime_rl.monitors.prime.time.time", lambda: 100.0)
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    _init(monitor, tmp_path)
    assert monitor.run is None
    assert prime.calls == []

    row = {"time/step": 1.0, "time/save_ckpt": 2.0, "optim/lr": 3.0}
    path = get_platform_run_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    asyncio.run(monitor.log_metrics(row, step=1))
    assert row == {"time/step": 1.0, "time/save_ckpt": 2.0, "optim/lr": 3.0}  # caller's dict untouched
    write_platform_record(tmp_path, {"kind": "eval", "run_id": "r", "evaluations": {}})
    asyncio.run(monitor.log_metrics({"optim/lr": 2.0}, step=2))
    assert monitor.run is None

    _train_record(tmp_path)
    asyncio.run(monitor.log_metrics({"optim/lr": 3.0}, step=3))
    assert prime.calls[0]["id"] == "run-123"
    assert [step for _, step in monitor.run.metrics] == [1, 2, 3]
    queued = monitor.run.metrics[0][0]
    assert "time/step" not in queued and "time/save_ckpt" not in queued  # excluded even while queued
    assert queued["_timestamp"] == 100.0  # a queued row keeps its own time


def test_metrics_only_drops_orchestrator_owned_keys(prime, monkeypatch):
    monkeypatch.setenv("RUN_ID", "run-123")
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    _init(monitor)

    asyncio.run(
        monitor.log_metrics(
            {"time/step": 1.0, "time/save_ckpt": 2.0, "time/wait_for_batch": 3.0, "optim/lr": 4.0}, step=5
        )
    )
    asyncio.run(monitor.log_metrics({"time/save_ckpt": 1.0}, step=6))  # nothing left to upload
    assert monitor.run.metrics == [({"time/wait_for_batch": 3.0, "optim/lr": 4.0}, 5)]


def test_selfhost_record_missing_warns_once(prime, monkeypatch, tmp_path):
    monkeypatch.setattr("prime_rl.monitors.prime.UNATTACHED_WARN_SECONDS", 0.0)
    monkeypatch.setattr("prime_rl.monitors.prime.PENDING_METRICS_LIMIT", 2)
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    _init(monitor, tmp_path)

    for step in range(1, 4):
        asyncio.run(monitor.log_metrics({"optim/lr": float(step)}, step=step))
    assert monitor.warned_unattached
    assert [step for step, _ in monitor.pending] == [2, 3]  # bounded, oldest dropped

    asyncio.run(monitor.finalize())
    assert monitor.run is None
    assert prime.calls == []


def test_finalize_unattached_warns(prime, tmp_path):
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    _init(monitor, tmp_path)
    asyncio.run(monitor.finalize())
    assert monitor.warned_unattached


def test_attach_failure_disables_sink(monkeypatch, tmp_path):
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("attach failed")

    monkeypatch.setattr("prime_rl.monitors.prime.pr.init", fake_init)
    monkeypatch.delenv("RUN_ID", raising=False)
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    _init(monitor, tmp_path)

    asyncio.run(monitor.log_metrics({"optim/lr": 1.0}, step=1))
    _train_record(tmp_path)
    asyncio.run(monitor.log_metrics({"optim/lr": 2.0}, step=2))

    assert monitor.attach_gave_up
    assert not monitor.pending

    asyncio.run(monitor.log_metrics({"optim/lr": 3.0}, step=3))
    asyncio.run(monitor.finalize())
    assert len(calls) == 1  # one attempt, no retries


def test_disabled_mode_inits_without_record(prime, monkeypatch, tmp_path):
    monkeypatch.setenv("PRIME_RUNS_MODE", "disabled")
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    _init(monitor, tmp_path)

    (call,) = prime.calls
    assert call["mode"] == "disabled"
    assert "id" not in call


def test_non_trainer_keeps_owner_semantics(prime):
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=None, output_dir=None))

    assert monitor.metrics_only is False
    assert prime.calls[0]["model"] == "unknown"

    timings = {"time/step": 1.0, "time/save_ckpt": 2.0, "optim/lr": 3.0}
    asyncio.run(monitor.log_metrics(timings, step=5))
    asyncio.run(monitor.finalize())

    assert monitor.run.metrics == [(timings, 5)]  # the owner path keeps the raw keys
    assert monitor.run.finishes  # it finishes; it is not a flush-only sidecar
