"""The RL trainer's platform wiring: a metrics-only attach on a run it does not own.

``monitors.prime`` on ``TrainerConfig`` must never register or finalize a platform
run: the trainer attaches to the run the managed launch injected (``$RUN_ID``) or,
selfhosted, to the orchestrator's registration in the shared platform record, uploads
stepped metrics, and only flushes on finalize — the run's owner (the orchestrator,
or the platform) alone finishes it. A selfhosted record that lags the trainer's
first log calls is attached lazily: metric rows queue bounded until it appears.
"""

import asyncio
from types import SimpleNamespace

import pytest

from prime_rl import monitors
from prime_rl.configs.monitors import PrimeTrainMonitorConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.monitors.prime import PrimeTrainMonitor, write_platform_record
from prime_rl.utils.pathing import get_platform_run_path


class FakeRun:
    """A ``prime_runs.Run`` that records the lifecycle calls a monitor can make."""

    def __init__(self):
        self.id = "run-123"
        self.url = "https://platform/training/run-123"
        self.attached = True
        self.finished = False
        self._atexit_hook = self._on_process_exit
        self.flushed = []
        self.finishes = []
        self.metrics = []
        self.episodes = []

    def _on_process_exit(self):
        pass

    def flush(self, timeout=None):
        self.flushed.append(timeout)

    def finish(self, *args, **kwargs):
        self.finished = True
        self.finishes.append((args, kwargs))

    def log_metrics(self, values, *, step=None):
        self.metrics.append((values, step))

    def log_episodes(self, episodes):
        self.episodes.append(list(episodes))


@pytest.fixture
def prime_init(monkeypatch):
    """Replace the ``prime_runs`` SDK entry point with one that records its kwargs."""
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return FakeRun()

    monkeypatch.setattr("prime_rl.monitors.prime.pr.init", fake_init)
    monkeypatch.delenv("RUN_ID", raising=False)
    return calls


@pytest.fixture
def unregistered_atexit(monkeypatch):
    """Record the atexit hooks the monitor unregisters."""
    unregistered = []
    monkeypatch.setattr("prime_rl.monitors.prime.atexit.unregister", unregistered.append)
    return unregistered


@pytest.fixture
def run_id(monkeypatch):
    monkeypatch.setenv("RUN_ID", "run-123")


@pytest.fixture
def clear_monitors():
    """``monitors.setup`` registers into a module-global list; empty it after each test."""
    yield
    monitors.MONITORS.clear()


def test_trainer_config_has_prime_monitor_field():
    config = TrainerConfig.model_validate({})

    assert config.monitors.prime is None

    config = TrainerConfig.model_validate({"monitors": {"prime": {"name": "my-run"}}})

    assert isinstance(config.monitors.prime, PrimeTrainMonitorConfig)
    assert config.monitors.prime.name == "my-run"


def test_metrics_only_attaches_without_registration(prime_init, unregistered_atexit, run_id):
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=None))

    (call,) = prime_init
    assert call["kind"] == "train"
    assert call["id"] == "run-123"
    # An attach carries no registration payload: no name, model, environments, or spec.
    assert not set(call) & {"name", "model", "environments", "training", "config"}
    # The trainer is one writer among several: its process exiting must never
    # report the shared run crashed.
    assert unregistered_atexit == [monitor.run._atexit_hook]


def test_metrics_only_without_run_id_defers_attach(prime_init, unregistered_atexit, tmp_path):
    """No $RUN_ID and no record yet: the attach defers, and nothing is registered."""
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))

    assert monitor.run is None
    assert prime_init == []
    assert unregistered_atexit == []


def test_selfhost_record_attaches_at_init(prime_init, unregistered_atexit, tmp_path):
    """A record the orchestrator already wrote is attached to right away."""
    write_platform_record(tmp_path, {"kind": "train", "id": "run-123", "url": "https://platform/training/run-123"})
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))

    (call,) = prime_init
    assert call["kind"] == "train"
    assert call["id"] == "run-123"
    assert not set(call) & {"name", "model", "environments", "training", "config"}
    assert unregistered_atexit == [monitor.run._atexit_hook]


def test_selfhost_record_attaches_lazily_and_flushes(prime_init, monkeypatch, tmp_path):
    """A record that lags the trainer's first log calls queues them; the attach
    flushes the queue in order, keeping each row its own timestamp."""
    monkeypatch.setattr("prime_rl.monitors.prime.time.time", lambda: 100.0)
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))

    row = {"optim/lr": 1.0}
    asyncio.run(monitor.log_metrics(row, step=1))
    assert monitor.run is None
    assert "_timestamp" not in row  # the queued copy is stamped, not the caller's dict
    write_platform_record(tmp_path, {"kind": "train", "id": "run-123", "url": "https://platform/training/run-123"})

    asyncio.run(monitor.log_metrics({"optim/lr": 2.0}, step=2))

    (call,) = prime_init
    assert call["id"] == "run-123"
    assert [step for _, step in monitor.run.metrics] == [1, 2]
    assert monitor.run.metrics[0][0]["_timestamp"] == 100.0  # its original queue time
    assert "_timestamp" not in monitor.run.metrics[1][0]


def test_selfhost_record_never_appears_warns_once(prime_init, monkeypatch, tmp_path):
    monkeypatch.setattr("prime_rl.monitors.prime.UNATTACHED_WARN_SECONDS", 0.0)
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))

    asyncio.run(monitor.log_metrics({"optim/lr": 1.0}, step=1))
    assert monitor.warned_unattached

    asyncio.run(monitor.log_metrics({"optim/lr": 2.0}, step=2))
    asyncio.run(monitor.finalize())

    assert monitor.run is None
    assert prime_init == []


def test_finalize_unattached_warns(prime_init, tmp_path):
    """A run that ends before the record appears: one warning, no crash."""
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))

    asyncio.run(monitor.finalize())

    assert monitor.run is None
    assert monitor.warned_unattached


def test_selfhost_invalid_or_foreign_record_ignored(prime_init, tmp_path):
    """A corrupt or non-train record is never attached to; rows keep queueing."""
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))

    path = get_platform_run_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    asyncio.run(monitor.log_metrics({"optim/lr": 1.0}, step=1))
    write_platform_record(tmp_path, {"kind": "eval", "run_id": "r", "evaluations": {}})
    asyncio.run(monitor.log_metrics({"optim/lr": 2.0}, step=2))

    assert monitor.run is None
    assert prime_init == []
    assert [step for step, _ in monitor.pending] == [1, 2]


def test_pending_queue_is_bounded_oldest_dropped(prime_init, monkeypatch, tmp_path):
    monkeypatch.setattr("prime_rl.monitors.prime.PENDING_METRICS_LIMIT", 2)
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))

    for step in range(1, 4):
        asyncio.run(monitor.log_metrics({"optim/lr": float(step)}, step=step))

    assert [step for step, _ in monitor.pending] == [2, 3]


def test_attach_failure_disables_sink(monkeypatch, tmp_path):
    """A record whose run cannot be attached disables the sink: one attempt, one
    warning, and the rest of the run trains untracked."""
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        if kwargs.get("id"):
            raise RuntimeError("attach failed")
        return FakeRun()

    monkeypatch.setattr("prime_rl.monitors.prime.pr.init", fake_init)
    monkeypatch.delenv("RUN_ID", raising=False)
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))

    asyncio.run(monitor.log_metrics({"optim/lr": 1.0}, step=1))  # queued
    write_platform_record(tmp_path, {"kind": "train", "id": "run-123", "url": "https://platform/training/run-123"})
    asyncio.run(monitor.log_metrics({"optim/lr": 2.0}, step=2))  # attach fails, sink disabled

    assert monitor.run is None
    assert monitor.attach_gave_up
    assert not monitor.pending

    asyncio.run(monitor.log_metrics({"optim/lr": 3.0}, step=3))
    asyncio.run(monitor.finalize())
    assert len(calls) == 1  # no retry loop, no disabled-handle fallback


def test_disabled_mode_inits_without_record(prime_init, monkeypatch, tmp_path):
    """A disabled platform never writes a record, so the trainer must not wait for
    one: it takes the SDK's disabled handle straight away."""
    monkeypatch.setenv("PRIME_RUNS_MODE", "disabled")
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=tmp_path))

    (call,) = prime_init
    assert call["mode"] == "disabled"
    assert "id" not in call

    asyncio.run(monitor.log_metrics({"optim/lr": 1.0}, step=1))
    assert monitor.run.metrics == [({"optim/lr": 1.0}, 1)]


def test_metrics_only_finalize_flushes_and_never_finishes(prime_init, run_id):
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=None))

    asyncio.run(monitor.finalize())
    asyncio.run(monitor.finalize(paused=True))

    assert monitor.run.flushed == [60.0, 60.0]
    assert monitor.run.finishes == []


def test_metrics_only_drops_episodes(prime_init, run_id):
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=None))

    asyncio.run(monitor.log_episodes([SimpleNamespace()], step=1, kind="train", subset="effective"))

    assert monitor.run.episodes == []


def test_metrics_only_preserves_metric_keys(prime_init, run_id):
    """All monitors receive the trainer's source metric names unchanged."""
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=None))

    asyncio.run(
        monitor.log_metrics(
            {"time/trainer/step": 1.0, "time/trainer/save_ckpt": 2.0, "time/wait_for_batch": 3.0, "optim/lr": 4.0},
            step=5,
        )
    )

    assert monitor.run.metrics == [
        (
            {
                "time/trainer/step": 1.0,
                "time/trainer/save_ckpt": 2.0,
                "time/wait_for_batch": 3.0,
                "optim/lr": 4.0,
            },
            5,
        )
    ]


def test_trainer_setup_wires_metrics_only(prime_init, run_id, clear_monitors):
    """The RL trainer's ``monitors.setup`` call attaches metrics-only when the
    config enables the prime sink; no separate sidecar flag is needed."""
    asyncio.run(
        monitors.setup(
            prime=TrainerConfig.model_validate({"monitors": {"prime": {}}}).monitors.prime,
            file=None,
            output_dir=None,
            run_config=TrainerConfig.model_validate({}),
        )
    )

    (monitor,) = monitors.MONITORS
    assert isinstance(monitor, PrimeTrainMonitor)
    assert monitor.metrics_only


def test_non_trainer_keeps_owner_semantics(prime_init):
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    asyncio.run(monitor.init(config=None, output_dir=None))

    assert monitor.metrics_only is False
    assert prime_init[0]["model"] == "unknown"
