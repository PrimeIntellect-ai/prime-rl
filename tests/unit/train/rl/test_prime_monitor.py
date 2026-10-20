"""The RL trainer's platform wiring: a metrics-only attach on a run it does not own.

``monitors.prime`` on ``TrainerConfig`` must never register or finalize a platform
run: the trainer attaches to the run the managed launch injected (``$RUN_ID``),
uploads stepped metrics, and only flushes on finalize — the run's owner (the
orchestrator, or the platform) alone finishes it.
"""

import asyncio
from types import SimpleNamespace

import pytest

from prime_rl import monitors
from prime_rl.configs.monitors import PrimeTrainMonitorConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.monitors.prime import PrimeTrainMonitor


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


def _metrics_only_monitor():
    return PrimeTrainMonitor(PrimeTrainMonitorConfig(), metrics_only=True)


def test_trainer_config_has_prime_monitor_field():
    config = TrainerConfig.model_validate({})

    assert config.monitors.prime is None

    config = TrainerConfig.model_validate({"monitors": {"prime": {"name": "my-run"}}})

    assert isinstance(config.monitors.prime, PrimeTrainMonitorConfig)
    assert config.monitors.prime.name == "my-run"


def test_metrics_only_attaches_without_registration(prime_init, unregistered_atexit, run_id):
    monitor = _metrics_only_monitor()
    asyncio.run(monitor.init(config=TrainerConfig.model_validate({}), output_dir=None))

    (call,) = prime_init
    assert call["kind"] == "train"
    assert call["id"] == "run-123"
    # An attach carries no registration payload: no name, model, environments, or spec.
    assert not set(call) & {"name", "model", "environments", "training", "config"}
    # The trainer is one writer among several: its process exiting must never
    # report the shared run crashed.
    assert unregistered_atexit == [monitor.run._atexit_hook]


def test_metrics_only_requires_run_id(prime_init, unregistered_atexit):
    monitor = _metrics_only_monitor()

    with pytest.raises(RuntimeError, match=r"\$RUN_ID"):
        asyncio.run(monitor.init(config=None, output_dir=None))

    assert prime_init == []


def test_metrics_only_finalize_flushes_and_never_finishes(prime_init, run_id):
    monitor = _metrics_only_monitor()
    asyncio.run(monitor.init(config=None, output_dir=None))

    asyncio.run(monitor.finalize())
    asyncio.run(monitor.finalize(paused=True))

    assert monitor.run.flushed == [60.0, 60.0]
    assert monitor.run.finishes == []


def test_metrics_only_drops_episodes(prime_init, run_id):
    monitor = _metrics_only_monitor()
    asyncio.run(monitor.init(config=None, output_dir=None))

    asyncio.run(monitor.log_episodes([SimpleNamespace()], step=1, kind="train", subset="effective"))

    assert monitor.run.episodes == []


def test_metrics_only_still_logs_stepped_metrics(prime_init, run_id):
    monitor = _metrics_only_monitor()
    asyncio.run(monitor.init(config=None, output_dir=None))

    asyncio.run(monitor.log_metrics({"optim/lr": 1e-4}, step=3))

    assert monitor.run.metrics == [({"optim/lr": 1e-4}, 3)]


def test_trainer_setup_wires_metrics_only(prime_init, run_id, clear_monitors):
    """The RL trainer's ``monitors.setup`` call attaches metrics-only when the
    config enables the prime sink — the SFT path (no flag) is unaffected."""
    asyncio.run(
        monitors.setup(
            prime=TrainerConfig.model_validate({"monitors": {"prime": {}}}).monitors.prime,
            prime_metrics_only=True,
            file=None,
            output_dir=None,
        )
    )

    (monitor,) = monitors.MONITORS
    assert isinstance(monitor, PrimeTrainMonitor)
    assert monitor.metrics_only


def test_owner_flavored_monitor_is_unchanged(prime_init, monkeypatch):
    """Without the flag the monitor keeps the owner semantics: it registers (or
    attaches via $RUN_ID) with its config kwargs and finalizes the run."""
    monkeypatch.setenv("RUN_ID", "run-123")
    monitor = PrimeTrainMonitor(PrimeTrainMonitorConfig())

    assert monitor.metrics_only is False
