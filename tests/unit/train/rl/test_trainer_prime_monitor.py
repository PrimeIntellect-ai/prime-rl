"""The RL trainer logs its metrics to the Prime run that the ``rl`` launcher registers."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from prime_rl.configs.rl import RLConfig
from prime_rl.monitors.prime import PrimeMonitor, launch_run


@pytest.fixture
def sdk_runs(monkeypatch):
    """Fake ``pr.init`` runs that record their kwargs, metrics, and how they ended."""
    runs = []

    def fake_init(**kwargs):
        run = SimpleNamespace(id=kwargs.get("id", "run-123"), url="https://x/run-123", attached="id" in kwargs)
        run.kwargs, run.metrics, run.finished, run.flushed, run.failed = kwargs, [], False, False, None
        run._atexit_hook = lambda: None
        run.log_metrics = lambda metrics, step=None: run.metrics.append((step, metrics))
        run.finish = lambda *a, **k: setattr(run, "finished", True)
        run.flush = lambda *a, **k: setattr(run, "flushed", True)
        run.fail = lambda error, **k: setattr(run, "failed", error)
        runs.append(run)
        return run

    monkeypatch.setattr("prime_rl.monitors.prime.pr.init", fake_init)
    for var in ("RUN_ID", "PRIME_RUNS_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    monkeypatch.setenv("PRL_RUN_ID", "local-uuid")
    return runs


def _rl_config(overrides: dict | None = None) -> RLConfig:
    return RLConfig.model_validate(
        {
            "model": {"name": "PrimeIntellect/Qwen3-0.6B"},
            "trainer": {},
            "orchestrator": {"renderer": {"name": "default"}},
        }
        | (overrides or {})
    )


@pytest.mark.parametrize("prime", [{}, {"name": "shared"}, "None"])
def test_shared_prime_monitor_reaches_the_trainer(prime):
    config = _rl_config({"monitors": {"prime": prime}})
    for component in (config.trainer, config.orchestrator):
        assert (component.monitors.prime is None) == (prime == "None")
        if component.monitors.prime is not None:
            assert component.monitors.prime.name == prime.get("name", config.run.name)


def test_trainer_and_orchestrator_share_the_launchers_run(sdk_runs, tmp_path, monkeypatch):
    """Both components attach to the run the launcher registered. The trainer's metrics
    land on it, and only the orchestrator finishes it."""
    config = _rl_config()
    trainer, orchestrator = (
        PrimeMonitor(config.trainer.monitors.prime),
        PrimeMonitor(config.orchestrator.monitors.prime),
    )
    with launch_run(config.orchestrator, config.orchestrator.monitors.prime):
        launched = sdk_runs[0]
        asyncio.run(trainer.init(config=config.trainer, output_dir=tmp_path, producer="trainer"))
        asyncio.run(orchestrator.init(config=config.orchestrator, output_dir=tmp_path, producer="orchestrator"))
        assert trainer.run.attached and orchestrator.run.attached
        assert trainer.run.id == orchestrator.run.id == launched.id == os.environ["RUN_ID"]

        asyncio.run(trainer.log_metrics({"time/trainer_step": 1.5, "optim/grad_norm": 0.3}, step=1))
        assert trainer.run.metrics == [(1, {"time/trainer_step": 1.5, "optim/grad_norm": 0.3})]

        asyncio.run(trainer.finalize())
        assert trainer.run.flushed and not trainer.run.finished
        asyncio.run(orchestrator.finalize())
        assert orchestrator.run.finished
    assert launched.failed is None


def test_trainer_without_a_launcher_run_fails_informatively(sdk_runs, tmp_path):
    config = _rl_config()
    with pytest.raises(RuntimeError, match=r"launch it with `rl`.*--no-monitors\.prime"):
        asyncio.run(PrimeMonitor(config.trainer.monitors.prime).init(config=config.trainer, output_dir=tmp_path))
    assert sdk_runs == []


def test_disabled_trainer_registers_nothing(sdk_runs, tmp_path, monkeypatch):
    monkeypatch.setenv("PRIME_RUNS_MODE", "disabled")
    config = _rl_config()
    monitor = PrimeMonitor(config.trainer.monitors.prime)
    asyncio.run(monitor.init(config=config.trainer, output_dir=tmp_path, producer="trainer"))
    assert sdk_runs[0].kwargs["mode"] == "disabled" and "environments" not in sdk_runs[0].kwargs
