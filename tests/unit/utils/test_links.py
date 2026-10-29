import asyncio
import os
import time
from types import SimpleNamespace

import pytest

import prime_rl.monitors.prime as prime_module
from prime_rl.configs.monitors import PrimeMonitorConfig
from prime_rl.monitors.prime import PrimeMonitor, launch_run
from prime_rl.utils.links import ATTEMPT_VAR, publish_link, register_links, wait_for_links


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs" / "attempt_2"
    monkeypatch.setenv(ATTEMPT_VAR, str(log_dir))
    return log_dir


def test_waits_for_every_registered_link(tmp_path, attempt):
    register_links(tmp_path, "prime", 0)
    register_links(tmp_path, "prime/eval", 2)
    register_links(tmp_path, "wandb", 1)
    publish_link(tmp_path, "wandb", "run", "https://wandb.ai/run")
    publish_link(tmp_path, "prime/eval", "gsm8k", "https://prime/ev-1")
    # A page that failed to open publishes None, which resolves the wait without a link.
    publish_link(tmp_path, "prime/eval", "aime", None)
    start = time.monotonic()
    assert wait_for_links(tmp_path, attempt, ["prime", "prime/eval", "wandb"], lambda: False, timeout=5.0) == {
        "prime": {},
        "prime/eval": {"gsm8k": "https://prime/ev-1"},
        "wandb": {"run": "https://wandb.ai/run"},
    }
    assert time.monotonic() - start < 1.0


def test_previous_attempt_links_never_count(tmp_path, attempt, monkeypatch):
    monkeypatch.setenv(ATTEMPT_VAR, str(tmp_path / "logs" / "attempt_1"))
    register_links(tmp_path, "prime", 1)
    publish_link(tmp_path, "prime", "run", "https://prime/old")
    monkeypatch.setenv(ATTEMPT_VAR, str(attempt))
    assert wait_for_links(tmp_path, attempt, ["prime"], lambda: True) == {}


@pytest.fixture
def no_prime_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("PRIME_API_KEY", "PRIME_RUNS_MODE", "RUN_ID"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("prime_mode", [None, "online"])
def test_prime_monitor_without_credentials_runs_disabled(tmp_path, attempt, monkeypatch, no_prime_key, prime_mode):
    """It warns and opens no links, so the launcher's banner does not wait on it."""
    if prime_mode is not None:
        monkeypatch.setenv("PRIME_RUNS_MODE", prime_mode)
    warnings: list[str] = []
    monkeypatch.setattr(prime_module, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    monitor = PrimeMonitor(PrimeMonitorConfig())
    asyncio.run(monitor.init(config=None, output_dir=tmp_path))
    asyncio.run(monitor.finalize())
    assert monitor.run.url is None
    assert len(warnings) == 1 and "prime login" in warnings[0] and "--no-monitors.prime" in warnings[0]
    assert wait_for_links(tmp_path, attempt, ["prime"], lambda: False, timeout=0.0) == {"prime": {}}


def test_launch_without_credentials_disables_its_components(monkeypatch, no_prime_key):
    """The launcher warns once, and its components run the monitor disabled instead
    of attaching to a run that does not exist."""
    monkeypatch.setattr(prime_module, "get_logger", lambda: SimpleNamespace(warning=lambda _: None))
    with launch_run(None, PrimeMonitorConfig()):
        assert os.environ["PRIME_RUNS_MODE"] == "disabled"
        assert "RUN_ID" not in os.environ


def test_attached_prime_monitor_without_credentials_fails(tmp_path, monkeypatch, no_prime_key):
    monkeypatch.setenv("RUN_ID", "launcher-run")
    with pytest.raises(Exception, match="API key"):
        asyncio.run(PrimeMonitor(PrimeMonitorConfig()).init(config=None, output_dir=tmp_path))
