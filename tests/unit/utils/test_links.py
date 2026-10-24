import asyncio
import time
from types import SimpleNamespace

import pytest

import prime_rl.monitors.prime as prime_module
from prime_rl.configs.monitors import PrimeTrainMonitorConfig
from prime_rl.monitors.prime import PrimeTrainMonitor
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


def test_prime_monitor_without_credentials_fails_informatively(tmp_path, monkeypatch):
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("PRIME_RUNS_MODE", raising=False)
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setattr(prime_module, "PrimeConfig", lambda: SimpleNamespace(api_key=None))
    with pytest.raises(RuntimeError, match=r"prime login.*--no-monitors\.prime"):
        asyncio.run(PrimeTrainMonitor(PrimeTrainMonitorConfig()).init(config=None, output_dir=tmp_path))
