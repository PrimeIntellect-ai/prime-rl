import asyncio
import time
from types import SimpleNamespace

import pytest

import prime_rl.monitors.prime as prime_module
from prime_rl.configs.monitors import PrimeTrainMonitorConfig, WandbMonitorConfig
from prime_rl.monitors.prime import PrimeTrainMonitor
from prime_rl.monitors.wandb import WandbMonitor
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


@pytest.mark.parametrize("prime_mode", [None, "online"])
def test_monitors_without_credentials_run_disabled(tmp_path, attempt, monkeypatch, prime_mode):
    """No Prime or W&B credentials: both monitors warn, run disabled, and open no links,
    so the launcher's banner does not wait on them."""
    for var in ("PRIME_API_KEY", "PRIME_RUNS_MODE", "RUN_ID", "WANDB_API_KEY", "WANDB_MODE"):
        monkeypatch.delenv(var, raising=False)
    if prime_mode is not None:
        monkeypatch.setenv("PRIME_RUNS_MODE", prime_mode)
    monkeypatch.setenv("NETRC", str(tmp_path / "missing-netrc"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb-config"))
    monkeypatch.setattr(prime_module, "PrimeConfig", lambda: SimpleNamespace(api_key=None))
    warnings: list[str] = []
    logger = SimpleNamespace(warning=warnings.append, info=print, debug=print)
    monkeypatch.setattr(prime_module, "get_logger", lambda: logger)
    prime = PrimeTrainMonitor(PrimeTrainMonitorConfig())
    wandb = WandbMonitor(WandbMonitorConfig())
    wandb.logger = logger

    asyncio.run(prime.init(config=None, output_dir=tmp_path))
    asyncio.run(wandb.init(output_dir=tmp_path))

    asyncio.run(prime.finalize())
    asyncio.run(wandb.finalize())
    assert prime.run.url is None
    assert [("prime login" in w, "wandb login" in w) for w in warnings] == [(True, False), (False, True)]
    assert wait_for_links(tmp_path, attempt, ["prime", "wandb"], lambda: False, timeout=0.0) == {
        "prime": {},
        "wandb": {},
    }
