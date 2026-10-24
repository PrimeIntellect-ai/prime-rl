import asyncio
import time
from types import SimpleNamespace

import pytest

from prime_rl import monitors
from prime_rl.configs.monitors import PrimeTrainMonitorConfig, WandbMonitorConfig
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


def test_monitors_without_credentials_are_skipped(tmp_path, attempt, monkeypatch):
    """No Prime or W&B credentials: setup warns, skips both monitors, and registers them
    with no links, so the launcher's banner does not wait on them."""
    for var in ("PRIME_API_KEY", "PRIME_RUNS_MODE", "WANDB_API_KEY", "WANDB_IDENTITY_TOKEN_FILE", "WANDB_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "missing-netrc"))
    monkeypatch.setattr("prime_sandboxes.Config", lambda: SimpleNamespace(api_key=None))
    warnings: list[str] = []
    monkeypatch.setattr(monitors, "get_logger", lambda: SimpleNamespace(warning=warnings.append, info=print))

    asyncio.run(monitors.setup(prime=PrimeTrainMonitorConfig(), wandb=WandbMonitorConfig(), output_dir=tmp_path))

    assert monitors.MONITORS == []
    assert any("prime login" in w and "--no-monitors.prime" in w for w in warnings)
    assert any("wandb login" in w and "--no-monitors.wandb" in w for w in warnings)
    assert wait_for_links(tmp_path, attempt, ["prime", "wandb"], lambda: False, timeout=0.0) == {
        "prime": {},
        "wandb": {},
    }
