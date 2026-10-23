import time

import pytest

from prime_rl.entrypoints.dashboard import log_run_links, run_links
from prime_rl.utils.links import ATTEMPT_VAR, publish_link, register_links, wait_for_links


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs" / "attempt_2"
    monkeypatch.setenv(ATTEMPT_VAR, str(log_dir))
    return log_dir


def test_waits_for_every_registered_link(tmp_path, attempt):
    register_links(tmp_path, "prime/eval", 2)
    register_links(tmp_path, "wandb", 1)
    publish_link(tmp_path, "wandb", "run", "https://wandb.ai/run")
    publish_link(tmp_path, "prime/eval", "gsm8k", "https://prime/ev-1")
    assert wait_for_links(tmp_path, attempt, ["prime/eval", "wandb"], lambda: False, timeout=0.0) == {
        "prime/eval": {"gsm8k": "https://prime/ev-1"},
        "wandb": {"run": "https://wandb.ai/run"},
    }

    # The second evaluation failed to open: its None resolves the wait without a link.
    publish_link(tmp_path, "prime/eval", "aime", None)
    start = time.monotonic()
    links = wait_for_links(tmp_path, attempt, ["prime/eval", "wandb"], lambda: False, timeout=5.0)
    assert time.monotonic() - start < 1.0
    assert links["prime/eval"] == {"gsm8k": "https://prime/ev-1"}


def test_monitor_without_pages_does_not_stall(tmp_path, attempt):
    """A monitor that opens no page (no credentials) registers zero links."""
    register_links(tmp_path, "prime", 0)
    start = time.monotonic()
    assert wait_for_links(tmp_path, attempt, ["prime"], lambda: False, timeout=5.0) == {"prime": {}}
    assert time.monotonic() - start < 1.0


def test_previous_attempt_links_never_count(tmp_path, attempt, monkeypatch):
    monkeypatch.setenv(ATTEMPT_VAR, str(tmp_path / "logs" / "attempt_1"))
    register_links(tmp_path, "prime", 1)
    publish_link(tmp_path, "prime", "run", "https://prime/old")
    monkeypatch.setenv(ATTEMPT_VAR, str(attempt))
    assert wait_for_links(tmp_path, attempt, ["prime"], lambda: True) == {}

    register_links(tmp_path, "prime", 1)
    publish_link(tmp_path, "prime", "run", "https://prime/new")
    assert wait_for_links(tmp_path, attempt, ["prime"], lambda: False, timeout=0.0) == {
        "prime": {"run": "https://prime/new"}
    }


def test_shared_registration_keeps_published_links(tmp_path, attempt):
    """Processes sharing one W&B run register the same entry, in any order."""
    register_links(tmp_path, "wandb", 1)
    publish_link(tmp_path, "wandb", "run", "https://wandb.ai/run")
    register_links(tmp_path, "wandb", 1)
    assert wait_for_links(tmp_path, attempt, ["wandb"], lambda: False, timeout=0.0) == {
        "wandb": {"run": "https://wandb.ai/run"}
    }


def test_stop_ends_the_wait(tmp_path, attempt):
    start = time.monotonic()
    assert wait_for_links(tmp_path, attempt, ["prime"], lambda: True, timeout=5.0) == {}
    assert time.monotonic() - start < 1.0


def test_run_links_skips_monitors_that_open_no_page(tmp_path, attempt, monkeypatch):
    monkeypatch.setenv("PRIME_RUNS_MODE", "disabled")
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert run_links(tmp_path, attempt, {"prime": object(), "wandb": object()}, lambda: False) is None
    monkeypatch.delenv("WANDB_MODE")
    assert run_links(tmp_path, attempt, {"prime": object(), "wandb": None}, lambda: False) is None
    assert run_links(tmp_path, attempt, {"prime": object(), "wandb": object()}, lambda: False) is not None


def test_banner_lists_each_link(tmp_path, attempt, capsys):
    from prime_rl.utils.logger import reset_logger, setup_logger

    reset_logger()
    setup_logger("info", console_level="ERROR")
    links = {"prime/eval": {"gsm8k": "https://prime/ev-1", "aime": "https://prime/ev-2"}, "wandb": {"run": "https://w"}}
    log_run_links("http://localhost:7788", lambda: links)
    out = capsys.readouterr().out
    reset_logger()
    assert "Dashboard" in out and "http://localhost:7788" in out
    assert "https://prime/ev-1" in out and "(gsm8k)" in out and "(aime)" in out
    assert "W&B" in out and "https://w" in out and "(run)" not in out


@pytest.fixture
def no_prime_credentials(monkeypatch):
    from types import SimpleNamespace

    import prime_rl.monitors.prime as prime_module

    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("PRIME_RUNS_MODE", raising=False)
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setattr(prime_module, "PrimeConfig", lambda: SimpleNamespace(api_key=None))


def test_prime_monitors_without_credentials_fail_informatively(tmp_path, no_prime_credentials):
    """Like W&B: a configured prime monitor without credentials fails the run, and says
    how to authenticate or turn it off."""
    import asyncio

    from prime_rl.configs.monitors import PrimeEvalMonitorConfig, PrimeTrainMonitorConfig
    from prime_rl.monitors.prime import PrimeEvalMonitor, PrimeTrainMonitor

    for monitor in (PrimeTrainMonitor(PrimeTrainMonitorConfig()), PrimeEvalMonitor(PrimeEvalMonitorConfig())):
        with pytest.raises(RuntimeError, match=r"prime login.*--no-monitors\.prime"):
            asyncio.run(monitor.init(config=None, output_dir=tmp_path))


def test_prime_registration_failure_fails_informatively(tmp_path, monkeypatch):
    import asyncio

    import prime_rl.monitors.prime as prime_module
    from prime_rl.configs.monitors import PrimeTrainMonitorConfig
    from prime_rl.monitors.prime import PrimeTrainMonitor

    def no_team(**kwargs):
        raise prime_module.pr.ConfigurationError("A training run needs a team")

    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    monkeypatch.delenv("PRIME_RUNS_MODE", raising=False)
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setattr(prime_module.pr, "init", no_team)
    with pytest.raises(RuntimeError, match=r"needs a team.*prime switch.*--no-monitors\.prime"):
        asyncio.run(PrimeTrainMonitor(PrimeTrainMonitorConfig()).init(config=None, output_dir=tmp_path))


def test_disabled_prime_monitors_register_no_links(tmp_path, attempt, no_prime_credentials, monkeypatch):
    """$PRIME_RUNS_MODE=disabled: the monitors run, register zero links, and the launcher
    does not wait on them."""
    import asyncio

    from prime_rl.configs.monitors import PrimeEvalMonitorConfig, PrimeTrainMonitorConfig
    from prime_rl.monitors.prime import PrimeEvalMonitor, PrimeTrainMonitor

    monkeypatch.setenv("PRIME_RUNS_MODE", "disabled")
    assert run_links(tmp_path, attempt, {"prime": object(), "wandb": None}, lambda: False) is None
    asyncio.run(PrimeTrainMonitor(PrimeTrainMonitorConfig()).init(config=None, output_dir=tmp_path))
    asyncio.run(PrimeEvalMonitor(PrimeEvalMonitorConfig()).init(config=None, output_dir=tmp_path))
    start = time.monotonic()
    assert wait_for_links(tmp_path, attempt, ["prime", "prime/eval"], lambda: False, timeout=5.0) == {
        "prime": {},
        "prime/eval": {},
    }
    assert time.monotonic() - start < 1.0


def test_wandb_without_credentials_fails_informatively(tmp_path, monkeypatch):
    import asyncio

    import prime_rl.monitors.wandb.monitor as wandb_module
    from prime_rl.configs.monitors import WandbMonitorConfig
    from prime_rl.monitors.wandb import WandbMonitor

    def no_key(**kwargs):
        raise wandb_module.UsageError("No API key configured. Use `wandb login` to log in.")

    for var in ("WANDB_SHARED_MODE", "WANDB_MODE", "WANDB_ARGS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(wandb_module.wandb, "init", no_key)
    with pytest.raises(RuntimeError, match=r"No API key configured.*--no-monitors\.wandb"):
        asyncio.run(WandbMonitor(WandbMonitorConfig()).init(output_dir=tmp_path))
