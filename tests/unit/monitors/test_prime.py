import asyncio
from types import SimpleNamespace

import prime_runs as pr
import pytest

from prime_rl.configs.monitors import PrimeMonitorConfig
from prime_rl.monitors.prime import FINISH_TIMEOUT, PrimeMonitor, _base_url


@pytest.fixture
def init_calls(monkeypatch):
    """Record what the monitor asks the SDK for; hand back a disabled run."""
    real_init, calls, runs = pr.init, [], []

    def fake_init(**kwargs):
        calls.append(kwargs)
        runs.append(real_init(kind="train", mode="disabled", id=kwargs.get("id"), model="m"))
        return runs[-1]

    monkeypatch.setattr(pr, "init", fake_init)
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.delenv(pr.MODE_ENV, raising=False)
    yield calls
    for run in runs:  # else the SDK's atexit hook reports them crashed
        run.finish()


class FakeConfig(SimpleNamespace):
    def model_dump(self, **kwargs):
        return {"max_steps": self.max_steps}


def orchestrator_config(wandb=None):
    return FakeConfig(
        model=SimpleNamespace(name="Qwen/Qwen3-8B"),
        train=SimpleNamespace(source=[SimpleNamespace(env_id="primeintellect/gsm8k")]),
        max_steps=100,
        batch_size=64,
        group_size=8,
        seq_len=4096,
        monitors=SimpleNamespace(wandb=wandb),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("https://api.primeintellect.ai/api/v1/rft", "https://api.primeintellect.ai/api/v1"),
        ("https://api.primeintellect.ai/api/v1/rft/", "https://api.primeintellect.ai/api/v1"),
        ("https://api.primeintellect.ai", "https://api.primeintellect.ai"),
    ],
)
def test_base_url_strips_the_rft_root(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("PRIME_API_BASE", raising=False)
    else:
        monkeypatch.setenv("PRIME_API_BASE", value)
    assert _base_url() == expected


def test_init_registers_the_run_from_the_orchestrator_config(init_calls):
    monitor = PrimeMonitor(PrimeMonitorConfig(name="exp"))

    asyncio.run(monitor.init(orchestrator_config(wandb=SimpleNamespace(project="proj"))))

    assert init_calls == [
        dict(
            kind="train",
            mode="online",
            base_url=None,
            finish_timeout=FINISH_TIMEOUT,
            name="exp",
            model="Qwen/Qwen3-8B",
            environments=["primeintellect/gsm8k"],
            training=pr.TrainingSpec(
                max_steps=100, batch_size=64, rollouts_per_example=8, seq_len=4096, wandb_project="proj"
            ),
            config={"max_steps": 100},
        )
    ]
    assert monitor.run.kind == "train"


def test_init_attaches_to_the_launcher_s_run(init_calls, monkeypatch):
    monkeypatch.setenv("RUN_ID", "run-managed")
    monitor = PrimeMonitor(PrimeMonitorConfig())

    asyncio.run(monitor.init(orchestrator_config()))

    (call,) = init_calls
    assert call["id"] == "run-managed" and "model" not in call and "training" not in call
    assert monitor.run.id == "run-managed"


def test_the_disabled_switch_reaches_the_sdk(init_calls, monkeypatch):
    monkeypatch.setenv(pr.MODE_ENV, "disabled")

    asyncio.run(PrimeMonitor(PrimeMonitorConfig()).init(orchestrator_config()))

    assert init_calls[0]["mode"] == "disabled"
