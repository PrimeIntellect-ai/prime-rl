"""The SFT trainer's platform wiring: registration shape and monitor fan-out.

``monitors.prime`` on ``SFTConfig`` must register the run with the dataset-batched
training fields (no environments, no rollouts) and flow the trainer's metrics and
finalize through the registered ``PrimeMonitor`` like the RL orchestrator does.
"""

import asyncio
from types import SimpleNamespace

import pytest

from prime_rl import monitors
from prime_rl.configs.monitors import (
    EvalMonitorsConfig,
    FileMonitorConfig,
    PrimeMonitorConfig,
    TrainMonitorsConfig,
    WandbMonitorConfig,
)
from prime_rl.configs.sft import SFTConfig
from prime_rl.monitors.prime import PrimeMonitor


@pytest.fixture
def prime_init(monkeypatch):
    """Replace the ``prime_runs`` SDK entry point with one that records its kwargs."""
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            id="run-123",
            url="https://platform/training/run-123",
            attached=False,
            finished=False,
            finish=lambda *args, **kwargs_: None,
            log_metrics=lambda *args, **kwargs_: None,
            log_episodes=lambda *args, **kwargs_: None,
        )

    monkeypatch.setattr("prime_rl.monitors.prime.pr.init", fake_init)
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.delenv("PRIME_RUNS_MODE", raising=False)
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    return calls


@pytest.fixture
def clear_monitors():
    """``monitors.setup`` registers into a module-global list; empty it after each test."""
    yield
    monitors.MONITORS.clear()


def _sft_config(overrides: dict | None = None) -> SFTConfig:
    config = {
        "model": {"name": "PrimeIntellect/Qwen3-0.6B"},
        "data": {"type": "fake", "batch_size": 8, "seq_len": 64},
        "max_steps": 10,
    }
    if overrides:
        config.update(overrides)
    return SFTConfig.model_validate(config)


def test_sft_prime_monitor_name_inherits_run_name():
    config = _sft_config({"monitors": {"prime": {}}})

    assert config.monitors.prime.name == config.run.name


def test_sft_prime_monitor_keeps_explicit_name():
    config = _sft_config({"monitors": {"prime": {"name": "my-experiment"}}})

    assert config.monitors.prime.name == "my-experiment"


def test_sft_registers_dataset_training_fields(prime_init, tmp_path):
    config = _sft_config({"monitors": {"prime": {}}})

    asyncio.run(PrimeMonitor(config.monitors.prime).init(config=config, output_dir=tmp_path))

    (call,) = prime_init
    assert call["kind"] == "train"
    assert call["name"] == config.run.name
    assert call["model"] == "PrimeIntellect/Qwen3-0.6B"
    # SFT trains on a dataset: no environments, no rollouts per example
    assert call["environments"] == []
    assert call["training"].max_steps == 10
    assert call["training"].batch_size == 8
    assert call["training"].seq_len == 64
    assert call["training"].rollouts_per_example is None
    assert call["config"]["model"]["name"] == "PrimeIntellect/Qwen3-0.6B"


def test_sft_registers_wandb_project(prime_init, tmp_path):
    config = _sft_config(
        {
            "monitors": {
                "wandb": {"project": "sft-tests"},
                "prime": {},
            }
        }
    )

    asyncio.run(PrimeMonitor(config.monitors.prime).init(config=config, output_dir=tmp_path))

    assert prime_init[0]["training"].wandb_project == "sft-tests"


def test_online_eval_monitors_keep_the_trainers_prime_monitor():
    """The online-eval process logs to the trainer's Prime run, under the same config."""
    from prime_rl.entrypoints.sft import build_online_eval_monitors

    monitors = build_online_eval_monitors(
        TrainMonitorsConfig(
            wandb=WandbMonitorConfig(project="sft-tests"),
            file=FileMonitorConfig(),
            prime=PrimeMonitorConfig(name="my-run"),
        )
    )

    assert isinstance(monitors, EvalMonitorsConfig)
    assert monitors.prime == PrimeMonitorConfig(name="my-run")
    assert monitors.wandb.project == "sft-tests"
    assert build_online_eval_monitors(TrainMonitorsConfig(prime=None)).prime is None


def test_sft_setup_registers_prime_monitor(prime_init, tmp_path, clear_monitors):
    config = _sft_config({"monitors": {"prime": {}}})

    asyncio.run(
        monitors.setup(
            prime=config.monitors.prime,
            output_dir=tmp_path,
            run_config=config,
        )
    )

    assert monitors.get(PrimeMonitor) is not None

    # The trainer's per-step metrics fan out to the platform run, and finalize
    # closes it out (drains the SDK's upload queue).
    asyncio.run(monitors.log({"loss/mean": 1.0, "step": 1}, step=1))
    asyncio.run(monitors.log({"val/loss": 0.5, "step": 1}, step=1))
    asyncio.run(monitors.finalize())


def test_online_eval_config_keeps_prime_monitor():
    """SFTOnlineEvalConfig.monitors must be EvalMonitorsConfig so the
    trainer's prime monitor survives the eval.json dump/re-parse boundary
    (base MonitorsConfig drops/forbids prime)."""
    from prime_rl.configs.eval import SFTOnlineEvalConfig
    from prime_rl.configs.monitors import EvalMonitorsConfig
    from prime_rl.entrypoints.sft import build_online_eval_monitors

    field = SFTOnlineEvalConfig.model_fields["monitors"]
    assert field.annotation is EvalMonitorsConfig

    train_monitors = TrainMonitorsConfig.model_validate({"prime": {"name": "r1"}})
    eval_monitors = build_online_eval_monitors(train_monitors)
    assert isinstance(eval_monitors.prime, PrimeMonitorConfig)
    assert eval_monitors.prime.name == "r1"

    # The online-eval process passes the prime monitor to monitors.setup
    # (src/prime_rl/eval/online.py) — same wiring as the standalone eval
    # entrypoint. Read the source text (importing the module pulls torch,
    # which this torch-free test env deliberately lacks).
    from pathlib import Path

    import prime_rl

    online_src = Path(prime_rl.__path__[0], "eval", "online.py").read_text()
    assert "prime=config.monitors.prime" in online_src


def test_online_evals_share_the_trainers_run(tmp_path, monkeypatch):
    """The online-eval process shares the trainer's run dir and attaches to the run the
    trainer registered. Only it finishes the run: it drains its final evals after the
    trainer exits."""
    from prime_rl.entrypoints.sft import build_online_eval_config
    from prime_rl.utils.links import ATTEMPT_VAR
    from prime_rl.utils.process import RUN_FINISHER_VAR

    runs = []

    def fake_init(**kwargs):
        run = SimpleNamespace(id=kwargs.get("id", "run-123"), url="https://x/run-123", attached="id" in kwargs)
        run.kwargs, run.finished, run.flushed = kwargs, False, False
        run._atexit_hook = lambda: None
        run.finish = lambda *a, **k: setattr(run, "finished", True)
        run.flush = lambda *a, **k: setattr(run, "flushed", True)
        runs.append(run)
        return run

    monkeypatch.setattr("prime_rl.monitors.prime.pr.init", fake_init)
    for var in ("RUN_ID", "PRIME_RUNS_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    monkeypatch.setenv(ATTEMPT_VAR, str(tmp_path / "logs" / "attempt_1"))
    monkeypatch.setenv(RUN_FINISHER_VAR, "online-eval")

    config = _sft_config(
        {"eval": {"interval": 50, "source": [{"name": "rev", "env": {"taskset": {"id": "harbor"}}}]}, "inference": {}}
    )
    eval_config = build_online_eval_config(config)
    # Same run dir: dashboard visibility for the eval artifacts.
    assert eval_config.output_dir == config.run_dir

    trainer, online_eval = PrimeMonitor(config.monitors.prime), PrimeMonitor(eval_config.monitors.prime)
    asyncio.run(trainer.init(config=config, output_dir=tmp_path, producer="trainer"))
    asyncio.run(online_eval.init(config=eval_config, output_dir=tmp_path, producer="online-eval"))
    assert online_eval.run.kwargs["id"] == trainer.run.id

    asyncio.run(trainer.finalize())
    assert trainer.run.flushed and not trainer.run.finished
    asyncio.run(online_eval.finalize())
    assert online_eval.run.finished


def test_platform_record_lock_keeps_a_stable_inode(tmp_path):
    """The record lock must NEVER unlink its lock file: a writer waiting on
    the flock can still hold the unlinked inode's lock while a later writer
    creates and locks a fresh file - two simultaneous "exclusive" locks and
    a lost merge. The lock file must persist with the SAME inode across
    acquisitions, and the locked update helper must merge two sequential
    writers into one record. (Interprocess contention is exercised by the
    two-process update_platform_record probe; here the writers are
    sequential.)"""
    import os

    from prime_rl.monitors.prime import update_platform_record
    from prime_rl.utils.pathing import get_platform_run_path
    from prime_rl.utils.record import record_lock

    lock_path = get_platform_run_path(tmp_path).with_suffix(".json.lock")
    inodes = []
    for _ in range(2):
        with record_lock(get_platform_run_path(tmp_path)):
            # The lock file exists DURING the critical section, always.
            assert lock_path.is_file()
            inodes.append(os.stat(lock_path).st_ino)

    # The lock file SURVIVES release, and every acquisition locked the same inode.
    assert lock_path.is_file()
    assert len(set(inodes)) == 1

    # Two sequential locked updates merge instead of clobbering.
    update_platform_record(tmp_path, {"kind": "train", "id": "run-1", "url": "https://x/run-1"})
    update_platform_record(tmp_path, {"evaluations": {"rev": {"step": 5, "id": "ev-1", "url": "https://x/ev-1"}}})
    from prime_rl.monitors.prime import read_platform_record

    record = read_platform_record(tmp_path)
    assert record["kind"] == "train"
    assert record["id"] == "run-1"
    assert record["evaluations"]["rev"]["id"] == "ev-1"


def test_record_persistence_runs_off_the_event_loop(prime_init, tmp_path, monkeypatch):
    """The record merge (bounded flock retry included) must run in a worker
    thread via asyncio.to_thread, not on the event loop thread: contention
    for the full lock bound would otherwise freeze the loop."""
    import threading
    from pathlib import Path

    import prime_rl.monitors.prime as prime_module

    seen_threads = []
    real_merge = prime_module._merge_platform_record

    def spy(output_dir, merge):
        seen_threads.append(threading.get_ident())
        return real_merge(output_dir, merge)

    monkeypatch.setattr(prime_module, "_merge_platform_record", spy)

    config = _sft_config({"monitors": {"prime": {}}})
    asyncio.run(PrimeMonitor(config.monitors.prime).init(config=config, output_dir=tmp_path))

    assert seen_threads, "the record merge never ran"
    assert threading.get_ident() not in seen_threads, "record merge ran on the caller thread"

    # The async write sites (train run, evaluation) go through to_thread; only the sync
    # helpers stay callable directly.
    import re

    import prime_rl

    src = Path(prime_rl.__path__[0], "monitors", "prime.py").read_text()
    assert len(re.findall(r"asyncio\.to_thread\(\s*self\._record", src)) == 2

    from prime_rl.monitors.prime import read_platform_record

    record = read_platform_record(tmp_path)
    assert record["kind"] == "train"
    assert record["id"] == "run-123"


def test_platform_record_lock_acquisition_is_bounded(tmp_path, monkeypatch):
    """flock(LOCK_EX) must never block the event-loop thread indefinitely:
    while another process holds the lock, acquisition retries
    non-blockingly and raises TimeoutError past the bound instead of
    hanging."""
    import fcntl

    import prime_rl.utils.record as record_module
    from prime_rl.utils.pathing import get_platform_run_path

    monkeypatch.setattr(record_module, "RECORD_LOCK_TIMEOUT", 0.2)
    path = get_platform_run_path(tmp_path)
    lock_path = path.with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        with pytest.raises(TimeoutError, match="still held"):
            with record_module.record_lock(path):
                pass


def test_platform_record_persistence_is_best_effort(tmp_path, monkeypatch):
    """A storage failure on the OPTIONAL dashboard record (e.g. ENOLCK on a
    filesystem without flock support) must not abort training: the write is
    skipped with a warning, and no record is left behind."""
    import errno

    import prime_rl.monitors.prime as prime_module
    import prime_rl.utils.record as record_module

    def enolck(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(record_module.fcntl, "flock", enolck)
    # Must not raise: the platform record is best-effort.
    prime_module.update_platform_record(tmp_path, {"kind": "train", "id": "run-1", "url": "https://x/run-1"})
    assert prime_module.read_platform_record(tmp_path) is None
