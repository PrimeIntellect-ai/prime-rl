"""The prime monitor's run wiring: SFT registration, online evals sharing the trainer's
run, and the platform record."""

import asyncio
import errno
import fcntl
import os
from types import SimpleNamespace

import pytest

import prime_rl.monitors.prime as prime_module
import prime_rl.utils.record as record_module
from prime_rl.configs.monitors import PrimeMonitorConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.monitors.base import MonitorContext
from prime_rl.monitors.prime import PrimeMonitor, read_platform_record, update_platform_record
from prime_rl.utils.links import ATTEMPT_VAR
from prime_rl.utils.pathing import get_platform_run_path
from prime_rl.utils.process import RUN_FINISHER_VAR
from prime_rl.utils.record import record_lock


class FakeRun(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(id=kwargs.get("id", "run-123"), url="https://prime/training/run-123", attached="id" in kwargs)
        self.finished = False
        self.flushed = False
        self.kwargs = kwargs

    def _atexit_hook(self):
        pass

    def finish(self, *args, **kwargs):
        self.finished = True

    def flush(self, *args, **kwargs):
        self.flushed = True

    def log_metrics(self, *args, **kwargs):
        pass

    def log_episodes(self, *args, **kwargs):
        pass


@pytest.fixture
def prime_init(monkeypatch, tmp_path):
    """Replace the ``prime_runs`` SDK entry point with one that records its runs."""
    runs: list[FakeRun] = []

    def fake_init(**kwargs):
        runs.append(FakeRun(**kwargs))
        return runs[-1]

    monkeypatch.setattr(prime_module.pr, "init", fake_init)
    for var in ("RUN_ID", "PRIME_RUNS_MODE", RUN_FINISHER_VAR):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PRIME_API_KEY", "test-key")
    monkeypatch.setenv(ATTEMPT_VAR, str(tmp_path / "logs" / "attempt_1"))
    return runs


def _sft_config(overrides: dict | None = None) -> SFTConfig:
    config = {
        "model": {"name": "PrimeIntellect/Qwen3-0.6B"},
        "data": {"type": "fake", "batch_size": 8, "seq_len": 64},
        "max_steps": 10,
    }
    config.update(overrides or {})
    return SFTConfig.model_validate(config)


def _monitor(tmp_path, config, producer: str) -> PrimeMonitor:
    return PrimeMonitor(PrimeMonitorConfig(name="my-run"), MonitorContext(tmp_path, producer, config))


def test_sft_registers_dataset_training_fields(prime_init, tmp_path):
    config = _sft_config({"monitors": {"wandb": {"project": "sft-tests"}}})
    asyncio.run(_monitor(tmp_path, config, "trainer").init())

    (run,) = prime_init
    assert run.kwargs["kind"] == "train"
    assert run.kwargs["environments"] == []
    training = run.kwargs["training"]
    assert (training.max_steps, training.batch_size, training.seq_len) == (10, 8, 64)
    assert training.rollouts_per_example is None
    assert training.wandb_project == "sft-tests"
    assert read_platform_record(tmp_path)["id"] == "run-123"


def test_online_evals_share_the_trainers_run(prime_init, tmp_path, monkeypatch):
    """The online-eval process attaches to the run the trainer registered, and only it
    finishes the run: it drains its final evals after the trainer exits."""
    from prime_rl.entrypoints.sft import build_online_eval_config

    config = _sft_config({"eval": {"interval": 5, "source": [{"env": {"taskset": {"id": "harbor"}}}]}, "inference": {}})
    eval_config = build_online_eval_config(config)
    assert eval_config.monitors.prime == config.monitors.prime
    monkeypatch.setenv(RUN_FINISHER_VAR, "online-eval")

    trainer = _monitor(tmp_path, config, "trainer")
    asyncio.run(trainer.init())
    online_eval = _monitor(tmp_path, eval_config, "online-eval")
    asyncio.run(online_eval.init())
    assert online_eval.run.kwargs["id"] == trainer.run.id

    asyncio.run(trainer.finalize())
    assert trainer.run.flushed and not trainer.run.finished
    asyncio.run(online_eval.finalize())
    assert online_eval.run.finished


def test_online_eval_ignores_a_previous_attempts_run(prime_init, tmp_path, monkeypatch):
    update_platform_record(tmp_path, {"kind": "train", "id": "old", "attempt": "attempt_0"})
    with pytest.raises(RuntimeError, match="registered no Prime run"):
        prime_module.wait_for_train_run(tmp_path, timeout=0.0)


def test_platform_record_lock_keeps_a_stable_inode(tmp_path):
    """The record lock never unlinks its lock file: a waiting writer could hold the
    unlinked inode's lock while a later writer locks a fresh file."""
    lock_path = get_platform_run_path(tmp_path).with_suffix(".json.lock")
    inodes = []
    for _ in range(2):
        with record_lock(get_platform_run_path(tmp_path)):
            inodes.append(os.stat(lock_path).st_ino)
    assert lock_path.is_file() and len(set(inodes)) == 1

    update_platform_record(tmp_path, {"kind": "train", "id": "run-1"})
    update_platform_record(tmp_path, {"url": "https://x/run-1"})
    assert read_platform_record(tmp_path) == {"kind": "train", "id": "run-1", "url": "https://x/run-1"}


def test_platform_record_lock_acquisition_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(record_module, "RECORD_LOCK_TIMEOUT", 0.2)
    path = get_platform_run_path(tmp_path)
    lock_path = path.with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        with pytest.raises(TimeoutError, match="still held"):
            with record_lock(path):
                pass


def test_platform_record_persistence_is_best_effort(tmp_path, monkeypatch):
    """A storage failure on the record (e.g. ENOLCK) is skipped, never fatal."""

    def enolck(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(record_module.fcntl, "flock", enolck)
    update_platform_record(tmp_path, {"kind": "train", "id": "run-1"})
    assert read_platform_record(tmp_path) is None
