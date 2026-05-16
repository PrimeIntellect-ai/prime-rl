import json
from pathlib import Path
from typing import Any

import pytest

from prime_rl.utils.monitor.file import FileMonitor
from prime_rl.utils.monitor.multi import MultiMonitor


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class _MemoryMonitor:
    required = False

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self.history.append(metrics)

    def log_samples(self, rollouts: list[Any], step: int) -> None:
        return

    def log_eval_samples(self, rollouts: list[Any], env_name: str, step: int) -> None:
        return

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        return

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        return

    def close(self) -> None:
        return


class _FailingOptionalMonitor(_MemoryMonitor):
    def log(self, metrics: dict[str, Any], step: int) -> None:
        raise RuntimeError("optional monitor failure")


def test_file_monitor_writes_one_line_per_log_call(tmp_path: Path) -> None:
    monitor = FileMonitor(tmp_path / "metrics.jsonl")
    try:
        monitor.log({"reward": 0.1, "loss": 1.5}, step=1)
        monitor.log({"reward": 0.4, "loss": 1.2}, step=2)
    finally:
        monitor.close()

    rows = _read_lines(tmp_path / "metrics.jsonl")
    assert rows == [
        {"step": 1, "reward": 0.1, "loss": 1.5},
        {"step": 2, "reward": 0.4, "loss": 1.2},
    ]


def test_file_monitor_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "metrics.jsonl"
    monitor = FileMonitor(target)
    monitor.close()
    assert target.exists()


def test_file_monitor_flushes_after_each_write(tmp_path: Path) -> None:
    """Polling readers see partial files; writes must hit disk between log calls."""
    path = tmp_path / "metrics.jsonl"
    monitor = FileMonitor(path)
    try:
        monitor.log({"reward": 0.5}, step=10)
        # File must already contain the line on disk; we did not close yet.
        rows = _read_lines(path)
        assert rows == [{"step": 10, "reward": 0.5}]
    finally:
        monitor.close()


def test_file_monitor_step_argument_is_canonical(tmp_path: Path) -> None:
    monitor = FileMonitor(tmp_path / "metrics.jsonl")
    try:
        monitor.log({"step": "not-the-sweep-step", "reward": 0.5}, step=10)
    finally:
        monitor.close()

    rows = _read_lines(tmp_path / "metrics.jsonl")
    assert rows == [{"step": 10, "reward": 0.5}]


@pytest.mark.parametrize("step", [True, 1.5, "1"])
def test_file_monitor_rejects_non_integer_step(tmp_path: Path, step: object) -> None:
    path = tmp_path / "metrics.jsonl"
    monitor = FileMonitor(path)
    try:
        with pytest.raises(TypeError, match="step must be an integer"):
            monitor.log({"reward": 0.5}, step=step)
    finally:
        monitor.close()

    assert path.read_text() == ""
    assert monitor.history == []


def test_file_monitor_rejects_negative_step(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    monitor = FileMonitor(path)
    try:
        with pytest.raises(ValueError, match="step must be non-negative"):
            monitor.log({"reward": 0.5}, step=-1)
    finally:
        monitor.close()

    assert path.read_text() == ""
    assert monitor.history == []


def test_multi_monitor_propagates_file_monitor_errors(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    file_monitor = FileMonitor(path)
    memory_monitor = _MemoryMonitor()
    monitor = MultiMonitor([memory_monitor, file_monitor])
    try:
        with pytest.raises(TypeError, match="step must be an integer"):
            monitor.log({"reward": 0.5}, step=True)
    finally:
        monitor.close()

    assert path.read_text() == ""


def test_multi_monitor_keeps_optional_monitor_errors_non_fatal(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    file_monitor = FileMonitor(path)
    memory_monitor = _MemoryMonitor()
    monitor = MultiMonitor([_FailingOptionalMonitor(), memory_monitor, file_monitor])
    try:
        monitor.log({"reward": 0.5}, step=1)
    finally:
        monitor.close()

    assert memory_monitor.history == [{"reward": 0.5}]
    assert _read_lines(path) == [{"step": 1, "reward": 0.5}]


def test_file_monitor_replaces_non_finite_values_with_null(tmp_path: Path) -> None:
    monitor = FileMonitor(tmp_path / "metrics.jsonl")
    try:
        monitor.log(
            {
                "reward": float("nan"),
                "loss": float("inf"),
                "nested": (1.0, float("-inf")),
                "ok": 0.5,
            },
            step=1,
        )
    finally:
        monitor.close()
    rows = _read_lines(tmp_path / "metrics.jsonl")
    assert rows == [{"step": 1, "reward": None, "loss": None, "nested": [1.0, None], "ok": 0.5}]


def test_file_monitor_no_op_on_non_master_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    target = tmp_path / "metrics.jsonl"
    monitor = FileMonitor(target)
    try:
        monitor.log({"reward": 1.0}, step=1)
    finally:
        monitor.close()
    assert not target.exists()


def test_file_monitor_appends_to_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"step": 0, "reward": 0.0}) + "\n")

    monitor = FileMonitor(path)
    try:
        monitor.log({"reward": 0.5}, step=1)
    finally:
        monitor.close()

    rows = _read_lines(path)
    assert rows == [
        {"step": 0, "reward": 0.0},
        {"step": 1, "reward": 0.5},
    ]


def test_file_monitor_history_buffer(tmp_path: Path) -> None:
    monitor = FileMonitor(tmp_path / "metrics.jsonl", keep_full_history=True)
    try:
        monitor.log({"reward": 0.1}, step=1)
        monitor.log({"reward": 0.2}, step=2)
    finally:
        monitor.close()
    assert monitor.history == [{"reward": 0.1}, {"reward": 0.2}]


def test_setup_monitor_adds_file_monitor_when_env_var_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sweep launcher injection: env var present => FileMonitor in the stack."""
    target = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("PRIME_RL_SWEEP_METRICS_JSONL", str(target))
    monkeypatch.setattr("prime_rl.utils.monitor._MONITOR", None)

    from prime_rl.utils.monitor import FileMonitor as FileMonitorClass
    from prime_rl.utils.monitor import setup_monitor

    monitor = setup_monitor()
    try:
        monitor.log({"reward": 0.7}, step=3)
    finally:
        if hasattr(monitor, "close"):
            monitor.close()

    # NoOpMonitor by itself if no other monitors. Single monitor path returns
    # the FileMonitor directly.
    assert isinstance(monitor, FileMonitorClass)
    assert target.exists()
