import asyncio
import errno
from pathlib import Path

from prime_rl import monitors
from prime_rl.monitors.file.traces.index import index_row, terminal_errors


def test_terminal_errors_preserve_exact_trace_and_episode_joins():
    rec = {
        "id": "episode-1",
        "run": {"id": "run-1"},
        "group": {"id": "group-1"},
        "task": {"key": "episode-task"},
        "ok": False,
        "errors": [{"type": "FinalizeError", "message": "episode failed", "status_code": 500}],
        "traces": [
            {
                "id": "trace-1",
                "task": {"key": "trace-task-1"},
                "agent": {"runtime": {"id": "runtime-1"}},
                "ok": False,
                "stop_condition": "agent_error",
                "errors": [
                    {"type": "ProviderError", "message": "recovered", "status_code": 429},
                    {"type": "ProviderError", "message": "terminal", "status_code": 503},
                ],
            },
            {
                "id": "trace-2",
                "task": {"key": "trace-task-2"},
                "agent": {"runtime": {"id": "runtime-2"}},
                "ok": False,
                "stop_condition": "setup_error",
                "errors": [{"type": "SetupError", "message": "setup failed"}],
            },
        ],
    }

    assert terminal_errors(rec) == [
        {
            "run": "run-1",
            "group": "group-1",
            "trace": "trace-1",
            "task": "trace-task-1",
            "stop_condition": "agent_error",
            "type": "ProviderError",
            "message": "terminal",
            "status_code": 503,
            "runtime_id": "runtime-1",
            "error_count": 2,
        },
        {
            "run": "run-1",
            "group": "group-1",
            "trace": "trace-2",
            "task": "trace-task-2",
            "stop_condition": "setup_error",
            "type": "SetupError",
            "message": "setup failed",
            "status_code": None,
            "runtime_id": "runtime-2",
            "error_count": 1,
        },
        {
            "run": "run-1",
            "group": "group-1",
            "trace": None,
            "task": "episode-task",
            "stop_condition": None,
            "type": "FinalizeError",
            "message": "episode failed",
            "status_code": 500,
            "runtime_id": None,
            "error_count": 1,
        },
    ]
    assert index_row(1, rec, chunk=0, offset=0)["terminal_errors"] == terminal_errors(rec)


def test_recovered_error_histories_are_not_terminal_failures():
    rec = {
        "run": {"id": "run-1"},
        "group": {"id": "group-1"},
        "task": {"key": "task-1"},
        "ok": True,
        "errors": [{"type": "ProviderError", "message": "recovered episode error"}],
        "traces": [
            {
                "id": "trace-1",
                "task": {"key": "task-1"},
                "agent": {"runtime": {"id": "runtime-1"}},
                "ok": True,
                "stop_condition": "completed",
                "errors": [
                    {"type": "ProviderError", "message": "first recovered error"},
                    {"type": "ProviderError", "message": "second recovered error"},
                ],
            }
        ],
    }

    row = index_row(1, rec, chunk=0, offset=0)
    assert row["terminal_errors"] == []
    assert row["num_errors"] == 1


class RecordingLogger:
    def __init__(self):
        self.fields = None
        self.exception = None
        self.messages = []

    def bind(self, **fields):
        self.fields = fields
        return self

    def opt(self, *, exception):
        self.exception = exception
        return self

    def warning(self, message):
        self.messages.append(message)


class FailingMonitor:
    producer = "orch"
    path = Path("/fallback/metrics.jsonl")

    def __init__(self):
        self.error = OSError(errno.ENOSPC, "No space left on device", "/actual/traces.jsonl")

    async def log(self, data, *, step, kind, subset):
        raise self.error


class HealthyMonitor:
    def __init__(self):
        self.calls = []

    async def log(self, data, *, step, kind, subset):
        self.calls.append((data, step, kind, subset))


def test_monitor_write_failure_is_nonfatal_and_reports_context(monkeypatch):
    failing = FailingMonitor()
    healthy = HealthyMonitor()
    logger = RecordingLogger()
    monkeypatch.setattr(monitors, "MONITORS", [failing, healthy])
    monkeypatch.setattr(monitors, "get_logger", lambda: logger)

    asyncio.run(monitors.log({"train/loss": 1.0}, step=3))

    assert healthy.calls == [({"train/loss": 1.0}, 3, "train", "effective")]
    assert logger.fields == {
        "metric": "monitor/errors",
        "value": 1,
        "producer": "orch",
        "operation": "log",
        "path": "/actual/traces.jsonl",
        "errno": errno.ENOSPC,
    }
    assert logger.exception is failing.error
    assert len(logger.messages) == 1
