import asyncio
import errno
import uuid
from pathlib import Path
from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl import monitors
from prime_rl.monitors.file.traces.index import index_row, terminal_errors
from prime_rl.orchestrator.dispatcher import Dispatcher, DispatcherMetrics
from prime_rl.orchestrator.types import DispatchFailure, InflightEpisode, Policy


def test_terminal_errors_preserve_exact_trace_and_episode_joins():
    rec = {
        "id": "episode-1",
        "run": {"id": "run-1"},
        "group": {"id": "group-1"},
        "task": {"key": "episode-task"},
        "ok": False,
        "errors": [],
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
            "episode": "episode-1",
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
            "episode": "episode-1",
            "trace": "trace-2",
            "task": "trace-task-2",
            "stop_condition": "setup_error",
            "type": "SetupError",
            "message": "setup failed",
            "status_code": None,
            "runtime_id": "runtime-2",
            "error_count": 1,
        },
    ]
    assert index_row(1, rec, chunk=0, offset=0)["terminal_errors"] == terminal_errors(rec)

    episode_only = {
        **rec,
        "traces": [],
        "errors": [{"type": "FinalizeError", "message": "episode failed", "status_code": 500}],
    }
    assert terminal_errors(episode_only) == [
        {
            "run": "run-1",
            "group": "group-1",
            "episode": "episode-1",
            "trace": None,
            "task": "episode-task",
            "stop_condition": None,
            "type": "FinalizeError",
            "message": "episode failed",
            "status_code": 500,
            "runtime_id": None,
            "error_count": 1,
        }
    ]


def test_trace_failure_does_not_promote_episode_retry_history_to_terminal():
    rec = {
        "id": "episode-1",
        "run": {"id": "run-1"},
        "group": {"id": "group-1"},
        "task": {"key": "task-1"},
        "ok": False,
        # A previous attempt's episode error is retained on the final envelope.
        "errors": [{"type": "SetupError", "message": "recovered episode error"}],
        "traces": [
            {
                "id": "trace-1",
                "task": {"key": "task-1"},
                "agent": {"runtime": {"id": "runtime-1"}},
                "ok": False,
                "stop_condition": "agent_error",
                "errors": [{"type": "ProviderError", "message": "terminal trace error", "status_code": 503}],
            }
        ],
    }

    row = index_row(1, rec, chunk=0, offset=0)
    assert [(error["episode"], error["trace"], error["message"]) for error in row["terminal_errors"]] == [
        ("episode-1", "trace-1", "terminal trace error")
    ]
    assert row["num_errors"] == 2


def test_recovered_error_histories_are_not_terminal_failures():
    rec = {
        "id": "episode-1",
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
    # Keep the existing consumer contract while including trace-only retry history.
    assert row["num_errors"] == 3


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


class UpstreamError(Exception):
    status_code = 503


def test_escaped_request_failure_preserves_accounting_and_records_indexable_episode(monkeypatch):
    async def run():
        dispatcher = Dispatcher.__new__(Dispatcher)
        dispatcher.groups = {}
        dispatcher.metrics = DispatcherMetrics()
        dispatcher.out_q = asyncio.Queue()
        dispatcher.policy = Policy(version=7)
        dispatcher.progress = None
        dispatcher.run_id = "run-1"
        dispatcher.run_name = None
        dispatcher.train_envs = None
        dispatcher.retire = lambda meta: None
        dispatcher.release = lambda *, refund_admission: None

        group_id = uuid.UUID(int=1)
        task = SimpleNamespace(data=vf.WireTaskData(), key="task-1", hash="hash-1")
        meta = InflightEpisode(
            kind="eval",
            env_name="test-env",
            group_id=group_id,
            task=task,
            policy_version=5,
            step=7,
        )

        async def fail_request():
            raise UpstreamError("upstream unavailable")

        request = asyncio.create_task(fail_request())
        await asyncio.gather(request, return_exceptions=True)
        dispatcher.inflight = {request: meta}

        monitor_calls = []

        async def record_log(episodes, step, kind, subset):
            monitor_calls.append((episodes, step, kind, subset))

        logger = RecordingLogger()
        monkeypatch.setattr("prime_rl.orchestrator.dispatcher.monitors.log", record_log)
        monkeypatch.setattr("prime_rl.orchestrator.dispatcher.get_logger", lambda: logger)

        await dispatcher.handle_completed_request(request)

        # The training/eval contract is unchanged: exactly one DispatchFailure,
        # with no synthetic episode entering the result queue.
        result = dispatcher.out_q.get_nowait()
        assert isinstance(result, DispatchFailure)
        assert result.group_id == str(group_id)
        assert result.policy_version == 5
        assert result.error.type == "UpstreamError"
        assert result.error.message == "upstream unavailable"
        assert result.error.status_code == 503
        assert result.error.traceback
        assert dispatcher.out_q.empty()

        ((episodes, step, kind, subset),) = monitor_calls
        assert (step, kind, subset) == (7, "eval", "all")
        (episode,) = episodes
        assert not episode.ok
        assert episode.group.id == str(group_id)
        assert episode.run.id == "run-1"
        assert episode.run.work.step == 7
        assert episode.run.work.policy.start == 5
        assert episode.run.work.policy.end == 7
        assert [(error.type, error.message, error.status_code, error.traceback) for error in episode.errors] == [
            ("UpstreamError", "upstream unavailable", 503, None)
        ]

        row = index_row(1, episode.to_record(), chunk=0, offset=0)
        assert row["num_errors"] == 1
        assert row["terminal_errors"] == [
            {
                "run": "run-1",
                "group": str(group_id),
                "episode": episode.id,
                "trace": None,
                "task": "task-1",
                "stop_condition": None,
                "type": "UpstreamError",
                "message": "upstream unavailable",
                "status_code": 503,
                "runtime_id": None,
                "error_count": 1,
            }
        ]
        assert any(episode.id in message for message in logger.messages)

    asyncio.run(run())
