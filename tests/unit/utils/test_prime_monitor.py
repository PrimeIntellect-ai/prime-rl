import asyncio
import io
import json
from collections import deque
from unittest.mock import Mock

import httpx
import pyarrow.parquet as pq

from prime_rl.utils.monitor.prime import (
    _MAX_PENDING_SAMPLE_UPLOADS,
    PrimeMonitor,
)


def _new_monitor() -> PrimeMonitor:
    monitor = PrimeMonitor.__new__(PrimeMonitor)
    monitor._closed = True
    return monitor


def _new_drain_monitor() -> PrimeMonitor:
    """Monitor with the minimum state needed for sample-upload drain tests."""
    monitor = _new_monitor()
    monitor.logger = Mock()
    monitor.run_id = "run-test"
    monitor._pending_sample_steps = set()
    monitor._sample_upload_queue = deque()
    monitor._sample_upload_lock = None
    monitor.last_log_samples_step = -1
    return monitor


def _build_rollout(*, example_id: int, reward: float, task: str) -> dict:
    return {
        "example_id": example_id,
        "prompt": [{"role": "user", "content": f"prompt-{example_id}"}],
        "completion": [{"role": "assistant", "content": f"completion-{example_id}"}],
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": f"prompt-{example_id}"}],
                "completion": [{"role": "assistant", "content": f"completion-{example_id}"}],
                "reward": reward,
                "advantage": reward / 2,
                "extras": {"source": "test"},
                "tokens": {
                    "prompt_ids": [1, 2, 3],
                    "completion_ids": [4, 5],
                },
            }
        ],
        "answer": f"answer-{example_id}",
        "task": task,
        "info": {"difficulty": "easy"},
        "reward": reward,
        "advantage": reward / 2,
        "metrics": {"accuracy": reward},
        "timing": {"generation": {"start": 0.0, "end": 12.5, "duration": 12.5}},
    }


def test_rollouts_to_parquet_bytes_preserves_all_rollouts_and_ids():
    monitor = _new_monitor()
    monitor.run_id = "run-123"

    parquet_bytes = monitor._rollouts_to_parquet_bytes(
        [
            _build_rollout(example_id=101, reward=1.0, task="task-a"),
            _build_rollout(example_id=202, reward=0.0, task="task-b"),
        ],
        step=7,
    )

    assert parquet_bytes is not None

    table = pq.read_table(io.BytesIO(parquet_bytes))
    rows = table.to_pylist()

    assert len(rows) == 2
    assert [row["problem_id"] for row in rows] == [101, 202]
    assert [row["sample_id"] for row in rows] == [0, 1]
    assert all(row["run_id"] == "run-123" for row in rows)
    assert all(row["step"] == 7 for row in rows)
    assert json.loads(rows[0]["prompt"])[0]["content"] == "prompt-101"
    assert json.loads(rows[1]["completion"])[0]["content"] == "completion-202"


def test_rollouts_to_parquet_bytes_skips_rollouts_without_trajectory():
    monitor = _new_monitor()
    monitor.run_id = "run-456"

    parquet_bytes = monitor._rollouts_to_parquet_bytes(
        [
            _build_rollout(example_id=1, reward=1.0, task="task-a"),
            {
                "example_id": 2,
                "prompt": [{"role": "user", "content": "missing-trajectory"}],
                "completion": [{"role": "assistant", "content": "ignored"}],
                "trajectory": [],
            },
        ],
        step=3,
    )

    assert parquet_bytes is not None

    table = pq.read_table(io.BytesIO(parquet_bytes))
    rows = table.to_pylist()

    assert len(rows) == 1
    assert rows[0]["problem_id"] == 1
    assert rows[0]["sample_id"] == 0


def test_sample_upload_failure_keeps_step_in_backlog_and_retries_next_tick():
    monitor = _new_drain_monitor()
    attempts: list[int] = []
    fail_steps_once = {0}

    async def upload_one(step: int, _bytes: bytes) -> None:
        attempts.append(step)
        if step in fail_steps_once:
            fail_steps_once.discard(step)
            raise httpx.ConnectError("transient")

    monitor._upload_one_sample_step = upload_one  # type: ignore[method-assign]

    async def scenario() -> None:
        monitor._pending_sample_steps.add(0)
        await monitor._enqueue_and_drain_samples_async(0, b"p0")
        # First tick fails: step 0 stays at the head of the backlog.
        assert list(monitor._sample_upload_queue) == [(0, b"p0")]
        assert 0 in monitor._pending_sample_steps

        # Second tick adds step 10 and drains both, oldest-first.
        monitor._pending_sample_steps.add(10)
        await monitor._enqueue_and_drain_samples_async(10, b"p10")
        assert list(monitor._sample_upload_queue) == []
        assert monitor._pending_sample_steps == set()
        assert monitor.last_log_samples_step == 10

    asyncio.run(scenario())
    # Step 0: failed, retried, succeeded; then step 10 succeeded.
    assert attempts == [0, 0, 10]


def test_in_flight_sample_upload_survives_concurrent_eviction():
    """Regression test: backlog eviction runs without the upload lock, so a
    concurrent log_samples() call could previously popleft the head that the
    drain coroutine was peeking at and currently uploading. The fix pops the
    in-flight item out of the queue into a local variable before awaiting,
    making it invisible to eviction.
    """
    monitor = _new_drain_monitor()
    completed: list[int] = []

    async def scenario() -> None:
        started = asyncio.Event()
        finish = asyncio.Event()

        async def slow_upload(step: int, _bytes: bytes) -> None:
            if not started.is_set():
                started.set()
            await finish.wait()
            completed.append(step)

        monitor._upload_one_sample_step = slow_upload  # type: ignore[method-assign]

        # Pre-load step 10; it will become the in-flight item.
        monitor._pending_sample_steps.add(10)
        monitor._sample_upload_queue.append((10, b"p10"))

        # Drain coroutine A: enqueues step 20, acquires lock, popleft's step 10
        # into a local variable, awaits.
        monitor._pending_sample_steps.add(20)
        a = asyncio.create_task(monitor._enqueue_and_drain_samples_async(20, b"p20"))
        await started.wait()
        assert list(monitor._sample_upload_queue) == [(20, b"p20")]

        # Five concurrent log_samples calls. Each appends + evicts outside the lock.
        # Queue grows to size 6 on the last call; eviction popleft's step 20.
        # Step 10 is in A's local var, so eviction can never see it.
        extra_steps = [30, 40, 50, 60, 70]
        for s in extra_steps:
            monitor._pending_sample_steps.add(s)
        bs = [asyncio.create_task(monitor._enqueue_and_drain_samples_async(s, f"p{s}".encode())) for s in extra_steps]
        await asyncio.sleep(0.05)
        queue_before_release = [s for s, _ in monitor._sample_upload_queue]
        assert queue_before_release == [30, 40, 50, 60, 70]

        finish.set()
        await asyncio.gather(a, *bs)

    asyncio.run(scenario())
    # The in-flight step survived, the correctly-oldest queued step was evicted.
    assert 10 in completed
    assert 20 not in completed
    for s in [30, 40, 50, 60, 70]:
        assert s in completed
    assert _MAX_PENDING_SAMPLE_UPLOADS == 5


def test_sanitize_json_payload_drops_non_finite_values_and_logs_paths():
    monitor = _new_monitor()
    monitor.logger = Mock()

    payload = {
        "metrics": {"finite": 1.0, "nan": float("nan")},
        "distributions": [0.5, float("inf")],
    }

    sanitized = monitor._sanitize_json_payload("metrics", payload)

    assert sanitized == {"metrics": {"finite": 1.0}, "distributions": [0.5]}
    monitor.logger.warning.assert_called_once_with(
        "Dropping 2 non-finite value(s) from Prime monitor metrics payload: metrics.nan, distributions[1]"
    )
