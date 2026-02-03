"""Tests for EnvWorker auto-restart functionality."""

import asyncio
import os
import signal
import time
from multiprocessing import Process, Queue
from unittest.mock import MagicMock, patch

import pytest

from prime_rl.orchestrator.env_worker import EnvWorker, WorkerDiedError


def dummy_worker_main(request_queue: Queue, response_queue: Queue, **kwargs):
    """A minimal worker that just sleeps until killed."""
    while True:
        try:
            request = request_queue.get(timeout=0.1)
            if request is None:
                break
        except Exception:
            pass
        time.sleep(0.1)


@pytest.fixture
def mock_client_config():
    """Create a mock client config."""
    config = MagicMock()
    config.model_dump.return_value = {}
    return config


@pytest.fixture
def env_worker(mock_client_config):
    """Create an EnvWorker with mocked dependencies."""
    worker = EnvWorker(
        env_id="test_env",
        env_args={},
        client_config=mock_client_config,
        model_name="test-model",
        seq_len=1024,
        interleaved_rollouts=False,
        max_concurrent_groups=1,
        tasks_per_minute=60,
        example_lookup={},
        worker_name="test_worker",
    )
    return worker


def test_restart_drains_queues(env_worker):
    """Test that restart() drains any stale data from queues."""
    # Put some data in the queues
    env_worker.request_queue.put("stale_request")
    env_worker.response_queue.put("stale_response")

    with patch.object(env_worker, "start"):
        env_worker._restart()

    # Queues should be empty after restart
    assert env_worker.request_queue.empty()
    assert env_worker.response_queue.empty()


def test_pending_count_returns_large_when_dead(env_worker):
    """Test that pending_count returns large number when worker is dead."""
    env_worker._dead = False
    env_worker.pending_futures = {"a": None, "b": None}
    assert env_worker.pending_count == 2

    env_worker._dead = True
    assert env_worker.pending_count == 999999


def test_collect_responses_restarts_on_worker_death(env_worker):
    """Test that collect_responses auto-restarts worker when it dies."""

    async def run_test():
        # Create a real process that we can kill
        env_worker.process = Process(
            target=dummy_worker_main,
            args=(env_worker.request_queue, env_worker.response_queue),
            daemon=True,
        )
        env_worker.process.start()
        env_worker._stopping = False
        env_worker._dead = False

        # Add a pending future that should be failed
        loop = asyncio.get_event_loop()
        pending_future = loop.create_future()
        env_worker.pending_futures["test_request"] = pending_future

        # Start collect_responses in background
        # Use side_effect to actually restart the process so the loop doesn't spin
        def real_restart():
            env_worker.process = Process(
                target=dummy_worker_main,
                args=(env_worker.request_queue, env_worker.response_queue),
                daemon=True,
            )
            env_worker.process.start()
            env_worker._dead = False

        with patch.object(env_worker, "_restart", side_effect=real_restart) as mock_restart:
            collect_task = asyncio.create_task(env_worker.collect_responses())

            # Give it a moment to start
            await asyncio.sleep(0.1)

            # Kill the worker process
            os.kill(env_worker.process.pid, signal.SIGKILL)
            env_worker.process.join(timeout=1)

            # Wait for collect_responses to detect death and restart
            await asyncio.sleep(0.2)

            # Cancel the collect task
            collect_task.cancel()
            try:
                await collect_task
            except asyncio.CancelledError:
                pass

            # Cleanup the restarted process
            if env_worker.process and env_worker.process.is_alive():
                env_worker.process.terminate()
                env_worker.process.join(timeout=1)

            # Verify restart was called exactly once
            mock_restart.assert_called_once()

            # Verify pending future was failed with WorkerDiedError
            assert pending_future.done()
            with pytest.raises(WorkerDiedError):
                pending_future.result()

    asyncio.run(run_test())


def test_update_model_name(env_worker):
    """Test that update_model_name changes the model used for future requests."""
    assert env_worker.model_name == "test-model"

    env_worker.update_model_name("new-lora-model")

    assert env_worker.model_name == "new-lora-model"


def test_submit_request_uses_current_model_name(env_worker):
    """Test that submit_request uses the current model_name."""
    import asyncio

    captured_requests = []

    def capture_put(request):
        captured_requests.append(request)

    async def run_test():
        # Mock the queue put to capture requests
        env_worker.request_queue.put = capture_put

        # Initial model name
        future, request_id = await env_worker.submit_request(
            example_id=1, rollouts_per_example=1, sampling_args={"temperature": 1.0}
        )
        assert captured_requests[-1].model_name == "test-model"

        # Update model name
        env_worker.update_model_name("lora-model")

        # New request should use new model name
        future2, request_id2 = await env_worker.submit_request(
            example_id=1, rollouts_per_example=1, sampling_args={"temperature": 1.0}
        )
        assert captured_requests[-1].model_name == "lora-model"

    asyncio.run(run_test())


def test_full_restart_cycle(mock_client_config):
    """Test a full restart cycle with actual process management."""
    worker = EnvWorker(
        env_id="test_env",
        env_args={},
        client_config=mock_client_config,
        model_name="test-model",
        seq_len=1024,
        interleaved_rollouts=False,
        max_concurrent_groups=1,
        tasks_per_minute=60,
        example_lookup={},
        worker_name="test_worker",
    )

    # Patch worker_main to use our dummy
    with patch("prime_rl.orchestrator.env_worker.worker_main", dummy_worker_main):
        # Start worker
        worker.start()
        assert worker.process is not None
        assert worker.process.is_alive()
        original_pid = worker.process.pid

        # Kill it
        os.kill(worker.process.pid, signal.SIGKILL)
        worker.process.join(timeout=1)
        assert not worker.process.is_alive()

        # Restart
        worker._restart()
        assert worker.process is not None
        assert worker.process.is_alive()
        assert worker.process.pid != original_pid  # New process
        assert not worker._dead

        # Cleanup
        worker.stop()
