"""Tests for EnvWorker subprocess management."""

from unittest.mock import MagicMock

import pytest

from prime_rl.orchestrator.env_worker import EnvWorker
from prime_rl.utils.config import ClientConfig


@pytest.fixture
def mock_client_config():
    """Return a minimal ClientConfig for testing."""
    return ClientConfig(base_url=["http://localhost:8000"])


@pytest.fixture
def env_worker(mock_client_config):
    """Return an EnvWorker instance with mocked dependencies."""
    return EnvWorker(
        env_id="test-env",
        env_args={},
        client_config=mock_client_config,
        model_name="test-model",
        seq_len=1024,
        interleaved_rollouts=False,
        max_concurrent=10,
        example_lookup={},
        sampling_args={},
        worker_name="test-worker",
    )


@pytest.mark.asyncio
async def test_collect_responses_detects_dead_worker(env_worker):
    """Test that collect_responses raises RuntimeError when worker process dies."""
    # Mock the process to simulate a dead worker
    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = 1
    env_worker.process = mock_process

    with pytest.raises(RuntimeError, match="Worker 'test-worker' died unexpectedly"):
        await env_worker.collect_responses()


@pytest.mark.asyncio
async def test_collect_responses_includes_exit_code(env_worker):
    """Test that the error message includes the exit code."""
    mock_process = MagicMock()
    mock_process.is_alive.return_value = False
    mock_process.exitcode = -9  # SIGKILL
    env_worker.process = mock_process

    with pytest.raises(RuntimeError, match="exit code: -9"):
        await env_worker.collect_responses()


@pytest.mark.asyncio
async def test_collect_responses_no_error_when_process_alive(env_worker):
    """Test that collect_responses continues normally when process is alive."""
    mock_process = MagicMock()
    mock_process.is_alive.return_value = True
    env_worker.process = mock_process

    # Run one iteration then cancel to avoid infinite loop
    import asyncio

    async def run_with_timeout():
        task = asyncio.create_task(env_worker.collect_responses())
        await asyncio.sleep(0.02)  # Let it run one iteration
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Should complete without raising RuntimeError
    await run_with_timeout()


@pytest.mark.asyncio
async def test_collect_responses_no_error_when_no_process(env_worker):
    """Test that collect_responses handles None process gracefully."""
    env_worker.process = None

    import asyncio

    async def run_with_timeout():
        task = asyncio.create_task(env_worker.collect_responses())
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Should not raise when process is None (not started yet)
    await run_with_timeout()
