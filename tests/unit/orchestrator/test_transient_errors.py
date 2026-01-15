"""
Tests for transient error handling and backoff in the scheduler.

This file demonstrates multiple approaches to test transient error handling:
1. Unit test with mocked client raising APIConnectionError
2. Integration test with a mock HTTP server that returns errors
3. Testing the error classification logic

To run these tests:
    pytest tests/unit/orchestrator/test_transient_errors.py -v
"""

import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ConnectError
from openai import APIConnectionError, AsyncOpenAI


# =============================================================================
# Approach 1: Unit Test - Error Classification
# =============================================================================


def is_transient_error(error: Exception) -> bool:
    """
    Check if an error is transient (network/infrastructure related).

    This function should be added to scheduler.py for the backoff implementation.
    """
    # Check error chain for transient causes
    current = error
    while current is not None:
        # Connection errors (DNS failure, connection refused, etc.)
        if isinstance(current, (ConnectionError, ConnectError)):
            return True
        # OpenAI connection errors
        if isinstance(current, APIConnectionError):
            return True
        # Check the string representation for common transient patterns
        error_str = str(current).lower()
        if any(
            pattern in error_str
            for pattern in [
                "connection",
                "timeout",
                "name or service not known",
                "dns",
                "temporary failure",
                "network",
            ]
        ):
            return True
        current = current.__cause__
    return False


class TestErrorClassification:
    """Test the transient error classification logic."""

    def test_connection_error_is_transient(self):
        """ConnectionError should be classified as transient."""
        error = ConnectionError("Connection refused")
        assert is_transient_error(error) is True

    def test_api_connection_error_is_transient(self):
        """APIConnectionError should be classified as transient."""
        error = APIConnectionError(message="Connection error", request=MagicMock())
        assert is_transient_error(error) is True

    def test_dns_error_is_transient(self):
        """DNS resolution errors should be classified as transient."""
        inner = OSError("[Errno -2] Name or service not known")
        error = APIConnectionError(message="Connection error", request=MagicMock())
        error.__cause__ = inner
        assert is_transient_error(error) is True

    def test_value_error_is_not_transient(self):
        """ValueError should NOT be classified as transient."""
        error = ValueError("Invalid argument")
        assert is_transient_error(error) is False

    def test_nested_connection_error_is_transient(self):
        """Nested connection errors should still be classified as transient."""
        inner = ConnectError("Connection refused")
        middle = APIConnectionError(message="Connection error", request=MagicMock())
        middle.__cause__ = inner
        outer = Exception("Rollout failed")
        outer.__cause__ = middle
        assert is_transient_error(outer) is True


# =============================================================================
# Approach 2: Unit Test - Mock OpenAI Client
# =============================================================================


class TestMockedClientErrors:
    """Test error handling with a mocked OpenAI client."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock AsyncOpenAI client."""
        client = AsyncMock(spec=AsyncOpenAI)
        client.chat = AsyncMock()
        client.chat.completions = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_client_raises_connection_error(self, mock_client):
        """Test that connection errors are properly raised."""
        # Configure mock to raise APIConnectionError
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(message="Connection error", request=MagicMock())
        )

        # Attempt to call the client
        with pytest.raises(APIConnectionError):
            await mock_client.chat.completions.create(
                model="test-model", messages=[{"role": "user", "content": "test"}]
            )

    @pytest.mark.asyncio
    async def test_intermittent_failures(self, mock_client):
        """Test handling of intermittent failures that recover."""
        call_count = 0

        async def intermittent_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIConnectionError(message="Connection error", request=MagicMock())
            return MagicMock()  # Success on 3rd call

        mock_client.chat.completions.create = intermittent_create

        # First two calls should fail
        with pytest.raises(APIConnectionError):
            await mock_client.chat.completions.create(model="test", messages=[])
        with pytest.raises(APIConnectionError):
            await mock_client.chat.completions.create(model="test", messages=[])

        # Third call should succeed
        result = await mock_client.chat.completions.create(model="test", messages=[])
        assert result is not None


# =============================================================================
# Approach 3: Integration Test - Mock HTTP Server
# =============================================================================


class FailingHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler that can be configured to fail."""

    # Class-level configuration
    fail_count = 0
    max_failures = 3
    failure_mode = "connection_reset"  # or "500", "timeout"

    def do_POST(self):
        if FailingHTTPHandler.fail_count < FailingHTTPHandler.max_failures:
            FailingHTTPHandler.fail_count += 1

            if FailingHTTPHandler.failure_mode == "500":
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Internal server error"}')
            elif FailingHTTPHandler.failure_mode == "timeout":
                time.sleep(10)  # Simulate timeout
            else:  # connection_reset
                self.connection.close()
            return

        # Success response
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        import json

        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass  # Suppress logging


class TestMockHTTPServer:
    """Integration tests using a mock HTTP server."""

    @pytest.fixture
    def mock_server(self):
        """Start a mock HTTP server for testing."""
        # Reset failure counter
        FailingHTTPHandler.fail_count = 0
        FailingHTTPHandler.max_failures = 3
        FailingHTTPHandler.failure_mode = "500"

        server = HTTPServer(("localhost", 0), FailingHTTPHandler)
        port = server.server_address[1]

        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()

        yield f"http://localhost:{port}/v1"

        server.shutdown()

    @pytest.mark.asyncio
    async def test_server_returns_500_then_succeeds(self, mock_server):
        """Test that the server returns 500 errors then succeeds."""
        client = AsyncOpenAI(
            api_key="test-key",
            base_url=mock_server,
            max_retries=0,  # Disable built-in retries for testing
        )

        # First 3 calls should fail with 500
        for i in range(3):
            try:
                await client.chat.completions.create(model="test-model", messages=[{"role": "user", "content": "test"}])
                pytest.fail("Expected error")
            except Exception:
                pass  # Expected

        # 4th call should succeed
        response = await client.chat.completions.create(
            model="test-model", messages=[{"role": "user", "content": "test"}]
        )
        assert response is not None


# =============================================================================
# Approach 4: Test Backoff Logic
# =============================================================================


class TestBackoffLogic:
    """Test the exponential backoff calculation."""

    def test_backoff_calculation(self):
        """Test exponential backoff values."""
        base = 2.0
        max_backoff = 60.0
        threshold = 10

        # Before threshold, no backoff
        for failures in range(threshold):
            backoff = 0 if failures < threshold else min(base ** (failures - threshold), max_backoff)
            assert backoff == 0

        # After threshold, exponential backoff
        expected_backoffs = [
            (10, 1.0),  # 2^0 = 1
            (11, 2.0),  # 2^1 = 2
            (12, 4.0),  # 2^2 = 4
            (13, 8.0),  # 2^3 = 8
            (14, 16.0),  # 2^4 = 16
            (15, 32.0),  # 2^5 = 32
            (16, 60.0),  # 2^6 = 64 -> capped at 60
            (20, 60.0),  # Still capped
        ]

        for failures, expected in expected_backoffs:
            backoff = min(base ** (failures - threshold), max_backoff)
            assert backoff == expected, f"Expected {expected} for {failures} failures, got {backoff}"

    @pytest.mark.asyncio
    async def test_backoff_timing(self):
        """Test that backoff actually delays execution."""

        async def simulate_backoff(consecutive_failures: int) -> float:
            """Simulate the backoff behavior and return actual delay."""
            threshold = 3
            base = 2.0
            max_backoff = 1.0  # Use small value for testing

            if consecutive_failures >= threshold:
                backoff = min(base ** (consecutive_failures - threshold), max_backoff)
                start = time.monotonic()
                await asyncio.sleep(backoff)
                return time.monotonic() - start
            return 0.0

        # No delay before threshold
        delay = await simulate_backoff(0)
        assert delay < 0.1

        # Delay after threshold
        delay = await simulate_backoff(3)  # 2^0 = 1s
        assert 0.9 < delay < 1.2  # Allow some tolerance


# =============================================================================
# Approach 5: Simulate Full Scheduler Behavior (Pseudo-Integration)
# =============================================================================


class MockSchedulerContext:
    """Simulates scheduler context for testing transient error handling."""

    def __init__(self):
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.backoff_base = 2.0
        self.max_backoff = 10.0
        self.total_backoff_time = 0.0

    async def handle_rollout_result(self, success: bool, error: Exception | None = None):
        """Simulate handling a rollout result with backoff logic."""
        if not success and error is not None:
            if is_transient_error(error):
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    backoff = min(
                        self.backoff_base ** (self.consecutive_failures - self.max_consecutive_failures),
                        self.max_backoff,
                    )
                    self.total_backoff_time += backoff
                    await asyncio.sleep(backoff)
            else:
                # Non-transient error - reset counter
                self.consecutive_failures = 0
        else:
            # Success - reset counter
            self.consecutive_failures = 0


class TestSchedulerBackoffBehavior:
    """Test the full scheduler backoff behavior."""

    @pytest.mark.asyncio
    async def test_consecutive_transient_failures_trigger_backoff(self):
        """Test that consecutive transient failures trigger exponential backoff."""
        ctx = MockSchedulerContext()
        ctx.max_backoff = 0.1  # Small for testing

        # Simulate 10 consecutive connection errors
        error = APIConnectionError(message="Connection error", request=MagicMock())

        for i in range(10):
            await ctx.handle_rollout_result(success=False, error=error)

        # Should have accumulated backoff after the 5th failure
        assert ctx.consecutive_failures == 10
        assert ctx.total_backoff_time > 0

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter(self):
        """Test that a success resets the consecutive failure counter."""
        ctx = MockSchedulerContext()
        error = APIConnectionError(message="Connection error", request=MagicMock())

        # Accumulate some failures
        for _ in range(3):
            await ctx.handle_rollout_result(success=False, error=error)

        assert ctx.consecutive_failures == 3

        # Success resets the counter
        await ctx.handle_rollout_result(success=True, error=None)
        assert ctx.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_non_transient_error_resets_counter(self):
        """Test that non-transient errors reset the failure counter."""
        ctx = MockSchedulerContext()
        transient_error = APIConnectionError(message="Connection error", request=MagicMock())
        non_transient_error = ValueError("Invalid argument")

        # Accumulate transient failures
        for _ in range(3):
            await ctx.handle_rollout_result(success=False, error=transient_error)

        assert ctx.consecutive_failures == 3

        # Non-transient error resets the counter
        await ctx.handle_rollout_result(success=False, error=non_transient_error)
        assert ctx.consecutive_failures == 0
