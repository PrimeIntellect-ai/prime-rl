import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from prime_rl.utils.client import (
    _is_retryable_admin_error,
    _is_retryable_lora_error,
    check_health,
    load_lora_adapter,
    maybe_check_has_model,
    reload_weights,
)


def _mock_response(status_code: int, payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}

    def _raise_for_status():
        if status_code >= 400:
            raise httpx.HTTPStatusError("error", request=MagicMock(), response=response)

    response.raise_for_status = MagicMock(side_effect=_raise_for_status)
    return response


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_is_retryable_admin_error_returns_true_for_502():
    response = MagicMock()
    response.status_code = 502
    error = httpx.HTTPStatusError("Bad gateway", request=MagicMock(), response=response)
    assert _is_retryable_admin_error(error) is True


def test_is_retryable_admin_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_admin_error(error) is False


def test_check_health_retries_until_success_status():
    mock_client = AsyncMock()
    mock_client.base_url = "http://localhost:8000"
    mock_client.get.side_effect = [
        _mock_response(503),
        _mock_response(200),
    ]

    asyncio.run(check_health([mock_client], interval=0.01, timeout=1))

    assert mock_client.get.call_count == 2


def test_maybe_check_has_model_retries_on_transient_status_then_succeeds():
    mock_client = AsyncMock()
    mock_client.base_url = "http://localhost:8000"
    mock_client.get.side_effect = [
        _mock_response(502),
        _mock_response(200, {"data": [{"id": "Qwen/Qwen3-0.6B"}]}),
    ]

    asyncio.run(
        maybe_check_has_model(
            [mock_client],
            "Qwen/Qwen3-0.6B",
            interval=0.01,
            timeout=1,
        )
    )

    assert mock_client.get.call_count == 2


def test_maybe_check_has_model_raises_on_model_mismatch_without_retry():
    mock_client = AsyncMock()
    mock_client.base_url = "http://localhost:8000"
    mock_client.get.return_value = _mock_response(200, {"data": [{"id": "another-model"}]})

    with pytest.raises(ValueError, match="was not found in the inference pool"):
        asyncio.run(
            maybe_check_has_model(
                [mock_client],
                "Qwen/Qwen3-0.6B",
                interval=0.01,
                timeout=1,
            )
        )

    assert mock_client.get.call_count == 1


def test_reload_weights_retries_on_502_then_succeeds():
    mock_client = AsyncMock()
    mock_client.post.side_effect = [
        _mock_response(502),
        _mock_response(200),
    ]

    asyncio.run(reload_weights([mock_client]))

    assert mock_client.post.call_count == 2


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
    )


def test_load_lora_adapter_retries_on_404_then_succeeds():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 404
    success_response = MagicMock()
    success_response.raise_for_status = MagicMock()

    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.HTTPStatusError("Not found", request=MagicMock(), response=error_response)
        return success_response

    mock_client.post = mock_post

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert call_count == 2


def test_load_lora_adapter_raises_non_retryable_error_immediately():
    mock_client = AsyncMock()

    error_response = MagicMock()
    error_response.status_code = 400
    mock_client.post.side_effect = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=error_response)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert exc_info.value.response.status_code == 400
    assert mock_client.post.call_count == 1
