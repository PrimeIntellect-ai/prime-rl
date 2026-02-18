import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from prime_rl.utils.client import _is_retryable_lora_error, load_lora_adapter, setup_admin_clients
from prime_rl.utils.config import ClientConfig


async def _close_clients(admin_clients):
    await asyncio.gather(*[client.aclose() for client in admin_clients])


@pytest.mark.parametrize(("status_code", "expected"), [(404, True), (500, True), (400, False)])
def test_is_retryable_lora_error_for_http_errors(status_code, expected):
    response = MagicMock()
    response.status_code = status_code
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is expected


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


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


@pytest.mark.parametrize(
    ("admin_base_url", "expected"),
    [
        (["http://localhost:8000/v1", "http://localhost:8001/v1"], ["http://localhost:8000", "http://localhost:8001"]),
        (None, ["http://localhost:9000"]),
    ],
)
def test_setup_admin_clients_resolves_urls(admin_base_url, expected):
    config = ClientConfig(base_url=["http://localhost:9000/v1"], admin_base_url=admin_base_url)
    admin_clients = setup_admin_clients(config)
    try:
        assert [str(client.base_url) for client in admin_clients] == expected
    finally:
        asyncio.run(_close_clients(admin_clients))
