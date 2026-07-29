import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from verifiers.v1.clients.config import EvalClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import _is_retryable_lora_error, check_health, load_lora_adapter, setup_clients


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


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


def test_setup_clients_creates_one_renderer_client_per_url():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1", "http://worker-b:8000/v1"],
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )

    renderer_settings = Qwen3VLRendererConfig()
    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=renderer_settings,
    )

    assert [client.type for client in clients] == ["train", "train"]
    assert [client.renderer for client in clients] == [renderer_settings, renderer_settings]
    assert [client.renderer_model_name for client in clients] == [None, None]
    assert [client.base_url for client in clients] == [
        "http://worker-a:8000/v1",
        "http://worker-b:8000/v1",
    ]
    assert all("X-data-parallel-rank" not in client.headers for client in clients)
    assert clients[0].headers["X-Test"] == "test"


def test_check_health_retries_non_success_status():
    client = AsyncMock()
    unavailable = httpx.Response(503, request=httpx.Request("GET", "http://worker/health"))
    healthy = httpx.Response(200, request=httpx.Request("GET", "http://worker/health"))
    client.get.side_effect = [unavailable, healthy]
    client.base_url = httpx.URL("http://worker")

    with patch("prime_rl.utils.client.asyncio.sleep", new=AsyncMock()):
        asyncio.run(check_health([client], interval=1, timeout=2))

    assert client.get.await_count == 2


def test_setup_clients_assigns_renderer_model_name():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=Qwen3VLRendererConfig(),
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    assert clients[0].renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_setup_clients_preserves_chat_client_defaults():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(client_config)

    assert clients == [
        EvalClientConfig(
            api_key_var="PRIME_API_KEY",
            base_url="http://worker-a:8000/v1",
            headers={},
        )
    ]
