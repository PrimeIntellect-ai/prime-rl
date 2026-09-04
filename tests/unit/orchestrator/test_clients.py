import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from verifiers.v1.configs.client import EvalClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import (
    AdminPlane,
    _admin_post,
    _bounded_request,
    _is_retryable_lora_error,
    check_health,
    load_lora_adapter,
    setup_client,
)


class SlowDripResponse(httpx.AsyncByteStream):
    async def __aiter__(self):
        for _ in range(4):
            await asyncio.sleep(0.007)
            yield b"x"


class OversizedResponse(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"x" * 17


def test_admin_post_has_absolute_deadline_for_slow_drip_response_body():
    async def exercise():
        async def handler(request):
            return httpx.Response(200, stream=SlowDripResponse())

        async with httpx.AsyncClient(
            base_url="http://worker",
            transport=httpx.MockTransport(handler),
        ) as client:
            await _admin_post(client, "/pause", timeout_s=0.01)

    with pytest.raises(TimeoutError):
        asyncio.run(exercise())


def test_admin_request_rejects_oversized_response_body():
    async def exercise():
        async def handler(request):
            return httpx.Response(200, stream=OversizedResponse())

        async with httpx.AsyncClient(
            base_url="http://worker",
            transport=httpx.MockTransport(handler),
        ) as client:
            await _bounded_request(client, "GET", "/health", max_response_bytes=16)

    with pytest.raises(ValueError, match="response body exceeds"):
        asyncio.run(exercise())


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
    admin_plane = MagicMock(clients=[mock_client])
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter(admin_plane, "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


def test_admin_plane_initializes_nccl():
    admin_plane = AdminPlane(ClientConfig())
    client = AsyncMock()
    response = MagicMock()
    response.raise_for_status.return_value = None
    client.post.return_value = response
    admin_plane.clients = [client]

    asyncio.run(
        admin_plane.initialize_nccl(
            host="trainer",
            port=29501,
            timeout=1200,
            inference_world_size=1,
        )
    )

    client.post.assert_awaited_once_with(
        "/init_broadcaster",
        json={
            "host": "trainer",
            "port": 29501,
            "rank_offset": 0,
            "inference_world_size": 1,
            "timeout": 1200,
            "quantize_in_weight_transfer": False,
        },
    )
    asyncio.run(admin_plane.aclose())


def test_setup_client_creates_renderer_client():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url="http://worker-a:8000/v1",
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
    )

    renderer_settings = Qwen3VLRendererConfig()
    client = setup_client(
        client_config,
        client_type="renderer",
        renderer_config=renderer_settings,
    )

    assert client.type == "train"
    assert client.renderer == renderer_settings
    assert client.renderer_model_name is None
    assert client.base_url == "http://worker-a:8000/v1"
    assert "X-data-parallel-rank" not in client.headers
    assert client.headers["X-Test"] == "test"


def test_check_health_retries_non_success_status():
    client = AsyncMock()
    unavailable = httpx.Response(503, request=httpx.Request("GET", "http://worker/health"))
    healthy = httpx.Response(200, request=httpx.Request("GET", "http://worker/health"))
    client.base_url = httpx.URL("http://worker")

    with (
        patch(
            "prime_rl.orchestrator.clients._bounded_request", new=AsyncMock(side_effect=[unavailable, healthy])
        ) as request,
        patch("prime_rl.orchestrator.clients.asyncio.sleep", new=AsyncMock()),
    ):
        asyncio.run(check_health([client], interval=1, timeout=2))

    assert request.await_count == 2


def test_setup_client_assigns_renderer_model_name():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url="http://worker-a:8000/v1",
        api_key_var="PRIME_API_KEY",
    )

    client = setup_client(
        client_config,
        client_type="renderer",
        renderer_config=Qwen3VLRendererConfig(),
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    assert client.renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_setup_client_preserves_chat_client_defaults():
    client_config = ClientConfig(
        base_url="http://worker-a:8000/v1",
        api_key_var="PRIME_API_KEY",
    )

    client = setup_client(client_config)

    assert client == EvalClientConfig(
        api_key_var="PRIME_API_KEY",
        base_url="http://worker-a:8000/v1",
        headers={},
    )
