import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from verifiers.v1.configs.client import EvalClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import (
    AdminPlane,
    _is_retryable_admin_error,
    check_health,
    setup_client,
)


@pytest.mark.parametrize(
    ("status", "retry_not_found", "expected"),
    [(500, False, True), (503, False, True), (404, False, False), (404, True, True), (400, True, False)],
)
def test_is_retryable_admin_error(status, retry_not_found, expected):
    error = httpx.HTTPStatusError("error", request=MagicMock(), response=MagicMock(status_code=status))
    assert _is_retryable_admin_error(error, retry_not_found=retry_not_found) is expected


def test_admin_plane_loads_lora_adapter():
    admin_plane = AdminPlane(ClientConfig())
    client = AsyncMock()
    client.post.return_value = MagicMock()
    admin_plane.clients = [client]

    asyncio.run(admin_plane.load_lora_adapter("test-lora", Path("/test/path")))

    client.post.assert_awaited_once_with(
        "/load_lora_adapter",
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0),
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
    )
    asyncio.run(admin_plane.aclose())


def test_admin_plane_initializes_broadcaster():
    admin_plane = AdminPlane(ClientConfig())
    clients = [AsyncMock(), AsyncMock()]
    for client in clients:
        client.post.return_value = MagicMock()
    admin_plane.clients = clients

    asyncio.run(admin_plane.init_broadcaster(host="trainer", port=29501, timeout=1200, inference_world_size=4))

    for index, client in enumerate(clients):
        client.post.assert_awaited_once_with(
            "/init_broadcaster",
            timeout=httpx.Timeout(connect=10.0, read=1200.0, write=60.0, pool=10.0),
            json={
                "host": "trainer",
                "port": 29501,
                "rank_offset": index * 2,
                "inference_world_size": 4,
                "timeout": 1200,
                "session_id": "default",
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
    client.get.side_effect = [unavailable, healthy]
    client.base_url = httpx.URL("http://worker")

    with patch("prime_rl.orchestrator.clients.asyncio.sleep", new=AsyncMock()):
        asyncio.run(check_health([client], interval=1, timeout=2))

    assert client.get.await_count == 2


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
