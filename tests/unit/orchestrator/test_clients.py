import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from verifiers.v1.configs.client import EvalClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import (
    AdminPlane,
    _is_retryable_lora_error,
    check_health,
    setup_client,
)


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
    admin_plane = AdminPlane(ClientConfig())
    admin_plane.clients = [mock_client]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(admin_plane.load_lora_adapter("test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        params={"version": 0},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


def test_load_lora_adapter_readmits_recovered_engine():
    serving, recovered = AsyncMock(), AsyncMock()
    serving.post.return_value = recovered.post.return_value = MagicMock()
    recovered.get.return_value = MagicMock(**{"json.return_value": {"version": 3}})
    admin_plane = AdminPlane(ClientConfig())
    admin_plane.clients = [serving]
    admin_plane.group_of = {serving: (serving,), recovered: (recovered,)}
    admin_plane.evicted = [recovered]
    admin_plane.recovered = {recovered}
    admin_plane.evicting = True

    asyncio.run(admin_plane.load_lora_adapter("test-lora", Path("/test/path"), step=3))

    recovered.post.assert_awaited_once()
    assert admin_plane.clients == [serving, recovered]
    assert not admin_plane.evicted and not admin_plane.recovered


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
        },
    )
    asyncio.run(admin_plane.aclose())


def test_setup_client_creates_renderer_client():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url="http://worker-a:8000/v1",
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
        headers_from_env={"X-Token": "PROVIDER_TOKEN"},
        timeout={"read": 12},
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
    assert client.headers_from_env == {"X-Token": "PROVIDER_TOKEN"}
    assert client.timeout.read == 12


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


def test_engine_group_is_evicted_and_readmitted_as_one():
    config = ClientConfig(
        admin_base_url=["http://a:8100/v1", "http://b:8100/v1", "http://c:8100/v1"], admin_engine_groups=[0, 0, 1]
    )
    admin_plane = AdminPlane(config)
    admin_plane._router_clients = []
    a, b, c = admin_plane.clients
    asyncio.run(admin_plane._evict(a, "test"))
    assert admin_plane.clients == [c] and admin_plane.evicted == [a, b]

    admin_plane.recovered = {a}
    assert admin_plane._rejoining() == []
    admin_plane.recovered = {a, b}
    with patch("prime_rl.orchestrator.clients._weight_version", AsyncMock(side_effect=[5, 4, 5, 5])):
        asyncio.run(admin_plane._readmit([a, b], [a, b], step=5))
        assert admin_plane.clients == [c] and not admin_plane.recovered
        admin_plane.recovered = {a, b}
        asyncio.run(admin_plane._readmit([a, b], [a, b], step=5))
    assert admin_plane.clients == [c, a, b] and not admin_plane.evicted
