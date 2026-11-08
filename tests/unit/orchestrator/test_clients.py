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
    load_lora_adapter,
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
    admin_plane = MagicMock(clients=[mock_client])
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter(admin_plane, "test-lora", Path("/test/path"), 3))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        params={"weight_version": 3},
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
        },
    )
    asyncio.run(admin_plane.aclose())


def test_pause_engines_pauses_prefill_before_decode():
    events = []

    def engine(name):
        async def post(path, **kwargs):
            events.append(f"{name} start")
            await asyncio.sleep(0.01 if name == "prefill" else 0)
            events.append(f"{name} done")
            return MagicMock()

        return MagicMock(post=post)

    admin_plane = AdminPlane(ClientConfig(), roles=["decode", "prefill"])
    admin_plane.clients = [engine("decode"), engine("prefill")]
    asyncio.run(admin_plane._pause([0, 1], step=1))

    assert events == ["prefill start", "prefill done", "decode start", "decode done"]


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


def _admin_plane(fail: set[int], hang: set[int], calls: list) -> AdminPlane:
    plane = AdminPlane(ClientConfig(), tolerate_restarts=True)

    def engine(index):
        async def post(path, **kwargs):
            calls.append((index, path))
            if index in fail:
                raise httpx.ConnectError("refused")
            if index in hang:
                await asyncio.sleep(1000)
            return MagicMock()

        return MagicMock(base_url=f"http://engine{index}:8000", post=post)

    plane.clients = [engine(index) for index in range(3)]
    plane._router_clients = [MagicMock(post=AsyncMock(return_value=MagicMock()))]
    return plane


def test_dead_engine_skips_updates_and_router_until_an_update_reaches_it_again():
    calls = []
    plane = _admin_plane(fail={1}, hang=set(), calls=calls)
    router = plane._router_clients[0].post
    asyncio.run(plane.update_weights(Path("/w"), transport="filesystem", step=1))
    assert plane.down == {1} and (1, "/resume") not in calls and (2, "/resume") in calls
    router.assert_awaited_with("/remove_worker", params={"url": "http://engine1:8000"}, timeout=60)

    plane.clients[1] = _admin_plane(fail=set(), hang=set(), calls=calls).clients[1]
    asyncio.run(plane._rejoin([1]))
    assert plane.rejoining == {1} and router.await_count == 1  # no traffic before the next update
    asyncio.run(plane.update_weights(Path("/w"), transport="filesystem", step=2))
    assert plane.rejoining == set() and (1, "/update_weights") in calls
    router.assert_awaited_with("/add_worker", params={"url": "http://engine1:8000"}, timeout=60)


def test_update_stops_waiting_for_an_engine_that_goes_down():
    calls = []
    plane = _admin_plane(fail=set(), hang={1}, calls=calls)

    async def run():
        update = asyncio.create_task(plane.update_weights(Path("/w"), transport="filesystem", step=1))
        await asyncio.sleep(0.5)
        await plane.mark_down(1, "missed two health checks")
        await asyncio.wait_for(update, 5)

    asyncio.run(run())
    assert (1, "/resume") not in calls and (0, "/resume") in calls
