import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from verifiers.v1.configs.client import EvalClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import (
    AdminClients,
    _is_retryable_lora_error,
    _rank_offsets,
    check_health,
    init_nccl_broadcast,
    init_nixl_broadcast,
    load_lora_adapter,
    setup_client,
    setup_policy_admin_clients,
    update_weights,
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
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


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


def successful_response() -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    return response


def test_policy_admin_factory_preserves_static_clients():
    admin = setup_policy_admin_clients(ClientConfig(), "Qwen/Qwen3-0.6B")

    assert isinstance(admin, AdminClients)
    assert admin.worker_world_sizes is None
    assert admin.use_collective_rpc is False
    asyncio.run(admin.aclose())


def test_rank_offsets_are_cumulative_for_heterogeneous_workers():
    assert _rank_offsets((2, 1, 3), inference_world_size=6) == (0, 2, 3)


def test_rank_offsets_reject_world_size_mismatch():
    with pytest.raises(ValueError, match="do not match"):
        _rank_offsets((2, 1), inference_world_size=4)


def test_collective_rpc_nccl_init_uses_exact_worker_offsets():
    clients = [AsyncMock(), AsyncMock()]
    for client in clients:
        client.post.return_value = successful_response()

    asyncio.run(
        init_nccl_broadcast(
            clients,
            host="trainer",
            port=29501,
            timeout=1200,
            inference_world_size=3,
            worker_world_sizes=(2, 1),
            use_collective_rpc=True,
        )
    )

    clients[0].post.assert_awaited_once_with(
        "/collective_rpc",
        json={
            "method": "init_broadcaster",
            "timeout": 1200,
            "args": ["trainer", 29501, 0, 3, 1200, False, "default"],
            "kwargs": {},
        },
    )
    clients[1].post.assert_awaited_once_with(
        "/collective_rpc",
        json={
            "method": "init_broadcaster",
            "timeout": 1200,
            "args": ["trainer", 29501, 2, 3, 1200, False, "default"],
            "kwargs": {},
        },
    )


def test_collective_rpc_nixl_init_uses_exact_worker_offsets():
    clients = [AsyncMock(), AsyncMock()]
    for client in clients:
        client.post.return_value = successful_response()

    asyncio.run(
        init_nixl_broadcast(
            clients,
            host="model-express",
            port=5555,
            timeout=1200,
            inference_world_size=3,
            session_id="run-a",
            worker_world_sizes=(2, 1),
            use_collective_rpc=True,
        )
    )

    for client, rank_offset, engine_world_size in zip(clients, (0, 2), (2, 1)):
        client.post.assert_awaited_once_with(
            "/collective_rpc",
            json={
                "method": "init_broadcaster",
                "timeout": 1200,
                "args": [
                    "model-express",
                    5555,
                    rank_offset,
                    3,
                    1200,
                    False,
                    "run-a",
                    engine_world_size,
                ],
                "kwargs": {},
            },
        )


def test_collective_rpc_weight_update_uses_prime_worker_method(tmp_path):
    client = AsyncMock()
    client.post.return_value = successful_response()
    step_dir = tmp_path / "step_1"

    with patch("prime_rl.orchestrator.clients.asyncio.wait_for", wraps=asyncio.wait_for) as bounded_wait:
        asyncio.run(update_weights([client], step_dir, step=1, use_collective_rpc=True))

    assert [call.args[0] for call in client.post.await_args_list] == ["/pause", "/collective_rpc", "/resume"]
    collective_call = client.post.await_args_list[1]
    assert collective_call.kwargs["json"] == {
        "method": "update_weights_from_path",
        "timeout": 720.0,
        "args": [step_dir.as_posix()],
        "kwargs": {},
    }
    assert bounded_wait.await_args.kwargs["timeout"] == 730.0


def test_collective_rpc_nixl_update_allows_no_weight_directory():
    client = AsyncMock()
    client.post.return_value = successful_response()

    asyncio.run(update_weights([client], None, step=2, use_collective_rpc=True))

    assert [call.args[0] for call in client.post.await_args_list] == [
        "/pause",
        "/collective_rpc",
        "/resume",
    ]
    assert client.post.await_args_list[1].kwargs["json"] == {
        "method": "update_weights_from_path",
        "timeout": 720.0,
        "args": [None],
        "kwargs": {},
    }


def test_collective_rpc_update_failure_keeps_engines_paused(tmp_path):
    client = AsyncMock()
    ok = successful_response()
    failed = successful_response()
    failed.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("POST", "http://worker/collective_rpc"),
        response=httpx.Response(400, request=httpx.Request("POST", "http://worker/collective_rpc")),
    )
    client.post.side_effect = [ok, failed]

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(update_weights([client], tmp_path / "step_1", step=1, use_collective_rpc=True))

    assert [call.args[0] for call in client.post.await_args_list] == ["/pause", "/collective_rpc"]


def test_collective_rpc_transport_timeout_is_bounded_and_keeps_engines_paused(tmp_path):
    client = AsyncMock()
    client.post.return_value = successful_response()

    async def expire(awaitable, *, timeout):
        awaitable.close()
        raise TimeoutError(f"expired after {timeout}")

    with (
        patch("prime_rl.orchestrator.clients.asyncio.wait_for", side_effect=expire) as bounded_wait,
        pytest.raises(TimeoutError, match="expired after 730.0"),
    ):
        asyncio.run(update_weights([client], tmp_path / "step_1", step=1, use_collective_rpc=True))

    assert [call.args[0] for call in client.post.call_args_list] == ["/pause", "/collective_rpc"]
    assert bounded_wait.call_count == 1
