import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
from verifiers.v1.clients.config import EvalClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import (
    _is_retryable_lora_error,
    init_nccl_broadcast,
    load_lora_adapter,
    setup_clients,
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


def test_load_lora_adapter_retries_wrapper_404_without_native_fallback():
    mock_client = AsyncMock()
    missing_wrapper = MagicMock()
    missing_wrapper.status_code = 404
    missing_wrapper.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not found",
        request=MagicMock(),
        response=missing_wrapper,
    )
    success_response = MagicMock()
    success_response.raise_for_status = MagicMock()
    mock_client.post.side_effect = [missing_wrapper, success_response]

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert [call.args for call in mock_client.post.await_args_list] == [
        ("/load_lora_adapter",),
        ("/load_lora_adapter",),
    ]


def test_init_nccl_broadcast_uses_native_collective_rpc_explicitly():
    clients = [AsyncMock() for _ in range(2)]
    for client in clients:
        response = MagicMock()
        response.raise_for_status = MagicMock()
        client.post.return_value = response

    asyncio.run(
        init_nccl_broadcast(
            clients,
            host="127.0.0.1",
            port=29519,
            timeout=1200,
            inference_world_size=4,
            engine_world_sizes=[2, 2],
            use_native_collective_rpc=True,
        )
    )

    for client, rank_offset in zip(clients, [0, 2], strict=True):
        client.post.assert_awaited_once_with(
            "/collective_rpc",
            json={
                "method": "init_broadcaster",
                "kwargs": {
                    "host": "127.0.0.1",
                    "port": 29519,
                    "rank_offset": rank_offset,
                    "inference_world_size": 4,
                    "timeout": 1200,
                    "quantize_in_weight_transfer": False,
                },
            },
        )


def test_init_nccl_broadcast_preserves_positional_quantize_argument():
    client = AsyncMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    client.post.return_value = response

    asyncio.run(init_nccl_broadcast([client], "127.0.0.1", 29519, 1200, 1, True))

    assert client.post.await_args.kwargs["json"]["quantize_in_weight_transfer"] is True


def test_init_nccl_broadcast_skips_missing_wrapper_without_native_probe():
    client = AsyncMock()
    missing_wrapper = MagicMock()
    missing_wrapper.status_code = 404
    missing_wrapper.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not found",
        request=MagicMock(),
        response=missing_wrapper,
    )
    client.post.return_value = missing_wrapper

    asyncio.run(init_nccl_broadcast([client], "127.0.0.1", 29519, 1200, 1))

    client.post.assert_awaited_once()
    assert client.post.await_args.args == ("/init_broadcaster",)


def test_setup_clients_assigns_renderer_and_dp_rank_headers():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
        dp_rank_count=2,
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
    assert [client.base_url for client in clients] == ["http://worker-a:8000/v1"] * 2
    assert [client.headers["X-data-parallel-rank"] for client in clients] == ["0", "1"]
    assert clients[0].headers["X-Test"] == "test"


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
