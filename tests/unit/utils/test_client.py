import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from verifiers.v1.clients.config import EvalClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import (
    DynamoInferencePool,
    _is_retryable_lora_error,
    _parse_dynamo_workers,
    _rank_offsets,
    init_nccl_broadcast,
    load_lora_adapter,
    setup_clients,
)


def test_parse_dynamo_workers_uses_stable_identity_order_not_url_order():
    workers = _parse_dynamo_workers(
        {
            "namespace": "training",
            "workers": [
                {
                    "component": "prefill",
                    "instance_id": 20,
                    "model": "Qwen/Qwen3-0.6B",
                    "admin_base_url": "http://prefill:8121",
                    "world_size": 2,
                    "routes": [],
                },
                {
                    "component": "backend",
                    "instance_id": 10,
                    "model": "Qwen/Qwen3-0.6B",
                    "admin_base_url": "http://decode:8120",
                    "world_size": 2,
                    "routes": [],
                },
            ],
        },
        model_name="Qwen/Qwen3-0.6B",
    )

    assert [worker.admin_base_url for worker in workers] == [
        "http://decode:8120",
        "http://prefill:8121",
    ]
    assert [worker.world_size for worker in workers] == [2, 2]


def test_parse_dynamo_workers_fails_closed_on_partial_or_duplicate_snapshot():
    valid = {
        "component": "backend",
        "instance_id": 10,
        "model": "Qwen/Qwen3-0.6B",
        "admin_base_url": "http://decode:8120",
        "world_size": 2,
        "routes": [],
    }

    for invalid in (
        {**valid, "error": "probe timed out"},
        {**valid, "admin_base_url": None},
        {**valid, "world_size": 0},
        {**valid, "model": "other/model"},
    ):
        try:
            _parse_dynamo_workers({"namespace": "training", "workers": [invalid]}, "Qwen/Qwen3-0.6B")
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid worker snapshot: {invalid}")

    try:
        _parse_dynamo_workers(
            {"namespace": "training", "workers": [valid, {**valid, "instance_id": 11}]},
            "Qwen/Qwen3-0.6B",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("accepted duplicate admin endpoint")

    try:
        _parse_dynamo_workers(
            {"namespace": "training", "workers": [valid, {**valid, "admin_base_url": "http://other:8120"}]},
            "Qwen/Qwen3-0.6B",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("accepted duplicate worker identity")

    try:
        _parse_dynamo_workers(
            {
                "namespace": "training",
                "workers": [{**valid, "admin_base_url": "http://user:password@decode:8120"}],
            },
            "Qwen/Qwen3-0.6B",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("accepted credentialed admin URL")


def test_rank_offsets_support_heterogeneous_engine_world_sizes():
    assert _rank_offsets([2, 2], inference_world_size=4) == [0, 2]
    assert _rank_offsets([1, 2], inference_world_size=3) == [0, 1]

    try:
        _rank_offsets([2, 2], inference_world_size=3)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a world-size mismatch")


def test_dynamo_inference_pool_discovers_admin_clients_from_one_snapshot():
    response = MagicMock()
    response.json.return_value = {
        "namespace": "training",
        "workers": [
            {
                "component": "backend",
                "instance_id": 10,
                "model": "Qwen/Qwen3-0.6B",
                "admin_base_url": "http://decode:8120",
                "world_size": 2,
                "routes": [],
            },
            {
                "component": "prefill",
                "instance_id": 20,
                "model": "Qwen/Qwen3-0.6B",
                "admin_base_url": "http://prefill:8121",
                "world_size": 2,
                "routes": [],
            },
        ],
    }
    discovery_client = AsyncMock()
    empty_response = MagicMock()
    empty_response.json.return_value = {"namespace": "training", "workers": []}
    discovery_client.get.side_effect = [empty_response, response]
    context = AsyncMock()
    context.__aenter__.return_value = discovery_client

    client_factory = MagicMock(return_value=context)
    with patch("prime_rl.utils.client.AsyncClient", client_factory):
        pool = asyncio.run(
            DynamoInferencePool.from_config(
                ClientConfig(base_url=["http://frontend:8000/v1"], rl_base_url="http://frontend:8001"),
                model_name="Qwen/Qwen3-0.6B",
            )
        )

    assert discovery_client.get.await_count == 2
    discovery_client.get.assert_awaited_with("http://frontend:8001/v1/rl/workers")
    assert [call.kwargs["base_url"] for call in client_factory.call_args_list[1:]] == [
        "http://decode:8120",
        "http://prefill:8121",
    ]
    assert pool.admin_world_sizes == [2, 2]


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


def test_load_lora_adapter_falls_back_to_native_vllm_route():
    mock_client = AsyncMock()
    missing_wrapper = MagicMock()
    missing_wrapper.status_code = 404
    missing_wrapper.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not found",
        request=MagicMock(),
        response=missing_wrapper,
    )
    native_response = MagicMock()
    native_response.raise_for_status = MagicMock()
    mock_client.post.side_effect = [missing_wrapper, native_response]

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    assert mock_client.post.await_args_list[0].args == ("/load_lora_adapter",)
    assert mock_client.post.await_args_list[1].args == ("/v1/load_lora_adapter",)
    assert mock_client.post.await_args_list[1].kwargs["json"] == {
        "lora_name": "test-lora",
        "lora_path": "/test/path",
        "load_inplace": True,
    }


def test_init_nccl_broadcast_falls_back_to_native_collective_rpc():
    mock_client = AsyncMock()
    missing_wrapper = MagicMock()
    missing_wrapper.status_code = 404
    missing_wrapper.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not found",
        request=MagicMock(),
        response=missing_wrapper,
    )
    native_response = MagicMock()
    native_response.raise_for_status = MagicMock()
    mock_client.post.side_effect = [missing_wrapper, native_response]

    asyncio.run(
        init_nccl_broadcast(
            [mock_client],
            host="127.0.0.1",
            port=29519,
            timeout=1200,
            engine_world_sizes=[2],
        )
    )

    assert mock_client.post.await_args_list[0].args == ("/init_broadcaster",)
    assert mock_client.post.await_args_list[1].args == ("/collective_rpc",)
    assert mock_client.post.await_args_list[1].kwargs == {
        "json": {
            "method": "init_broadcaster",
            "args": ["127.0.0.1", 29519, 0, 2, 1200, False],
        }
    }


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
