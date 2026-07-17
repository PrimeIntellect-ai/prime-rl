import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
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
                    "system_url": "http://prefill:8182",
                    "world_size": 2,
                    "routes": ["update/load_lora"],
                },
                {
                    "component": "backend",
                    "instance_id": 10,
                    "model": "Qwen/Qwen3-0.6B",
                    "admin_base_url": "http://decode:8120",
                    "system_url": "http://decode:8181",
                    "world_size": 2,
                    "routes": ["update/load_lora"],
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
    assert [worker.system_url for worker in workers] == [
        "http://decode:8181",
        "http://prefill:8182",
    ]
    assert all("update/load_lora" in worker.routes for worker in workers)


def test_parse_dynamo_workers_rejects_partial_lora_control_discovery():
    common = {
        "model": "Qwen/Qwen3-0.6B",
        "world_size": 1,
        "routes": ["update/load_lora"],
    }

    with pytest.raises(ValueError):
        _parse_dynamo_workers(
            {
                "workers": [
                    {
                        **common,
                        "component": "backend",
                        "instance_id": 10,
                        "admin_base_url": "http://decode:8120",
                        "system_url": "http://decode:8181",
                    },
                    {
                        **common,
                        "component": "prefill",
                        "instance_id": 20,
                        "admin_base_url": "http://prefill:8121",
                        "system_url": None,
                    },
                ]
            },
            "Qwen/Qwen3-0.6B",
        )


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
        with pytest.raises(ValueError):
            _parse_dynamo_workers({"namespace": "training", "workers": [invalid]}, "Qwen/Qwen3-0.6B")

    with pytest.raises(ValueError):
        _parse_dynamo_workers(
            {"namespace": "training", "workers": [valid, {**valid, "instance_id": 11}]},
            "Qwen/Qwen3-0.6B",
        )

    with pytest.raises(ValueError):
        _parse_dynamo_workers(
            {"namespace": "training", "workers": [valid, {**valid, "admin_base_url": "http://other:8120"}]},
            "Qwen/Qwen3-0.6B",
        )

    workers = _parse_dynamo_workers(
        {
            "namespace": "training",
            "workers": [
                {
                    **valid,
                    "admin_base_url": "http://user:password@decode:8120/admin",
                    "system_url": "http://sidecar:8181/control",
                }
            ],
        },
        "Qwen/Qwen3-0.6B",
    )
    assert workers[0].admin_base_url == "http://user:password@decode:8120/admin"
    assert workers[0].system_url == "http://sidecar:8181/control"


def test_rank_offsets_support_heterogeneous_engine_world_sizes():
    assert _rank_offsets([2, 2], inference_world_size=4) == [0, 2]
    assert _rank_offsets([1, 2], inference_world_size=3) == [0, 1]
    assert _rank_offsets([2, 2, 2, 2], inference_world_size=8) == [0, 2, 4, 6]

    with pytest.raises(ValueError):
        _rank_offsets([2, 2], inference_world_size=3)


def test_dynamo_inference_pool_discovers_admin_clients_from_one_snapshot():
    transient_response = MagicMock()
    transient_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Service unavailable",
        request=httpx.Request("GET", "http://frontend:8001/v1/rl/workers"),
        response=httpx.Response(503),
    )
    partial_response = MagicMock()
    partial_response.json.return_value = {
        "namespace": "training",
        "workers": [
            {
                "component": "backend",
                "instance_id": 10,
                "model": "Qwen/Qwen3-0.6B",
                "admin_base_url": "http://decode:8120",
                "system_url": "http://decode:8181",
                "world_size": 2,
                "routes": ["update/load_lora"],
            },
        ],
    }
    response = MagicMock()
    response.json.return_value = {
        "namespace": "training",
        "workers": [
            *partial_response.json.return_value["workers"],
            {
                "component": "backend",
                "instance_id": 11,
                "model": "Qwen/Qwen3-0.6B",
                "admin_base_url": "http://decode-1:8120",
                "system_url": "http://decode-1:8181",
                "world_size": 2,
                "routes": ["update/load_lora"],
            },
            {
                "component": "prefill",
                "instance_id": 20,
                "model": "Qwen/Qwen3-0.6B",
                "admin_base_url": "http://prefill:8121",
                "system_url": "http://prefill:8182",
                "world_size": 2,
                "routes": ["update/load_lora"],
            },
            {
                "component": "prefill",
                "instance_id": 21,
                "model": "Qwen/Qwen3-0.6B",
                "admin_base_url": "http://prefill-1:8121",
                "system_url": "http://prefill-1:8182",
                "world_size": 2,
                "routes": ["update/load_lora"],
            },
        ],
    }
    discovery_client = AsyncMock()
    discovery_client.get.side_effect = [transient_response, partial_response, response]
    context = AsyncMock()
    context.__aenter__.return_value = discovery_client

    client_factory = MagicMock(return_value=context)
    with patch("prime_rl.utils.client.AsyncClient", client_factory):
        pool = asyncio.run(
            DynamoInferencePool.from_config(
                ClientConfig(base_url=["http://frontend:8000/v1"], dynamo_base_url="http://frontend:8001"),
                model_name="Qwen/Qwen3-0.6B",
                expected_inference_world_size=8,
            )
        )

    assert discovery_client.get.await_count == 3
    discovery_client.get.assert_awaited_with("http://frontend:8001/v1/rl/workers")
    assert [call.kwargs["base_url"] for call in client_factory.call_args_list[1:]] == [
        "http://decode:8120",
        "http://decode-1:8120",
        "http://prefill:8121",
        "http://prefill-1:8121",
        "http://decode:8181",
        "http://decode-1:8181",
        "http://prefill:8182",
        "http://prefill-1:8182",
        "http://frontend:8000",
    ]
    assert pool.admin_world_sizes == [2, 2, 2, 2]


def test_dynamo_inference_pool_loads_lora_through_discovered_system_routes():
    workers = _parse_dynamo_workers(
        {
            "workers": [
                {
                    "component": "backend",
                    "instance_id": 10,
                    "model": "Qwen/Qwen3-0.6B",
                    "admin_base_url": "http://decode:8120",
                    "system_url": "http://decode:8181",
                    "world_size": 1,
                    "routes": ["update/load_lora"],
                }
            ]
        },
        "Qwen/Qwen3-0.6B",
    )
    with patch("prime_rl.utils.client.AsyncClient") as client_factory:
        admin_client = AsyncMock()
        system_client = AsyncMock()
        frontend_client = AsyncMock()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"status": "success"}
        admin_client.post.return_value = response
        system_client.post.return_value = response
        frontend_response = MagicMock()
        frontend_response.raise_for_status = MagicMock()
        frontend_response.json.return_value = {"data": [{"id": "math-r8"}]}
        frontend_client.get.return_value = frontend_response
        client_factory.side_effect = [admin_client, system_client, frontend_client]
        pool = DynamoInferencePool(
            ClientConfig(base_url=["http://frontend:8000/v1"]),
            workers,
            model_name="Qwen/Qwen3-0.6B",
        )

    asyncio.run(pool.update_weights(Path("/shared/adapter/step_1"), lora_name="math-r8", step=1))

    system_client.post.assert_awaited_once_with(
        "/v1/loras",
        json={
            "lora_name": "math-r8",
            "source": {"uri": "file:///shared/adapter/step_1"},
            "load_inplace": True,
        },
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )
    assert [call.args[0] for call in admin_client.post.await_args_list] == ["/pause", "/resume"]
    frontend_client.get.assert_awaited_once_with("/v1/models")


def test_dynamo_inference_pool_resumes_workers_when_lora_update_fails():
    workers = _parse_dynamo_workers(
        {
            "workers": [
                {
                    "component": "backend",
                    "instance_id": 10,
                    "model": "Qwen/Qwen3-0.6B",
                    "admin_base_url": "http://decode:8120",
                    "system_url": "http://decode:8181",
                    "world_size": 1,
                    "routes": ["update/load_lora"],
                }
            ]
        },
        "Qwen/Qwen3-0.6B",
    )
    with patch("prime_rl.utils.client.AsyncClient") as client_factory:
        admin_client = AsyncMock()
        system_client = AsyncMock()
        frontend_client = AsyncMock()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        admin_client.post.return_value = response
        system_client.post.side_effect = RuntimeError("LoRA update failed")
        client_factory.side_effect = [admin_client, system_client, frontend_client]
        pool = DynamoInferencePool(
            ClientConfig(base_url=["http://frontend:8000/v1"]),
            workers,
            model_name="Qwen/Qwen3-0.6B",
        )

    with pytest.raises(RuntimeError, match="LoRA update failed"):
        asyncio.run(pool.update_weights(Path("/shared/adapter/step_1"), lora_name="math-r8", step=1))

    assert [call.args[0] for call in admin_client.post.await_args_list] == ["/pause", "/resume"]
    frontend_client.get.assert_not_awaited()


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
    mock_clients = []
    for _ in range(4):
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
        mock_clients.append(mock_client)

    asyncio.run(
        init_nccl_broadcast(
            mock_clients,
            host="127.0.0.1",
            port=29519,
            timeout=1200,
            inference_world_size=8,
            engine_world_sizes=[2, 2, 2, 2],
        )
    )

    for mock_client, rank_offset in zip(mock_clients, [0, 2, 4, 6], strict=True):
        assert mock_client.post.await_args_list[0].args == ("/init_broadcaster",)
        assert mock_client.post.await_args_list[1].args == ("/collective_rpc",)
        assert mock_client.post.await_args_list[1].kwargs == {
            "json": {
                "method": "init_broadcaster",
                "args": ["127.0.0.1", 29519, rank_offset, 8, 1200, False],
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
