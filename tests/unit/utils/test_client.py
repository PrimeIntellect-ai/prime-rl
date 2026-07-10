import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from verifiers.v1.clients.config import EvalClientConfig

from prime_rl.configs.shared import ClientConfig, ElasticConfig
from prime_rl.inference.dynamo_admin import DynamoTopology, DynamoWorker
from prime_rl.utils.client import (
    DynamoInferencePool,
    StaticInferencePool,
    _is_retryable_lora_error,
    check_health,
    load_lora_adapter,
    maybe_check_has_model,
    setup_clients,
    setup_inference_pool,
)
from prime_rl.utils.policy_client_config import policy_client_config_from_environment


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


def test_native_pool_preserves_admin_base_url_and_reuses_admin_clients():
    pool = StaticInferencePool(
        ClientConfig(
            base_url=["http://router:8000/v1"],
            admin_base_url=["http://worker:8001/v1"],
        ),
        model_name="test-model",
    )

    assert pool._frontend_admin_clients is pool._admin_clients
    assert [str(client.base_url).rstrip("/") for client in pool.admin_clients] == ["http://worker:8001"]
    asyncio.run(pool.stop())


def test_setup_inference_pool_selects_dynamo_pool_once():
    pool = asyncio.run(
        setup_inference_pool(
            ClientConfig(
                base_url=["http://frontend:8000/v1"],
                admin_api="dynamo",
                dynamo_worker_roles=("agg",),
                dynamo_gpus_per_worker=1,
            ),
            model_name="test-model",
        )
    )

    assert isinstance(pool, DynamoInferencePool)
    asyncio.run(pool.stop())


def test_generated_dgd_topology_selects_dynamo_pool_without_mutating_config(monkeypatch: pytest.MonkeyPatch):
    client_config = ClientConfig()
    original_config = client_config.model_copy(deep=True)
    monkeypatch.setenv(
        "DYN_RL_TOPOLOGY",
        """{"schema_version":1,"admin_api":"dynamo","base_url":["http://frontend:8000/v1"],"rl_base_url":["http://frontend-rl:8001"],"dynamo_worker_roles":["prefill","decode"],"dynamo_gpus_per_worker":1}""",
    )

    resolved = policy_client_config_from_environment(client_config)
    pool = asyncio.run(setup_inference_pool(resolved, model_name="test-model"))

    assert isinstance(pool, DynamoInferencePool)
    assert pool.admin_api == "dynamo"
    assert resolved.base_url == ["http://frontend:8000/v1"]
    assert resolved.rl_base_url == ["http://frontend-rl:8001"]
    assert client_config == original_config
    asyncio.run(pool.stop())


def test_generated_dgd_topology_rejects_explicit_client_conflict(monkeypatch: pytest.MonkeyPatch):
    client_config = ClientConfig(admin_api="vllm")
    monkeypatch.setenv(
        "DYN_RL_TOPOLOGY",
        """{"schema_version":1,"admin_api":"dynamo","base_url":["http://frontend:8000/v1"],"rl_base_url":["http://frontend-rl:8001"],"dynamo_worker_roles":["agg"],"dynamo_gpus_per_worker":1}""",
    )

    with pytest.raises(ValueError, match="admin_api.*conflicts"):
        policy_client_config_from_environment(client_config)


@pytest.mark.asyncio
async def test_setup_inference_pool_rejects_dynamo_elastic_before_pool_selection():
    client_config = ClientConfig(
        base_url=["http://frontend:8000/v1"],
        admin_api="dynamo",
        dynamo_worker_roles=("agg",),
        dynamo_gpus_per_worker=1,
        elastic=ElasticConfig(hostname="inference.example"),
    )
    original_config = client_config.model_copy(deep=True)

    with patch("prime_rl.utils.elastic.ElasticInferencePool.from_config", new=AsyncMock()) as from_config:
        with pytest.raises(ValueError, match="Dynamo admin API does not support elastic inference pools"):
            await setup_inference_pool(client_config, model_name="test-model")

    from_config.assert_not_awaited()
    assert client_config == original_config


@pytest.mark.asyncio
async def test_model_registration_retries_transient_status_and_empty_models():
    client = AsyncMock()
    client.base_url = httpx.URL("http://frontend:8000")
    request = httpx.Request("GET", "http://frontend:8000/v1/models")
    conflict = httpx.Response(409, request=request)
    unavailable = httpx.Response(
        503,
        request=request,
    )
    empty = httpx.Response(200, json={"data": []}, request=request)
    ready = httpx.Response(200, json={"data": [{"id": "test-model"}]}, request=request)
    client.get.side_effect = [httpx.ConnectError("not listening", request=request), conflict, unavailable, empty, ready]

    await maybe_check_has_model([client], "test-model", timeout=1, interval=0)

    assert client.get.await_count == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 404])
async def test_model_registration_fails_permanent_http_status_immediately(status_code: int):
    client = AsyncMock()
    client.base_url = httpx.URL("http://frontend:8000")
    client.get.return_value = httpx.Response(
        status_code,
        request=httpx.Request("GET", "http://frontend:8000/v1/models"),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await maybe_check_has_model([client], "test-model", timeout=1, interval=0)

    client.get.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(
            200,
            content=b"not-json",
            headers={"content-type": "application/json"},
            request=httpx.Request("GET", "http://frontend:8000/v1/models"),
        ),
        httpx.Response(
            200,
            json={"data": {}},
            request=httpx.Request("GET", "http://frontend:8000/v1/models"),
        ),
        httpx.Response(
            200,
            json={"data": [{"object": "model"}]},
            request=httpx.Request("GET", "http://frontend:8000/v1/models"),
        ),
    ],
    ids=["invalid-json", "data-not-list", "model-id-missing"],
)
async def test_model_registration_fails_invalid_response_immediately(response: httpx.Response):
    client = AsyncMock()
    client.base_url = httpx.URL("http://frontend:8000")
    client.get.return_value = response

    with pytest.raises(ValueError, match=r"Invalid /v1/models response"):
        await maybe_check_has_model([client], "test-model", timeout=1, interval=0)

    client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_strict_health_retries_only_transient_failures():
    client = AsyncMock()
    client.base_url = httpx.URL("http://frontend:8000")
    request = httpx.Request("GET", "http://frontend:8000/health")
    client.get.side_effect = [
        httpx.ConnectError("not listening", request=request),
        httpx.Response(429, request=request),
        httpx.Response(503, request=request),
        httpx.Response(200, request=request),
    ]

    await check_health([client], timeout=1, interval=0, strict=True)

    assert client.get.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 404])
async def test_strict_health_fails_permanent_http_status_immediately(status_code: int):
    client = AsyncMock()
    client.base_url = httpx.URL("http://frontend:8000")
    client.get.return_value = httpx.Response(
        status_code,
        request=httpx.Request("GET", "http://frontend:8000/health"),
    )

    with pytest.raises(httpx.HTTPStatusError):
        await check_health([client], timeout=1, interval=0, strict=True)

    client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_registration_timeout_reports_last_error():
    client = AsyncMock()
    client.base_url = httpx.URL("http://frontend:8000")
    client.get.return_value = httpx.Response(
        503,
        request=httpx.Request("GET", "http://frontend:8000/v1/models"),
    )

    with pytest.raises(TimeoutError, match=r"test-model.*frontend:8000.*503 Service Unavailable"):
        await maybe_check_has_model([client], "test-model", timeout=0.02, interval=0.001)

    assert client.get.await_count > 1


@pytest.mark.asyncio
async def test_dynamo_readiness_shares_one_monotonic_deadline(monkeypatch: pytest.MonkeyPatch):
    pool = DynamoInferencePool(
        ClientConfig(
            base_url=["http://frontend:8000/v1"],
            admin_api="dynamo",
            dynamo_worker_roles=("agg",),
            dynamo_gpus_per_worker=2,
        ),
        model_name="test-model",
    )
    observed_timeouts: list[float] = []
    worker = DynamoWorker(
        instance_id=11,
        component="backend",
        role="agg",
        system_url="http://worker:8081",
        model="test-model",
        routes=frozenset(
            {
                "init_weights_update_group",
                "pause_generation",
                "resume_generation",
                "update_weights_from_disk",
                "update_weights_from_distributed",
            }
        ),
    )

    async def health(_clients, *, timeout, strict=False, **_kwargs):
        assert strict is True
        observed_timeouts.append(timeout)
        await asyncio.sleep(0.01)

    async def models(_clients, _model_name, *, skip_model_check, timeout):
        assert skip_model_check is False
        observed_timeouts.append(timeout)
        await asyncio.sleep(0.01)

    async def discover(_clients, timeout, *, model_name, topology):
        assert model_name == "test-model"
        assert topology == DynamoTopology(roles=("agg",), gpus_per_worker=2)
        observed_timeouts.append(timeout)
        await asyncio.sleep(0.01)
        return (worker,)

    monkeypatch.setattr("prime_rl.utils.client.check_health", health)
    monkeypatch.setattr("prime_rl.utils.client.maybe_check_has_model", models)
    monkeypatch.setattr("prime_rl.utils.client.discover_workers", discover)
    try:
        await pool.wait_for_ready("test-model", timeout=1)
    finally:
        await pool.stop()

    assert len(observed_timeouts) == 4
    assert all(later < earlier for earlier, later in zip(observed_timeouts, observed_timeouts[1:]))
