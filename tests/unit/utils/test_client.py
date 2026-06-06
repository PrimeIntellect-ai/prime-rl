import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import verifiers as vf

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import (
    DynamoAdminAPI,
    _dynamo_rl_discovery_base_urls,
    _is_retryable_lora_error,
    discover_dynamo_admin_base_urls,
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


def test_dynamo_load_lora_adapter_uses_existing_lora_engine_route():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"status": "success"}
    mock_client.post.return_value = mock_response

    asyncio.run(
        DynamoAdminAPI().load_lora_adapter(
            mock_client,
            "test-lora",
            "/test/path",
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
        )
    )

    mock_client.post.assert_called_once_with(
        "/engine/load_lora",
        json={
            "lora_name": "test-lora",
            "source": {"uri": Path("/test/path").absolute().as_uri()},
        },
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

    assert [client.client_type for client in clients] == ["renderer", "renderer"]
    assert [client.renderer_config for client in clients] == [renderer_settings, renderer_settings]
    assert [client.renderer_model_name for client in clients] == [None, None]
    assert [client.api_base_url for client in clients] == ["http://worker-a:8000/v1"] * 2
    assert [client.extra_headers["X-data-parallel-rank"] for client in clients] == ["0", "1"]
    assert clients[0].extra_headers["X-Test"] == "test"
    assert clients[0].extra_headers_from_state == {"X-Session-ID": "session_id"}


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
        vf.ClientConfig(
            client_idx=0,
            client_type="openai_chat_completions",
            api_key_var="PRIME_API_KEY",
            api_base_url="http://worker-a:8000/v1",
            timeout=client_config.timeout,
            connect_timeout=client_config.connect_timeout,
            max_connections=8192,
            max_keepalive_connections=8192,
            max_retries=10,
            extra_headers={},
            extra_headers_from_state={},
        )
    ]


def test_dynamo_rl_discovery_base_urls_derive_from_base_url(monkeypatch):
    monkeypatch.setenv("DYN_RL_PORT", "18001")
    client_config = ClientConfig(
        base_url=["http://frontend.local:8000/v1"],
        backend="dynamo",
    )

    assert _dynamo_rl_discovery_base_urls(client_config) == ["http://frontend.local:18001"]


def test_dynamo_rl_discovery_base_urls_honor_explicit_config():
    client_config = ClientConfig(
        base_url=["http://frontend.local:8000/v1"],
        backend="dynamo",
        rl_base_url=["http://frontend.local:8001/v1"],
    )

    assert _dynamo_rl_discovery_base_urls(client_config) == ["http://frontend.local:8001/v1"]


def test_discover_dynamo_admin_base_urls_reads_workers(monkeypatch):
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "workers": [
                    {"system_url": "http://worker-0:8081"},
                    {"system_url": "http://worker-0:8081"},
                    {"system_url": "http://worker-1:8081"},
                    {},
                ]
            }

    class FakeClient:
        def __init__(self, *, base_url, headers, timeout):
            calls.append(("init", base_url, headers, timeout))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, path):
            calls.append(("get", path))
            return FakeResponse()

    monkeypatch.setattr("prime_rl.utils.client.httpx.Client", FakeClient)

    client_config = ClientConfig(
        base_url=["http://frontend.local:8000/v1"],
        backend="dynamo",
        rl_base_url=["http://frontend.local:8001/v1"],
    )

    assert discover_dynamo_admin_base_urls(client_config) == [
        "http://worker-0:8081",
        "http://worker-1:8081",
    ]
    assert calls[0][0:2] == ("init", "http://frontend.local:8001")
    assert ("get", "/v1/rl/workers") in calls
