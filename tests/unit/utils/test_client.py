import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import verifiers as vf

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import (
    _is_retryable_lora_error,
    get_admin_backend,
    init_nccl_broadcast,
    load_lora_adapter,
    setup_admin_clients,
    setup_clients,
    update_weights,
)
from prime_rl.utils.weight_broadcast import NCCL_BROADCAST_MARKER, NCCL_READY_MARKER


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


def test_setup_admin_clients_tags_backend():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        admin_backend="sglang",
    )

    clients = setup_admin_clients(client_config)

    assert len(clients) == 1
    assert str(clients[0].base_url) == "http://worker-a:8000"
    assert get_admin_backend(clients[0]) == "sglang"
    asyncio.run(clients[0].aclose())


def test_update_weights_uses_sglang_reload_endpoint(tmp_path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/update_weights_from_disk":
            return httpx.Response(200, json={"success": True, "message": "ok"})
        return httpx.Response(404)

    client = httpx.AsyncClient(base_url="http://worker-a:8000", transport=httpx.MockTransport(handler))
    setattr(client, "prime_rl_admin_backend", "sglang")

    asyncio.run(update_weights([client], tmp_path / "step_1"))
    asyncio.run(client.aclose())

    assert [request.url.path for request in requests] == ["/update_weights_from_disk"]
    assert requests[0].content == b'{"model_path":"' + (tmp_path / "step_1").as_posix().encode() + b'"}'


def test_update_weights_uses_sglang_nccl_endpoint_when_marker_exists(tmp_path):
    requests: list[httpx.Request] = []
    (tmp_path / NCCL_BROADCAST_MARKER).touch()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/update_weights":
            return httpx.Response(200, json={"success": True, "message": "ok"})
        return httpx.Response(404)

    client = httpx.AsyncClient(base_url="http://worker-a:8000", transport=httpx.MockTransport(handler))
    setattr(client, "prime_rl_admin_backend", "sglang")

    asyncio.run(update_weights([client], tmp_path, step=7))
    asyncio.run(client.aclose())

    assert (tmp_path / NCCL_READY_MARKER).exists()
    assert [request.url.path for request in requests] == ["/update_weights"]
    assert requests[0].content == b'{"weight_dir":"' + tmp_path.as_posix().encode() + b'"}'


def test_init_nccl_broadcast_uses_sglang_broadcaster_endpoint():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/init_broadcaster":
            return httpx.Response(200, json={"success": True, "message": "ok"})
        return httpx.Response(404)

    client = httpx.AsyncClient(base_url="http://worker-a:8000", transport=httpx.MockTransport(handler))
    setattr(client, "prime_rl_admin_backend", "sglang")

    asyncio.run(init_nccl_broadcast([client], "localhost", 29501, 120, inference_world_size=2))
    asyncio.run(client.aclose())

    assert [request.url.path for request in requests] == ["/init_broadcaster"]
    payload = requests[0].content
    assert payload == (b'{"host":"localhost","port":29501,"rank_offset":0,"inference_world_size":2,"timeout":120}')


def test_setup_clients_assigns_renderer_and_dp_rank_headers():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
        dp_rank_count=2,
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )

    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_name="qwen3_vl",
    )

    assert [client.client_type for client in clients] == ["renderer", "renderer"]
    assert [client.renderer for client in clients] == ["qwen3_vl", "qwen3_vl"]
    assert [client.renderer_model_name for client in clients] == [None, None]
    assert [client.api_base_url for client in clients] == ["http://worker-a:8000/v1"] * 2
    assert [client.extra_headers["X-data-parallel-rank"] for client in clients] == ["0", "1"]
    assert clients[0].extra_headers["X-Test"] == "test"
    assert clients[0].extra_headers_from_state == {"X-Session-ID": "session_id"}


def test_setup_clients_assigns_renderer_model_name():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_name="qwen3_vl",
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
            renderer="auto",
            renderer_model_name=None,
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
