import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import httpx
import pytest
from verifiers.v1.configs.client import EvalClientConfig

import prime_rl.orchestrator.clients as clients_module
from prime_rl.configs.shared import ADMIN_TOKEN_HEADER, ClientConfig, VLLMClientConfig
from prime_rl.orchestrator.clients import (
    PAUSE_ATTEMPT_TIMEOUT_S,
    AdminPlane,
    _admin_post,
    _is_retryable_lora_error,
    _pause_engines,
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


def test_admin_plane_sends_the_admin_token_to_engines_only(monkeypatch):
    monkeypatch.setenv("PRIME_RL_ADMIN_TOKEN", "secret")
    config = VLLMClientConfig(base_url="http://router:8000/v1", admin_base_url=["http://engine:8100/v1"])
    admin_plane = AdminPlane(config)

    assert [client.headers.get(ADMIN_TOKEN_HEADER) for client in admin_plane.clients] == ["secret"]
    assert [client.headers.get(ADMIN_TOKEN_HEADER) for client in admin_plane._router_clients] == [None]
    asyncio.run(admin_plane.aclose())


def test_update_draft_weights_fails_fast_on_a_failed_load():
    admin_plane = AdminPlane(ClientConfig())
    request = httpx.Request("POST", "http://engine/update_draft_weights")
    failed = MagicMock()
    failed.raise_for_status.side_effect = httpx.HTTPStatusError(
        "load failed", request=request, response=httpx.Response(500, request=request)
    )
    client = create_autospec(httpx.AsyncClient, instance=True)
    client.post.return_value = failed
    admin_plane.clients = [client]

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(admin_plane.update_draft_weights(Path("/drafts/v2")))

    client.post.assert_awaited_once()
    assert client.post.await_args.args == ("/update_draft_weights",)
    assert client.post.await_args.kwargs["json"] == {"weight_dir": "/drafts/v2"}
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

    asyncio.run(_pause_engines([engine("decode"), engine("prefill")], step=1, roles=["decode", "prefill"]))

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


def test_admin_timeout_can_be_overridden_by_env(monkeypatch):
    """A loaded pool can take longer than the default to drain on /pause: PRIME_ADMIN_TIMEOUT_S raises the bound."""
    import importlib

    import prime_rl.orchestrator.clients as clients

    try:
        monkeypatch.setenv("PRIME_ADMIN_TIMEOUT_S", "1800")
        assert importlib.reload(clients).ADMIN_TIMEOUT_S == 1800.0
        monkeypatch.delenv("PRIME_ADMIN_TIMEOUT_S")
        assert importlib.reload(clients).ADMIN_TIMEOUT_S == 300.0
    finally:
        monkeypatch.delenv("PRIME_ADMIN_TIMEOUT_S", raising=False)
        importlib.reload(clients)


def _dead_connection_client(calls):
    """A pooled admin client whose connection died: every request on it times out."""

    async def post(path, **kwargs):
        calls.append(("pooled", path, kwargs["timeout"].read))
        raise httpx.ReadTimeout("no response")

    return MagicMock(
        base_url=httpx.URL("http://engine:8206"), headers=httpx.Headers({ADMIN_TOKEN_HEADER: "secret"}), post=post
    )


def test_pause_retries_on_a_new_connection(monkeypatch):
    calls, created = [], []

    class FreshClient:
        def __init__(self, base_url, headers):
            created.append((str(base_url), headers.get(ADMIN_TOKEN_HEADER)))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, path, **kwargs):
            calls.append(("fresh", path, kwargs["timeout"].read))
            return MagicMock()

    monkeypatch.setattr(clients_module, "AsyncClient", FreshClient)
    asyncio.run(_pause_engines([_dead_connection_client(calls)], step=1, roles=["decode"]))

    assert calls == [("pooled", "/pause", PAUSE_ATTEMPT_TIMEOUT_S), ("fresh", "/pause", PAUSE_ATTEMPT_TIMEOUT_S)]
    assert created == [("http://engine:8206", "secret")]


def test_admin_post_retries_on_the_pooled_client_without_reconnect(monkeypatch):
    calls = []
    client = _dead_connection_client(calls)
    attempts = iter([httpx.ReadTimeout("no response"), None])

    async def post(path, **kwargs):
        calls.append(("pooled", path, kwargs["timeout"].read))
        if (error := next(attempts)) is not None:
            raise error
        return MagicMock()

    client.post = post
    monkeypatch.setattr(clients_module, "AsyncClient", MagicMock(side_effect=AssertionError("no new connection")))
    asyncio.run(_admin_post(client, "/finish_session", timeout_s=5.0))

    assert calls == [("pooled", "/finish_session", 5.0), ("pooled", "/finish_session", 5.0)]
