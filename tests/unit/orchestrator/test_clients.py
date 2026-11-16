import asyncio
import contextlib
import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import httpx
import pytest
from verifiers.v1.configs.client import EvalClientConfig

from prime_rl.configs.shared import ClientConfig, NCCLWeightBroadcastConfig
from prime_rl.orchestrator.clients import (
    AdminPlane,
    _is_retryable_lora_error,
    _pause_engines,
    check_health,
    load_lora_adapter,
    setup_admin_clients,
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
        timeout=httpx.Timeout(connect=10.0, read=1200, write=60.0, pool=10.0),
        json={
            "host": "trainer",
            "port": 29501,
            "rank_offset": 0,
            "inference_world_size": 1,
            "timeout": 1200,
        },
    )
    asyncio.run(admin_plane.aclose())


def fake_engine(outcomes: dict[str, list[int | Exception]] | None = None):
    """An admin client that answers a POST with the next outcome listed for its path, a status to return or an
    error to raise, and with 200 after those."""
    outcomes = {path: list(listed) for path, listed in (outcomes or {}).items()}
    client = create_autospec(httpx.AsyncClient, instance=True)

    def post(path, **kwargs):
        listed = outcomes.get(path)
        outcome = listed.pop(0) if listed else 200
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, request=httpx.Request("POST", f"http://engine{path}"))

    client.post.side_effect = post
    return client


def posted_paths(client) -> list[str]:
    return [call.args[0] for call in client.post.await_args_list]


@pytest.mark.parametrize(
    "status, expectation",
    [(404, contextlib.nullcontext()), (500, pytest.raises(httpx.HTTPStatusError))],
)
def test_admin_plane_only_tolerates_a_missing_init_broadcaster_route(status, expectation):
    admin_plane = AdminPlane(ClientConfig())
    client = fake_engine({"/init_broadcaster": [status]})
    admin_plane.clients = [client]

    with expectation:
        asyncio.run(admin_plane.initialize_nccl(host="trainer", port=29501, timeout=1200, inference_world_size=1))

    assert posted_paths(client) == ["/init_broadcaster"]
    asyncio.run(admin_plane.aclose())


@pytest.mark.parametrize(
    "failure, names_the_timeout",
    [(500, False), (httpx.ReadError("connection reset"), False), (httpx.ReadTimeout("timed out"), True)],
    ids=["500", "read-error", "read-timeout"],
)
def test_failed_nccl_update_is_sent_once_and_leaves_the_engines_paused(failure, names_the_timeout):
    admin_plane = AdminPlane(ClientConfig())
    failed, healthy = fake_engine({"/update_weights": [failure]}), fake_engine()
    admin_plane.clients = [failed, healthy]

    with pytest.raises(RuntimeError, match="restart the inference servers") as error:
        asyncio.run(admin_plane.update_weights(Path("/broadcasts/step_3"), transport="nccl", step=3))

    assert ("update_weights_timeout (720 s)" in str(error.value)) is names_the_timeout
    assert posted_paths(failed) == ["/pause", "/update_weights"]
    assert posted_paths(healthy) == ["/pause", "/update_weights"]
    asyncio.run(admin_plane.aclose())


@pytest.mark.parametrize(
    "transport, failure",
    [("filesystem", 500), ("nccl", httpx.ConnectError("connection refused"))],
    ids=["filesystem-500", "nccl-request-never-sent"],
)
def test_update_is_retried_and_resumes_the_engines(transport, failure):
    admin_plane = AdminPlane(ClientConfig())
    client = fake_engine({"/update_weights": [failure]})
    admin_plane.clients = [client]

    with patch("prime_rl.orchestrator.clients.asyncio.sleep", new=AsyncMock()):
        asyncio.run(admin_plane.update_weights(Path("/broadcasts/step_3"), transport=transport, step=3))

    assert posted_paths(client) == ["/pause", "/update_weights", "/update_weights", "/resume"]
    asyncio.run(admin_plane.aclose())


def test_admin_clients_expire_idle_connections_before_the_vllm_server_closes_them():
    from vllm import envs

    limits = inspect.signature(setup_admin_clients).parameters["limits"].default
    assert limits.keepalive_expiry < envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE


def test_weight_broadcast_config_sets_the_update_weights_timeout(tmp_path):
    from prime_rl.transports.weights import setup_weight_receiver

    assert NCCLWeightBroadcastConfig().update_weights_timeout == 720
    admin_plane = AdminPlane(ClientConfig())
    client = fake_engine()
    admin_plane.clients = [client]
    receiver = setup_weight_receiver(
        tmp_path, NCCLWeightBroadcastConfig(update_weights_timeout=1800), admin_plane, "model"
    )
    receiver.step_dir(3).mkdir()

    asyncio.run(receiver.receive(3))

    assert posted_paths(client) == ["/pause", "/update_weights", "/resume"]
    assert client.post.await_args_list[1].kwargs["timeout"].read == 1800
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
