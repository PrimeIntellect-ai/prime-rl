import asyncio
import json
from pathlib import Path
from runpy import run_path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
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

    asyncio.run(load_lora_adapter(admin_plane, "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
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
            "quantize_in_weight_transfer": False,
        },
    )
    asyncio.run(admin_plane.aclose())


@pytest.mark.parametrize("failure_path", [None, "/pause", "/update_weights", "/resume"])
@pytest.mark.parametrize("timed", [False, True])
def test_mx_update_resumes_only_after_every_engine_succeeds(failure_path, timed):
    calls = []
    phase_timer = run_path(Path(__file__).parents[3] / "src/prime_rl/transports/weights/mx_phases.py")["PhaseTimer"]
    timer = phase_timer("orchestrator", 1, "test:1")

    async def handle(request):
        calls.append((request.url.host, request.url.path))
        if request.url.path == "/update_weights":
            assert json.loads(request.content) == {"weight_dir": None, "version_uid": "test:1"}
        status = (500 if failure_path == "/update_weights" else 400) if request.url.path == failure_path else 200
        return httpx.Response(status, json={"status": "ok"})

    async def run():
        admin = AdminPlane(ClientConfig())
        await admin.aclose()
        admin.clients = [
            httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url=f"http://worker-{rank}")
            for rank in range(2)
        ]
        try:
            with timer.phase("update_rpc", timeline=True):
                await admin.update_weights(
                    None, transport="mx_refit", step=1, version_uid="test:1", phase_timer=timer if timed else None
                )
        finally:
            await admin.aclose()

    if failure_path:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    paths = ["/pause", "/update_weights", "/resume"]
    reached = paths[: paths.index(failure_path) + 1] if failure_path else paths
    assert calls == [(f"worker-{rank}", path) for path in reached for rank in range(2)]
    if timed:
        spans = timer.payload()["spans"]
        assert [span["name"] for span in spans] == [
            {"/pause": "admin_pause", "/update_weights": "admin_update", "/resume": "admin_resume"}[path]
            for path in reached
        ] + ["update_rpc"]
        assert spans[-1]["status"] == ("failed" if failure_path else "complete")
        assert spans[-2]["status"] == ("failed" if failure_path else "complete")
        assert all(span["status"] == "complete" for span in spans[:-2])
        assert all(left["end_offset_s"] <= right["start_offset_s"] for left, right in zip(spans[:-2], spans[1:-1]))
        assert set(timer.phases) == {"update_rpc"}
        assert timer.marks["admin_initial_verification_enabled"] == 0


def test_mx_update_waits_for_every_replica_before_propagating():
    """A failing replica must not release the caller while siblings still read.

    asyncio.gather raises the first exception without cancelling its siblings.
    Those siblings are still pulling the trainer's registered buffers over RDMA,
    so returning early lets the caller retire the version, satisfy the release
    wait and publish the next step over memory that is still being read.
    """
    events = []

    async def handle(request):
        if request.url.path != "/update_weights":
            return httpx.Response(200, json={"status": "ok"})
        worker = request.url.host
        events.append((worker, "start"))
        if worker == "worker-0":
            return httpx.Response(500, json={"status": "error"})
        await asyncio.sleep(0.05)
        events.append((worker, "finish"))
        return httpx.Response(200, json={"status": "ok"})

    async def run():
        admin = AdminPlane(ClientConfig())
        await admin.aclose()
        admin.clients = [
            httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url=f"http://worker-{rank}")
            for rank in range(2)
        ]
        try:
            await admin.update_weights(None, transport="mx_refit", step=1, version_uid="test:1")
        finally:
            await admin.aclose()

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())
    assert ("worker-1", "finish") in events, "returned while a replica was still updating"


@pytest.mark.parametrize("changed", [False, True])
def test_mx_initial_verification_spans_preserve_recovery_pause(changed, monkeypatch):
    monkeypatch.setenv("MX_VERIFY_INITIAL_REFIT", "1")
    phase_timer = run_path(Path(__file__).parents[3] / "src/prime_rl/transports/weights/mx_phases.py")["PhaseTimer"]
    timer = phase_timer("orchestrator", 0, "test:0")
    calls = []

    async def handle(request):
        calls.append(request.url.path)
        result = {"status": "ok"}
        if request.url.path == "/mx_generation_control":
            result = {"replica": 0, "tokens": [int(changed and calls.count(request.url.path) > 1)]}
        return httpx.Response(200, json=result)

    async def run():
        admin = AdminPlane(ClientConfig())
        await admin.aclose()
        admin.clients = [httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://worker")]
        try:
            with timer.phase("update_rpc", timeline=True):
                await admin.update_weights(None, transport="mx_refit", step=0, version_uid="test:0", phase_timer=timer)
        finally:
            await admin.aclose()

    if changed:
        with pytest.raises(RuntimeError, match="Initial greedy generation changed"):
            asyncio.run(run())
    else:
        asyncio.run(run())
    expected_paths = [
        "/mx_generation_control",
        "/pause",
        "/mx_prepare_initial_refit",
        "/update_weights",
        "/resume",
        "/mx_generation_control",
    ]
    expected_spans = [
        "admin_initial_generation_before",
        "admin_pause",
        "admin_initial_prepare",
        "admin_update",
        "admin_resume",
        "admin_initial_generation_after",
    ]
    if changed:
        expected_paths.append("/pause")
        expected_spans.append("admin_initial_recovery_pause")
    assert calls == expected_paths
    assert [span["name"] for span in timer.spans] == expected_spans + ["update_rpc"]
    assert timer.spans[-1]["status"] == ("failed" if changed else "complete")
    assert timer.marks["admin_initial_verification_enabled"] == 1


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
