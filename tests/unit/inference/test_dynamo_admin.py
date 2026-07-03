import json
from pathlib import Path

import httpx
import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.inference.dynamo_admin import (
    DynamoAdminAPI,
    discover_worker_urls,
    discovery_urls,
    validate_worker_membership,
)


def async_client(handler, base_url: str = "http://worker:8081") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url)


def test_discovery_url_defaults_to_rl_listener_port(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DYN_RL_DISCOVERY_URL", raising=False)
    monkeypatch.delenv("DYN_RL_PORT", raising=False)
    config = ClientConfig(base_url=["http://frontend.example:8000/v1"])
    assert discovery_urls(config) == ["http://frontend.example:8001"]


@pytest.mark.asyncio
async def test_worker_discovery_validates_and_sorts_system_urls():
    routes = [
        "init_weights_update_group",
        "pause_generation",
        "resume_generation",
        "update_weights_from_disk",
        "update_weights_from_distributed",
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "namespace": "test",
                "workers": [
                    {"system_url": "http://worker-b:8081", "routes": routes},
                    {"system_url": "http://worker-a:8081", "routes": routes},
                ],
            },
        )

    client = async_client(handler, "http://frontend:8001")
    try:
        assert await discover_worker_urls([client], timeout=1) == [
            "http://worker-a:8081",
            "http://worker-b:8081",
        ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_worker_discovery_rejects_incomplete_admin_surface():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"workers": [{"system_url": "http://worker:8081", "routes": []}]})

    client = async_client(handler, "http://frontend:8001")
    try:
        with pytest.raises(TimeoutError, match="missing RL routes"):
            await discover_worker_urls([client], timeout=1)
    finally:
        await client.aclose()


def test_worker_membership_change_is_rejected():
    with pytest.raises(RuntimeError, match="membership changed"):
        validate_worker_membership(("http://worker-a:8081",), ["http://worker-b:8081"])


@pytest.mark.asyncio
async def test_nccl_initialization_and_update_use_engine_routes(tmp_path: Path):
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"status": "ok"})

    clients = [async_client(handler, f"http://worker-{index}:8081") for index in range(2)]
    admin = DynamoAdminAPI()
    try:
        await admin.initialize_nccl(
            clients,
            host="localhost",
            port=29511,
            timeout=12000,
            inference_world_size=4,
            quantize_in_weight_transfer=False,
        )
        await admin.update_weights(clients, tmp_path / "step_1", step=1)
    finally:
        for client in clients:
            await client.aclose()

    init_bodies = [body for path, body in requests if path.endswith("/init_weights_update_group")]
    assert [body["rank_offset"] for body in init_bodies] == [0, 2]
    assert all(body["inference_world_size"] == 4 for body in init_bodies)

    updates = [body for path, body in requests if path.endswith("/update_weights_from_distributed")]
    assert len(updates) == 2
    assert all(body["engine_rpc"] == "update_weights_from_path" for body in updates)
    assert all(body["weight_version"] == "1" for body in updates)
    assert (tmp_path / "step_1" / "NCCL_READY").exists()

    paths = [path for path, _body in requests]
    assert paths.count("/engine/pause_generation") == 2
    assert paths.count("/engine/resume_generation") == 2


@pytest.mark.asyncio
async def test_engine_status_error_is_not_accepted():
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("resume_generation"):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"status": "error", "message": "not paused"})

    client = async_client(handler)
    try:
        with pytest.raises(RuntimeError, match="not paused"):
            await DynamoAdminAPI().update_weights([client], Path("weights"), step=1)
    finally:
        await client.aclose()
    assert paths == ["/engine/pause_generation", "/engine/resume_generation"]


@pytest.mark.asyncio
async def test_resume_error_does_not_hide_primary_update_error(monkeypatch: pytest.MonkeyPatch):
    admin = DynamoAdminAPI()

    async def post(_client, method, *_args, **_kwargs):
        if method == "pause_generation":
            raise RuntimeError("pause failed")
        if method == "resume_generation":
            raise RuntimeError("resume failed")
        raise AssertionError(method)

    monkeypatch.setattr(admin, "_post", post)

    with pytest.raises(RuntimeError, match="pause failed") as exc:
        await admin.update_weights([object()], Path("weights"), step=1)

    assert exc.value.__notes__ == ["Dynamo resume_generation cleanup also failed: RuntimeError('resume failed')"]
