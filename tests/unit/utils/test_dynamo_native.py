import json

import httpx
import pytest

from prime_rl.utils.dynamo import (
    REQUIRED_ROUTES,
    DynamoDiscoveryPending,
    DynamoInferencePool,
    DynamoVLLMWeightSyncClient,
    DynamoWorker,
    parse_dynamo_workers,
)


def worker(component: str, instance_id: int, world_size: int = 1, *, backend: str | None = "nccl") -> dict:
    return {
        "namespace": "dynamo",
        "component": component,
        "instance_id": instance_id,
        "model": "Qwen/Qwen3-0.6B",
        "system_url": f"http://{component}-{instance_id}:8080",
        "admin_base_url": f"http://{component}-{instance_id}:8120",
        "world_size": world_size,
        "weight_transfer_backend": backend,
        "routes": sorted(REQUIRED_ROUTES),
    }


def test_parse_dynamo_workers_preserves_heterogeneous_topology():
    workers = parse_dynamo_workers(
        {
            "protocol_version": 1,
            "workers": [worker("prefill", 3, 2), worker("decode", 9, 4)],
        },
        "Qwen/Qwen3-0.6B",
    )
    assert [(item.component, item.world_size) for item in workers] == [("decode", 4), ("prefill", 2)]


def test_parse_dynamo_workers_waits_for_complete_capabilities():
    incomplete = worker("backend", 1)
    incomplete["routes"] = []
    with pytest.raises(DynamoDiscoveryPending, match="missing native routes"):
        parse_dynamo_workers(
            {"protocol_version": 1, "workers": [incomplete]},
            "Qwen/Qwen3-0.6B",
        )


def test_parse_dynamo_workers_accepts_custom_transfer_without_native_backend():
    workers = parse_dynamo_workers(
        {"protocol_version": 1, "workers": [worker("backend", 1, backend=None)]},
        "Qwen/Qwen3-0.6B",
    )

    assert workers[0].weight_transfer_backend is None
    assert workers[0].admin_base_url == "http://backend-1:8120"


@pytest.mark.asyncio
async def test_dynamo_nixl_initialization_uses_collective_rpc_and_cumulative_offsets():
    requests: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.host or "", request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"results": [None]})

    workers = (
        DynamoWorker.model_validate(worker("decode", 9, 4, backend=None)),
        DynamoWorker.model_validate(worker("prefill", 3, 2, backend=None)),
    )
    pool = object.__new__(DynamoInferencePool)
    pool.workers = workers
    pool._weight_update_timeout = 10
    pool._collective_rpc_clients = [
        httpx.AsyncClient(base_url=item.admin_base_url, transport=httpx.MockTransport(handler)) for item in workers
    ]
    try:
        await pool.init_nixl_broadcast(
            host="model-express",
            port=8001,
            timeout=30,
            inference_world_size=6,
            session_id="run-7",
        )
    finally:
        await _close(pool._collective_rpc_clients)

    assert [(host, body["args"][2]) for host, _, body in requests] == [("decode-9", 0), ("prefill-3", 4)]
    assert {path for _, path, _ in requests} == {"/collective_rpc"}
    assert all(body["method"] == "init_broadcaster" for _, _, body in requests)
    assert all(body["args"][3:] == [6, 30, "run-7"] for _, _, body in requests)


def test_weight_sync_client_assigns_cumulative_rank_offsets(monkeypatch):
    requests: list[tuple[str, str, dict]] = []
    real_client = httpx.Client

    def client_factory(*, base_url, **kwargs):
        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.url.host or "", request.url.path, json.loads(request.content)))
            return httpx.Response(200, json={"status": "ok"})

        return real_client(base_url=base_url, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("prime_rl.utils.dynamo.httpx.Client", client_factory)
    workers = (
        DynamoWorker.model_validate(worker("decode", 9, 4)),
        DynamoWorker.model_validate(worker("prefill", 3, 2)),
    )
    client = DynamoVLLMWeightSyncClient(workers, {}, timeout=10)
    client.init_weight_transfer_engine({"master_address": "trainer", "world_size": 7})

    assert [(host, body["init_info"]["rank_offset"]) for host, _, body in requests] == [
        ("decode-9", 1),
        ("prefill-3", 5),
    ]
    assert {path for _, path, _ in requests} == {"/engine/update/init_weight_transfer_engine"}


@pytest.mark.asyncio
async def test_dynamo_update_drives_native_lifecycle_and_commits_version(tmp_path):
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content or b"{}")
        requests.append((path, body))
        if path.endswith("is_paused"):
            return httpx.Response(200, json={"is_paused": True})
        if path.endswith("get_weight_version"):
            return httpx.Response(200, json={"weight_version": "3"})
        return httpx.Response(200, json={"status": "ok"})

    pool = object.__new__(DynamoInferencePool)
    pool.workers = (DynamoWorker.model_validate(worker("backend", 1)),)
    pool._weight_update_timeout = 10
    pool._admin_clients = [httpx.AsyncClient(base_url="http://worker:8080", transport=httpx.MockTransport(handler))]
    try:
        await pool.update_weights(tmp_path, step=3, native_nccl=True)
    finally:
        await _close(pool._admin_clients)

    assert (tmp_path / "NCCL_READY").exists()
    assert requests[0] == ("/engine/control/pause_generation", {"mode": "keep", "clear_cache": False})
    assert [path for path, _ in requests] == [
        "/engine/control/pause_generation",
        "/engine/control/is_paused",
        "/engine/control/get_weight_version",
        "/engine/control/resume_generation",
    ]


@pytest.mark.asyncio
async def test_dynamo_update_fails_closed_when_pause_is_not_confirmed(tmp_path):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("is_paused"):
            return httpx.Response(200, json={"is_paused": False})
        return httpx.Response(200, json={"status": "ok"})

    pool = object.__new__(DynamoInferencePool)
    pool.workers = (DynamoWorker.model_validate(worker("backend", 1)),)
    pool._weight_update_timeout = 10
    pool._admin_clients = [httpx.AsyncClient(base_url="http://worker:8080", transport=httpx.MockTransport(handler))]
    try:
        with pytest.raises(RuntimeError, match="every pinned worker was paused"):
            await pool.update_weights(tmp_path, step=3, native_nccl=True)
    finally:
        await _close(pool._admin_clients)

    assert "/engine/control/resume_generation" not in paths


@pytest.mark.asyncio
async def test_dynamo_nixl_update_uses_collective_rpc_and_commits_version():
    sidecar_requests: list[tuple[str, dict]] = []
    collective_requests: list[dict] = []

    def sidecar_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content or b"{}")
        sidecar_requests.append((path, body))
        if path.endswith("is_paused"):
            return httpx.Response(200, json={"is_paused": True})
        if path.endswith("get_weight_version"):
            return httpx.Response(200, json={"weight_version": "3"})
        return httpx.Response(200, json={"status": "ok"})

    def collective_handler(request: httpx.Request) -> httpx.Response:
        collective_requests.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [None]})

    pool = object.__new__(DynamoInferencePool)
    pool.workers = (DynamoWorker.model_validate(worker("backend", 1, backend=None)),)
    pool._weight_update_timeout = 10
    pool._weight_update_backend = "nixl"
    pool._admin_clients = [
        httpx.AsyncClient(base_url="http://worker:8080", transport=httpx.MockTransport(sidecar_handler))
    ]
    pool._collective_rpc_clients = [
        httpx.AsyncClient(base_url="http://worker:8120", transport=httpx.MockTransport(collective_handler))
    ]
    try:
        await pool.update_weights(None, step=3, native_nccl=False)
    finally:
        await _close([*pool._admin_clients, *pool._collective_rpc_clients])

    assert collective_requests == [{"method": "update_weights_from_path", "timeout": 10, "args": [None], "kwargs": {}}]
    assert [path for path, _ in sidecar_requests] == [
        "/engine/control/pause_generation",
        "/engine/control/is_paused",
        "/engine/update/update_weight_version",
        "/engine/control/get_weight_version",
        "/engine/control/resume_generation",
    ]
    assert sidecar_requests[2][1] == {"new_version": "3"}


@pytest.mark.asyncio
async def test_dynamo_filesystem_update_uses_native_reload_and_commits_version(tmp_path):
    sidecar_requests: list[tuple[str, dict]] = []
    collective_requests: list[dict] = []

    def sidecar_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content or b"{}")
        sidecar_requests.append((path, body))
        if path.endswith("is_paused"):
            return httpx.Response(200, json={"is_paused": True})
        if path.endswith("get_weight_version"):
            return httpx.Response(200, json={"weight_version": "4"})
        return httpx.Response(200, json={"status": "ok"})

    def collective_handler(request: httpx.Request) -> httpx.Response:
        collective_requests.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [None]})

    pool = object.__new__(DynamoInferencePool)
    pool.workers = (DynamoWorker.model_validate(worker("backend", 1, backend=None)),)
    pool._weight_update_timeout = 10
    pool._weight_update_backend = None
    pool._admin_clients = [
        httpx.AsyncClient(base_url="http://worker:8080", transport=httpx.MockTransport(sidecar_handler))
    ]
    pool._collective_rpc_clients = [
        httpx.AsyncClient(base_url="http://worker:8120", transport=httpx.MockTransport(collective_handler))
    ]
    try:
        await pool.update_weights(tmp_path, step=4, native_nccl=False)
    finally:
        await _close([*pool._admin_clients, *pool._collective_rpc_clients])

    assert collective_requests == [
        {
            "method": "reload_weights",
            "timeout": 10,
            "args": [],
            "kwargs": {"weights_path": tmp_path.as_posix()},
        }
    ]
    assert [path for path, _ in sidecar_requests] == [
        "/engine/control/pause_generation",
        "/engine/control/is_paused",
        "/engine/update/update_weight_version",
        "/engine/control/get_weight_version",
        "/engine/control/resume_generation",
    ]
    assert sidecar_requests[2][1] == {"new_version": "4"}


async def _close(clients: list[httpx.AsyncClient]) -> None:
    for client in clients:
        await client.aclose()
