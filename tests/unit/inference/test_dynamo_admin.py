import json
from pathlib import Path

import httpx
import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.inference.dynamo_admin import (
    DynamoAdminAPI,
    DynamoTopology,
    DynamoWorker,
    discover_workers,
    discovery_urls,
    validate_worker_membership,
)

ROUTES = frozenset(
    {
        "init_weights_update_group",
        "pause_generation",
        "resume_generation",
        "update_weights_from_disk",
        "update_weights_from_distributed",
    }
)


def worker(
    instance_id: int,
    *,
    component: str = "backend",
    system_url: str | None = None,
    model: str = "test-model",
    routes: frozenset[str] = ROUTES,
) -> dict:
    return {
        "component": component,
        "instance_id": instance_id,
        "system_url": system_url or f"http://worker-{instance_id}:8081",
        "model": model,
        "routes": sorted(routes),
    }


def async_client(handler, base_url: str = "http://worker:8081") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url)


def test_discovery_url_defaults_to_rl_listener_port(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DYN_RL_DISCOVERY_URL", raising=False)
    monkeypatch.delenv("DYN_RL_PORT", raising=False)
    config = ClientConfig(base_url=["http://frontend.example:8000/v1"])
    assert discovery_urls(config) == ["http://frontend.example:8001"]


@pytest.mark.asyncio
async def test_worker_discovery_validates_and_sorts_system_urls():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "namespace": "test",
                "workers": [
                    worker(2, system_url="http://worker-b:8081"),
                    worker(1, system_url="http://worker-a:8081"),
                ],
            },
        )

    client = async_client(handler, "http://frontend:8001")
    try:
        discovered = await discover_workers(
            [client],
            timeout=1,
            model_name="test-model",
            topology=DynamoTopology(roles=("agg", "agg"), gpus_per_worker=2),
        )
    finally:
        await client.aclose()

    assert [item.system_url for item in discovered] == ["http://worker-a:8081", "http://worker-b:8081"]
    assert discovered[0] == DynamoWorker(
        instance_id=1,
        component="backend",
        role="agg",
        system_url="http://worker-a:8081",
        model="test-model",
        routes=ROUTES,
    )


@pytest.mark.asyncio
async def test_worker_discovery_waits_for_exact_staggered_topology(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        workers = [worker(1, component="prefill")]
        if calls > 1:
            # Dynamo advertises decode as `backend`; the expected topology
            # disambiguates it from an aggregated `backend` worker.
            workers.append(worker(2, component="backend"))
        return httpx.Response(200, json={"namespace": "test", "workers": workers})

    monkeypatch.setattr("prime_rl.inference.dynamo_admin.DISCOVERY_POLL_INTERVAL_S", 0)
    client = async_client(handler, "http://frontend:8001")
    try:
        discovered = await discover_workers(
            [client],
            timeout=1,
            model_name="test-model",
            topology=DynamoTopology(roles=("prefill", "decode"), gpus_per_worker=1),
        )
    finally:
        await client.aclose()

    assert calls == 2
    assert {item.role for item in discovered} == {"prefill", "decode"}


@pytest.mark.asyncio
async def test_worker_discovery_deduplicates_consistent_frontend_snapshots():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"namespace": "test", "workers": [worker(7)]})

    clients = [async_client(handler, f"http://frontend-{index}:8001") for index in range(2)]
    try:
        discovered = await discover_workers(
            clients,
            timeout=1,
            model_name="test-model",
            topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
        )
    finally:
        for client in clients:
            await client.aclose()

    assert len(discovered) == 1
    assert discovered[0].instance_id == 7


@pytest.mark.asyncio
async def test_worker_discovery_rejects_inconsistent_frontend_snapshots():
    def first(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"namespace": "test", "workers": [worker(7)]})

    def restarted(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"namespace": "test", "workers": [worker(8)]})

    clients = [async_client(first, "http://frontend-a:8001"), async_client(restarted, "http://frontend-b:8001")]
    try:
        with pytest.raises(TimeoutError, match="inconsistent worker snapshots"):
            await discover_workers(
                clients,
                timeout=0.01,
                model_name="test-model",
                topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
            )
    finally:
        for client in clients:
            await client.aclose()


@pytest.mark.asyncio
async def test_worker_discovery_bounds_each_get_by_remaining_deadline():
    request_timeouts: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json={"namespace": "test", "workers": [worker(1)]})

    client = async_client(handler, "http://frontend:8001")
    try:
        await discover_workers(
            [client],
            timeout=0.5,
            model_name="test-model",
            topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
        )
    finally:
        await client.aclose()

    assert len(request_timeouts) == 1
    assert 0 < request_timeouts[0]["read"] <= 0.5


@pytest.mark.asyncio
async def test_worker_discovery_rejects_incomplete_admin_surface():
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"namespace": "test", "workers": [worker(1, routes=frozenset())]})

    client = async_client(handler, "http://frontend:8001")
    try:
        with pytest.raises(ValueError, match="missing RL routes"):
            await discover_workers(
                [client],
                timeout=1,
                model_name="test-model",
                topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
            )
    finally:
        await client.aclose()

    assert calls == 1


@pytest.mark.asyncio
async def test_worker_discovery_rejects_model_mismatch_without_retry(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"namespace": "test", "workers": [worker(1, model="other-model")]})

    monkeypatch.setattr("prime_rl.inference.dynamo_admin.DISCOVERY_POLL_INTERVAL_S", 0)
    client = async_client(handler, "http://frontend:8001")
    try:
        with pytest.raises(ValueError, match="serves 'other-model', expected 'test-model'"):
            await discover_workers(
                [client],
                timeout=1,
                model_name="test-model",
                topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
            )
    finally:
        await client.aclose()

    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
async def test_worker_discovery_rejects_permanent_http_errors_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code)

    monkeypatch.setattr("prime_rl.inference.dynamo_admin.DISCOVERY_POLL_INTERVAL_S", 0)
    client = async_client(handler, "http://frontend:8001")
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await discover_workers(
                [client],
                timeout=1,
                model_name="test-model",
                topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
            )
    finally:
        await client.aclose()

    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 409, 429, 500])
async def test_worker_discovery_retries_transient_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status_code)
        return httpx.Response(200, json={"namespace": "test", "workers": [worker(1)]})

    monkeypatch.setattr("prime_rl.inference.dynamo_admin.DISCOVERY_POLL_INTERVAL_S", 0)
    client = async_client(handler, "http://frontend:8001")
    try:
        discovered = await discover_workers(
            [client],
            timeout=1,
            model_name="test-model",
            topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
        )
    finally:
        await client.aclose()

    assert calls == 2
    assert discovered[0].instance_id == 1


@pytest.mark.asyncio
async def test_worker_discovery_retries_transport_errors(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("frontend is starting", request=request)
        return httpx.Response(200, json={"namespace": "test", "workers": [worker(1)]})

    monkeypatch.setattr("prime_rl.inference.dynamo_admin.DISCOVERY_POLL_INTERVAL_S", 0)
    client = async_client(handler, "http://frontend:8001")
    try:
        discovered = await discover_workers(
            [client],
            timeout=1,
            model_name="test-model",
            topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
        )
    finally:
        await client.aclose()

    assert calls == 2
    assert discovered[0].instance_id == 1


@pytest.mark.asyncio
async def test_worker_discovery_retries_request_timeouts(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("frontend response timed out", request=request)
        return httpx.Response(200, json={"namespace": "test", "workers": [worker(1)]})

    monkeypatch.setattr("prime_rl.inference.dynamo_admin.DISCOVERY_POLL_INTERVAL_S", 0)
    client = async_client(handler, "http://frontend:8001")
    try:
        discovered = await discover_workers(
            [client],
            timeout=1,
            model_name="test-model",
            topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
        )
    finally:
        await client.aclose()

    assert calls == 2
    assert discovered[0].instance_id == 1


@pytest.mark.asyncio
async def test_worker_discovery_retries_transient_worker_probe_errors(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        discovered_worker = worker(1)
        if calls == 1:
            discovered_worker["error"] = "worker endpoint has not converged"
        return httpx.Response(200, json={"namespace": "test", "workers": [discovered_worker]})

    monkeypatch.setattr("prime_rl.inference.dynamo_admin.DISCOVERY_POLL_INTERVAL_S", 0)
    client = async_client(handler, "http://frontend:8001")
    try:
        discovered = await discover_workers(
            [client],
            timeout=1,
            model_name="test-model",
            topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
        )
    finally:
        await client.aclose()

    assert calls == 2
    assert discovered[0].instance_id == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_type", "match"),
    [
        (httpx.Response(200, content=b"{"), json.JSONDecodeError, "Expecting property name"),
        (httpx.Response(200, json={"namespace": "test"}), ValueError, "invalid response"),
        (
            httpx.Response(
                200,
                json={"namespace": "test", "workers": [{**worker(1), "error": 123}]},
            ),
            ValueError,
            "invalid error",
        ),
    ],
)
async def test_worker_discovery_rejects_invalid_payload_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    error_type: type[Exception],
    match: str,
):
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr("prime_rl.inference.dynamo_admin.DISCOVERY_POLL_INTERVAL_S", 0)
    client = async_client(handler, "http://frontend:8001")
    try:
        with pytest.raises(error_type, match=match):
            await discover_workers(
                [client],
                timeout=1,
                model_name="test-model",
                topology=DynamoTopology(roles=("agg",), gpus_per_worker=1),
            )
    finally:
        await client.aclose()

    assert calls == 1


def test_worker_membership_change_is_rejected():
    expected = (DynamoWorker(1, "backend", "agg", "http://worker:8081", "test-model", ROUTES),)
    restarted = [
        DynamoWorker(2, "backend", "agg", "http://worker:8081", "test-model", ROUTES),
    ]
    with pytest.raises(RuntimeError, match="membership changed"):
        validate_worker_membership(expected, restarted)


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
            gpus_per_worker=2,
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
    pause_bodies = [body for path, body in requests if path.endswith("/pause_generation")]
    assert pause_bodies == [{"mode": "wait", "clear_cache": False}] * 2


@pytest.mark.asyncio
async def test_nccl_initialization_does_not_replay_ambiguous_timeout():
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        raise httpx.ReadTimeout("response lost after collective may have started", request=request)

    client = async_client(handler)
    try:
        with pytest.raises(httpx.ReadTimeout):
            await DynamoAdminAPI().initialize_nccl(
                [client],
                host="localhost",
                port=29511,
                timeout=12000,
                inference_world_size=1,
                gpus_per_worker=1,
                quantize_in_weight_transfer=False,
            )
    finally:
        await client.aclose()

    assert paths == ["/engine/init_weights_update_group"]


@pytest.mark.asyncio
async def test_nccl_initialization_rejects_world_size_that_conflicts_with_topology():
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={"status": "ok"})

    client = async_client(handler)
    try:
        with pytest.raises(ValueError, match="does not match"):
            await DynamoAdminAPI().initialize_nccl(
                [client],
                host="localhost",
                port=29511,
                timeout=12000,
                inference_world_size=1,
                gpus_per_worker=2,
                quantize_in_weight_transfer=False,
            )
    finally:
        await client.aclose()

    assert requests == 0


@pytest.mark.asyncio
async def test_weight_collective_does_not_replay_ambiguous_timeout(tmp_path: Path):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("update_weights_from_disk"):
            raise httpx.ReadTimeout("response lost after update may have committed", request=request)
        return httpx.Response(200, json={"status": "ok"})

    client = async_client(handler)
    try:
        with pytest.raises(httpx.ReadTimeout):
            await DynamoAdminAPI().update_weights([client], tmp_path / "weights", step=1)
    finally:
        await client.aclose()

    assert paths == [
        "/engine/pause_generation",
        "/engine/update_weights_from_disk",
        "/engine/resume_generation",
    ]


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
