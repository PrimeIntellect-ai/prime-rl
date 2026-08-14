import json

import httpx
import pytest

from prime_rl.utils.dynamo import (
    REQUIRED_ROUTES,
    DynamoDiscoveryPending,
    DynamoInferencePool,
    DynamoWorker,
    parse_dynamo_workers,
)


def worker(component: str, instance_id: int, world_size: int = 1) -> dict:
    return {
        "namespace": "dynamo",
        "component": component,
        "instance_id": instance_id,
        "model": "Qwen/Qwen3-0.6B",
        "system_url": f"http://{component}-{instance_id}:8080",
        "world_size": world_size,
        "weight_transfer_backend": "nccl",
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


@pytest.mark.asyncio
async def test_dynamo_init_assigns_cumulative_rank_offsets():
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.host or "", json.loads(request.content)))
        return httpx.Response(200, json={"status": "ok"})

    pool = object.__new__(DynamoInferencePool)
    pool.workers = (
        DynamoWorker.model_validate(worker("decode", 9, 4)),
        DynamoWorker.model_validate(worker("prefill", 3, 2)),
    )
    pool._weight_update_timeout = 10
    pool._admin_clients = [
        httpx.AsyncClient(base_url=item.system_url, transport=httpx.MockTransport(handler)) for item in pool.workers
    ]
    try:
        await pool.init_nccl_broadcast(host="trainer", port=29501, timeout=10, inference_world_size=6)
    finally:
        await _close(pool._admin_clients)

    assert [(host, body["init_info"]["rank_offset"]) for host, body in requests] == [
        ("decode-9", 1),
        ("prefill-3", 5),
    ]
    assert {body["init_info"]["world_size"] for _, body in requests} == {7}


@pytest.mark.asyncio
async def test_dynamo_update_drives_native_lifecycle_and_commits_version(tmp_path):
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content or b"{}")
        requests.append((path, body))
        if path.endswith("is_paused"):
            return httpx.Response(200, json={"is_paused": True})
        if path.endswith("start_weight_update"):
            (tmp_path / "NCCL_MANIFEST.json").write_text(json.dumps({"num_chunks": 1}))
            (tmp_path / "NCCL_CHUNK_0.json").write_text(
                json.dumps({"names": ["model.weight"], "dtype_names": ["bfloat16"], "shapes": [[2, 2]]})
            )
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
        "/engine/update/start_weight_update",
        "/engine/update/update_weights",
        "/engine/update/finish_weight_update",
        "/engine/control/get_weight_version",
        "/engine/control/resume_generation",
    ]
    assert requests[4][1] == {"weight_version": "3"}


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


async def _close(clients: list[httpx.AsyncClient]) -> None:
    for client in clients:
        await client.aclose()
