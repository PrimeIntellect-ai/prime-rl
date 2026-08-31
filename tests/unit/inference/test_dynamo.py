import base64
import importlib
import io
import json
import sys
from types import ModuleType

import httpx
import numpy as np
import pytest

from prime_rl.inference.dynamo import (
    REQUIRED_ROUTES,
    DynamoAdminClients,
    DynamoDiscoveryPending,
    parse_dynamo_workers,
)
from prime_rl.inference.vllm.routed_experts import (
    install_native_routed_experts_normalizer,
    normalize_native_routed_experts,
)


def _worker(component: str, instance_id: int, world_size: int = 1) -> dict:
    return {
        "namespace": "dynamo",
        "component": component,
        "endpoint": "rl",
        "instance_id": instance_id,
        "model": "Qwen/Qwen3-0.6B",
        "request_plane_url": f"dyn://dynamo.{component}.rl",
        "system_url": f"http://{component}-{instance_id}:8080",
        "admin_base_url": f"http://{component}-{instance_id}:8120",
        "world_size": world_size,
        "routes": sorted(REQUIRED_ROUTES),
    }


def test_parse_dynamo_workers_preserves_topology_without_backend_metadata():
    workers = parse_dynamo_workers(
        {
            "protocol_version": 1,
            "workers": [_worker("prefill", 3, 2), _worker("decode", 9, 4)],
        },
        "Qwen/Qwen3-0.6B",
    )

    assert [(item.component, item.world_size) for item in workers] == [("decode", 4), ("prefill", 2)]
    assert not hasattr(workers[0], "weight_transfer_backend")


def test_parse_dynamo_workers_waits_for_required_routes():
    incomplete = _worker("backend", 1)
    incomplete["routes"] = []

    with pytest.raises(DynamoDiscoveryPending, match="missing required routes"):
        parse_dynamo_workers(
            {"protocol_version": 1, "workers": [incomplete]},
            "Qwen/Qwen3-0.6B",
        )


def test_parse_dynamo_workers_requires_world_size_only_for_in_memory_transfer():
    worker = _worker("backend", 1)
    worker.pop("world_size")

    parsed = parse_dynamo_workers(
        {"protocol_version": 1, "workers": [worker]},
        "Qwen/Qwen3-0.6B",
        require_world_size=False,
    )
    assert parsed[0].world_size is None

    with pytest.raises(DynamoDiscoveryPending, match="world_size"):
        parse_dynamo_workers(
            {"protocol_version": 1, "workers": [worker]},
            "Qwen/Qwen3-0.6B",
            require_world_size=True,
        )


def test_native_npy_routed_experts_are_normalized_at_prime_boundary():
    routed = np.arange(12, dtype=np.int32).reshape(3, 2, 2)
    encoded = io.BytesIO()
    np.save(encoded, routed, allow_pickle=False)
    result = {"routed_experts": base64.b64encode(encoded.getvalue()).decode("ascii")}

    normalize_native_routed_experts(result, start=7)

    payload = result["routed_experts"]
    assert payload["shape"] == [3, 2, 2]
    assert payload["dtype"] == "int32"
    assert payload["start"] == 7
    assert base64.b64decode(payload["data"]) == routed.tobytes()


def test_prime_routed_experts_envelope_is_left_unchanged():
    payload = {"data": "AQID", "shape": [1, 1, 3], "dtype": "uint8", "start": 0}
    result = {"routed_experts": payload}

    normalize_native_routed_experts(result, start=9)

    assert result["routed_experts"] is payload


@pytest.mark.asyncio
async def test_dynamo_admin_uses_system_routes_for_pause_and_version():
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path, body))
        if request.url.path.endswith("is_paused"):
            return httpx.Response(200, json={"is_paused": True})
        if request.url.path.endswith("get_weight_version"):
            return httpx.Response(200, json={"weight_version": "3"})
        return httpx.Response(200, json={"status": "ok"})

    admin = object.__new__(DynamoAdminClients)
    admin.system_clients = [httpx.AsyncClient(base_url="http://worker:8080", transport=httpx.MockTransport(handler))]
    admin.timeout = 10
    try:
        await admin.pause()
        assert await admin.is_paused()
        await admin.update_weight_version("3")
        assert await admin.weight_versions() == ["3"]
        await admin.resume()
    finally:
        await admin.system_clients[0].aclose()

    assert [path for path, _ in requests] == [
        "/engine/control/pause_generation",
        "/engine/control/is_paused",
        "/engine/update/update_weight_version",
        "/engine/control/get_weight_version",
        "/engine/control/resume_generation",
    ]


@pytest.mark.asyncio
async def test_renderer_boundary_supplies_prime_prompt_start(monkeypatch):
    import renderers.client as renderer_client

    routed = np.arange(4, dtype=np.int16).reshape(1, 2, 2)
    encoded = io.BytesIO()
    np.save(encoded, routed, allow_pickle=False)

    async def generate(**kwargs):
        return {"routed_experts": base64.b64encode(encoded.getvalue()).decode("ascii")}

    monkeypatch.setattr(renderer_client, "generate", generate)
    install_native_routed_experts_normalizer()

    result = await renderer_client.generate(sampling_params={"routed_experts_prompt_start": 11})

    assert result["routed_experts"]["start"] == 11


def test_env_server_workers_install_native_routed_experts_normalizer(monkeypatch):
    utils_module = ModuleType("prime_rl.utils.utils")
    utils_module.clean_exit = lambda function: function
    monkeypatch.setitem(sys.modules, "prime_rl.utils.utils", utils_module)
    sys.modules.pop("prime_rl.entrypoints.env_server", None)
    env_server = importlib.import_module("prime_rl.entrypoints.env_server")

    installed: list[bool] = []
    monkeypatch.setattr(env_server, "setup_env_server_logging", lambda *_args: None)
    monkeypatch.setattr(env_server, "set_base_sandbox_labels", lambda _labels: None)
    monkeypatch.setattr(
        env_server,
        "install_native_routed_experts_normalizer",
        lambda: installed.append(True),
        raising=False,
    )

    env_server.setup_worker(None, False, [])

    assert installed == [True]
