import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.inference.dynamo import (
    DynamoAdminClients,
    DynamoDiscoveryPending,
    parse_dynamo_workers,
)


def worker(instance_id: int, *, admin_base_url: str, world_size: int, model: str = "Qwen/Qwen3-0.6B") -> dict:
    return {
        "namespace": "dynamo",
        "component": "backend",
        "endpoint": "rl",
        "instance_id": instance_id,
        "transport": {"nats_tcp": f"nats://worker-{instance_id}:4222"},
        "request_plane_url": "dyn://dynamo.backend.rl",
        "system_url": f"http://worker-{instance_id}:8081",
        "admin_base_url": admin_base_url,
        "world_size": world_size,
        "model": model,
        "routes": [],
    }


def snapshot(*workers: dict) -> dict:
    return {"protocol_version": 1, "namespace": "dynamo", "workers": list(workers)}


def test_parse_dynamo_workers_validates_and_orders_topology():
    parsed = parse_dynamo_workers(
        snapshot(
            worker(9, admin_base_url="http://worker-9:8120", world_size=1),
            worker(3, admin_base_url="http://worker-3:8120", world_size=2),
        ),
        "Qwen/Qwen3-0.6B",
    )

    assert [item.instance_id for item in parsed] == [3, 9]
    assert [item.world_size for item in parsed] == [2, 1]
    assert parsed[0].transport == {"nats_tcp": "nats://worker-3:4222"}


@pytest.mark.parametrize(
    "payload",
    [
        {"protocol_version": 2, "namespace": "dynamo", "workers": []},
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=0)),
        snapshot(worker(1, admin_base_url="file:///tmp/admin", world_size=1)),
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=True)),
    ],
)
def test_parse_dynamo_workers_rejects_invalid_contract(payload):
    with pytest.raises((ValueError, DynamoDiscoveryPending)):
        parse_dynamo_workers(payload, "Qwen/Qwen3-0.6B")


def test_parse_dynamo_workers_rejects_duplicate_admin_urls():
    with pytest.raises(ValueError, match="duplicate worker admin endpoints"):
        parse_dynamo_workers(
            snapshot(
                worker(1, admin_base_url="http://worker:8120", world_size=1),
                worker(2, admin_base_url="http://worker:8120", world_size=1),
            ),
            "Qwen/Qwen3-0.6B",
        )


def test_dynamo_admin_clients_pin_two_identical_snapshots(monkeypatch):
    workers = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    changed = parse_dynamo_workers(
        snapshot(worker(2, admin_base_url="http://worker-2:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    discover = AsyncMock(side_effect=[workers, changed, workers, workers])
    monkeypatch.setattr("prime_rl.inference.dynamo.discover_dynamo_workers", discover)
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://dynamo-frontend:8001"},
    )
    admin = DynamoAdminClients(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))

    assert discover.await_count == 4
    assert admin.worker_world_sizes == (1,)
    assert admin.use_collective_rpc is True
    assert admin.worker_extension_cls == "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker"
    assert str(admin.clients[0].base_url) == "http://worker-1:8120"
    assert "authorization" not in admin.clients[0].headers
    asyncio.run(admin.aclose())


def test_dynamo_admin_clients_fail_closed_on_topology_drift(monkeypatch):
    pinned = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    changed = parse_dynamo_workers(
        snapshot(worker(2, admin_base_url="http://worker-2:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    discover = AsyncMock(side_effect=[pinned, pinned, changed])
    monkeypatch.setattr("prime_rl.inference.dynamo.discover_dynamo_workers", discover)
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://dynamo-frontend:8001"},
    )
    admin = DynamoAdminClients(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
        with pytest.raises(RuntimeError, match="topology changed"):
            asyncio.run(admin.ensure_topology_current())

    asyncio.run(admin.aclose())
