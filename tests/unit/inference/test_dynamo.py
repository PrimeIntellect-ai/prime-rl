import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.inference.dynamo import (
    DynamoAdminPlane,
    DynamoDiscoveryPending,
    parse_dynamo_workers,
)
from prime_rl.orchestrator.clients import setup_admin_plane


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


def successful_response() -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    return response


def dynamo_admin(*worker_specs: dict) -> DynamoAdminPlane:
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://dynamo-frontend:8001"},
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)
    admin.workers = parse_dynamo_workers(snapshot(*worker_specs), "Qwen/Qwen3-0.6B")
    return admin


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


def test_admin_plane_factory_selects_dynamo():
    config = ClientConfig(dynamo={"discovery_url": "http://dynamo-frontend:8001"})

    admin = setup_admin_plane(config, "Qwen/Qwen3-0.6B")

    assert isinstance(admin, DynamoAdminPlane)
    asyncio.run(admin.aclose())


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


def test_parse_dynamo_workers_treats_incomplete_matching_worker_as_pending():
    incomplete_worker = worker(1, admin_base_url="http://worker-1:8120", world_size=1)
    incomplete_worker.pop("admin_base_url")

    with pytest.raises(DynamoDiscoveryPending, match="missing required RL metadata"):
        parse_dynamo_workers(snapshot(incomplete_worker), "Qwen/Qwen3-0.6B")


def test_dynamo_admin_plane_retries_invalid_startup_snapshot(monkeypatch):
    workers = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    discover = AsyncMock(side_effect=[ValueError("temporary invalid snapshot"), workers, workers])
    monkeypatch.setattr("prime_rl.inference.dynamo.discover_dynamo_workers", discover)
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://dynamo-frontend:8001"},
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))

    assert discover.await_count == 3
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_pins_two_identical_snapshots(monkeypatch):
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
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))

    assert discover.await_count == 4
    assert str(admin.clients[0].base_url) == "http://worker-1:8120"
    assert "authorization" not in admin.clients[0].headers
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_fails_closed_on_topology_drift(monkeypatch):
    pinned = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    changed = parse_dynamo_workers(
        snapshot(worker(2, admin_base_url="http://worker-2:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    discover = AsyncMock(side_effect=[pinned, pinned, changed, changed])
    monkeypatch.setattr("prime_rl.inference.dynamo.discover_dynamo_workers", discover)
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://dynamo-frontend:8001"},
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
        with pytest.raises(RuntimeError, match="topology changed"):
            asyncio.run(admin.ensure_topology_current())

    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_retries_transient_topology_probe(monkeypatch):
    pinned = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    discover = AsyncMock(side_effect=[pinned, pinned, httpx.ConnectError("temporary discovery failure"), pinned])
    monkeypatch.setattr("prime_rl.inference.dynamo.discover_dynamo_workers", discover)
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://dynamo-frontend:8001"},
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
        asyncio.run(admin.ensure_topology_current())

    assert discover.await_count == 4
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_retries_invalid_topology_snapshot(monkeypatch):
    pinned = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    discover = AsyncMock(side_effect=[pinned, pinned, ValueError("temporary invalid snapshot"), pinned])
    monkeypatch.setattr("prime_rl.inference.dynamo.discover_dynamo_workers", discover)
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://dynamo-frontend:8001"},
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
        asyncio.run(admin.ensure_topology_current())

    assert discover.await_count == 4
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_ignores_unconfirmed_topology_drift(monkeypatch):
    pinned = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    changed = parse_dynamo_workers(
        snapshot(worker(2, admin_base_url="http://worker-2:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    discover = AsyncMock(side_effect=[pinned, pinned, changed, pinned])
    monkeypatch.setattr("prime_rl.inference.dynamo.discover_dynamo_workers", discover)
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://dynamo-frontend:8001"},
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
        asyncio.run(admin.ensure_topology_current())

    assert discover.await_count == 4
    asyncio.run(admin.aclose())


def test_dynamo_nccl_initialization_assigns_discovered_rank_offsets():
    admin = dynamo_admin(
        worker(3, admin_base_url="http://worker-3:8120", world_size=2),
        worker(9, admin_base_url="http://worker-9:8120", world_size=1),
    )
    admin.clients = [AsyncMock(), AsyncMock()]
    for client in admin.clients:
        client.post.return_value = successful_response()

    with patch.object(admin, "ensure_topology_current", new=AsyncMock()):
        asyncio.run(
            admin.initialize_nccl(
                host="trainer",
                port=29501,
                timeout=1200,
                inference_world_size=3,
            )
        )

    assert admin.clients[0].post.await_args.kwargs["json"]["args"][2] == 0
    assert admin.clients[1].post.await_args.kwargs["json"]["args"][2] == 2
    asyncio.run(admin.aclose())


def test_dynamo_nccl_initialization_rejects_world_size_mismatch():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=2))

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        pytest.raises(ValueError, match="do not match"),
    ):
        asyncio.run(
            admin.initialize_nccl(
                host="trainer",
                port=29501,
                timeout=1200,
                inference_world_size=3,
            )
        )

    asyncio.run(admin.aclose())


def test_dynamo_nccl_weight_update_uses_collective_rpc(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    client = AsyncMock()
    client.post.return_value = successful_response()
    admin.clients = [client]
    step_dir = tmp_path / "step_1"

    with patch.object(admin, "ensure_topology_current", new=AsyncMock()):
        asyncio.run(admin.update_weights(step_dir, transport="nccl", step=1))

    assert [call.args[0] for call in client.post.await_args_list] == ["/pause", "/collective_rpc", "/resume"]
    assert client.post.await_args_list[1].kwargs["json"] == {
        "method": "update_weights_from_path",
        "timeout": 720.0,
        "args": [step_dir.as_posix()],
        "kwargs": {},
    }
    asyncio.run(admin.aclose())


@pytest.mark.parametrize(("transport", "uses_path"), [("filesystem", True), ("nixl", False)])
def test_dynamo_non_nccl_weight_updates_use_default_admin_route(tmp_path, transport, uses_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    client = AsyncMock()
    client.post.return_value = successful_response()
    admin.clients = [client]
    weight_dir = tmp_path / "step_1" if uses_path else None

    asyncio.run(admin.update_weights(weight_dir, transport=transport, step=1))

    assert [call.args[0] for call in client.post.await_args_list] == ["/pause", "/update_weights", "/resume"]
    assert client.post.await_args_list[1].kwargs["json"] == {
        "weight_dir": weight_dir.as_posix() if weight_dir is not None else None
    }
    asyncio.run(admin.aclose())


def test_dynamo_nccl_update_failure_keeps_engines_paused(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    client = AsyncMock()
    failed = successful_response()
    failed.raise_for_status.side_effect = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("POST", "http://worker/collective_rpc"),
        response=httpx.Response(400, request=httpx.Request("POST", "http://worker/collective_rpc")),
    )
    client.post.side_effect = [successful_response(), failed]
    admin.clients = [client]

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        pytest.raises(httpx.HTTPStatusError),
    ):
        asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

    assert [call.args[0] for call in client.post.await_args_list] == ["/pause", "/collective_rpc"]
    asyncio.run(admin.aclose())
