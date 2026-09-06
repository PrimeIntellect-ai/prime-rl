import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.inference.dynamo import (
    DynamoAdminPlane,
    DynamoDiscoveryPending,
    parse_dynamo_worker,
    topology_fingerprint,
)
from prime_rl.orchestrator.clients import AdminPlane, setup_admin_plane

MODEL = "Qwen/Qwen3-0.6B"


def dynamo_config() -> dict:
    return {"discovery_url": "http://worker:8001"}


def worker(
    instance_id: int,
    *,
    admin_base_url: str | None = None,
    model: str = MODEL,
) -> dict:
    return {
        "instance_id": instance_id,
        "admin_base_url": admin_base_url or f"http://worker:{8200 + instance_id}",
        "world_size": 1,
        "model": model,
    }


def snapshot(*workers: dict) -> dict:
    return {"protocol_version": 1, "workers": list(workers)}


def parsed(*workers: dict):
    return parse_dynamo_worker(snapshot(*workers), MODEL, expected_admin_host="worker")


def admin_for(*workers: dict) -> DynamoAdminPlane:
    config = ClientConfig(
        base_url="http://worker:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, MODEL, poll_interval=0)
    admin._fingerprint = topology_fingerprint(parsed(*workers))
    admin.clients = [AsyncMock() for _ in workers]
    return admin


def test_parse_dynamo_worker_filters_model_and_checks_admin_host():
    discovered_worker = parse_dynamo_worker(
        snapshot(
            worker(3),
            worker(1, model="other"),
        ),
        MODEL,
        expected_admin_host="worker",
    )

    assert discovered_worker.admin_base_url == "http://worker:8203"

    with pytest.raises(ValueError, match="does not match discovery host"):
        parse_dynamo_worker(
            snapshot(worker(1, admin_base_url="http://other-worker:8201")),
            MODEL,
            expected_admin_host="worker",
        )


def test_parse_dynamo_worker_rejects_multiple_matching_workers():
    with pytest.raises(ValueError, match="exactly one inference worker"):
        parse_dynamo_worker(snapshot(worker(1), worker(2)), MODEL, expected_admin_host="worker")


@pytest.mark.parametrize(
    ("worker_update", "error", "match"),
    [
        ({"admin_base_url": None}, DynamoDiscoveryPending, "admin_base_url"),
        ({"world_size": 2}, ValueError, "exactly one inference rank"),
    ],
)
def test_parse_dynamo_worker_validates_required_singleton_metadata(worker_update, error, match):
    payload = {**worker(1), **worker_update}
    with pytest.raises(error, match=match):
        parse_dynamo_worker(snapshot(payload), MODEL, expected_admin_host="worker")


def test_dynamo_admin_plane_factory_pins_two_identical_snapshots():
    discovered_worker = parsed(worker(1))
    discover = AsyncMock(side_effect=[discovered_worker, discovered_worker])
    admin = setup_admin_plane(
        ClientConfig(
            base_url="http://worker:8000/v1",
            skip_model_check=True,
            wait_for_ready_timeout=2,
            dynamo=dynamo_config(),
        ),
        MODEL,
    )
    assert isinstance(admin, DynamoAdminPlane)
    assert admin._discovery_url == "http://worker:8001"
    admin._poll_interval = 0

    with (
        patch.object(admin, "_discover", discover),
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready(MODEL))

    assert discover.await_count == 2
    assert str(admin.clients[0].base_url) == "http://worker:8201"
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_waits_for_frontend_model_after_worker_discovery():
    discovered_worker = parsed(worker(1))
    discover = AsyncMock(side_effect=[discovered_worker] * 4)
    model_check = AsyncMock(side_effect=[ValueError("model is still loading"), None])
    admin = setup_admin_plane(
        ClientConfig(
            base_url="http://worker:8000/v1",
            wait_for_ready_timeout=2,
            dynamo=dynamo_config(),
        ),
        MODEL,
    )
    assert isinstance(admin, DynamoAdminPlane)
    admin._poll_interval = 0

    with (
        patch.object(admin, "_discover", discover),
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=model_check),
    ):
        asyncio.run(admin.wait_for_ready(MODEL))

    assert discover.await_count == 4
    assert model_check.await_count == 2
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_derives_discovery_url_from_client_port():
    admin = setup_admin_plane(
        ClientConfig(
            base_url="http://worker:8000/v1",
            dynamo={"enabled": True},
        ),
        MODEL,
    )

    assert isinstance(admin, DynamoAdminPlane)
    assert admin._discovery_url == "http://worker:8001"
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_can_be_disabled():
    admin = setup_admin_plane(
        ClientConfig(
            base_url="http://worker:8000/v1",
            dynamo={"enabled": False},
        ),
        MODEL,
    )

    assert type(admin) is AdminPlane
    asyncio.run(admin.aclose())


def test_dynamo_discovery_url_derivation_requires_an_explicit_port():
    with pytest.raises(ValueError, match="Set dynamo.discovery_url"):
        setup_admin_plane(
            ClientConfig(
                base_url="http://worker/v1",
                dynamo={"enabled": True},
            ),
            MODEL,
        )


def test_dynamo_admin_plane_rejects_confirmed_topology_drift():
    admin_url = "http://worker:8200"
    changed = parsed(worker(2, admin_base_url=admin_url))
    admin = admin_for(worker(1, admin_base_url=admin_url))

    with (
        patch.object(admin, "_discover", new=AsyncMock(side_effect=[changed, changed])),
        pytest.raises(RuntimeError, match="topology changed"),
    ):
        asyncio.run(admin.ensure_topology_current())

    asyncio.run(admin.aclose())


def test_dynamo_nccl_lifecycle_initializes_and_updates_weights(tmp_path):
    admin = admin_for(worker(3))

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_collective_rpc", new=AsyncMock()) as collective_rpc,
        patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock()) as post,
    ):
        with pytest.raises(ValueError, match="exactly one inference rank"):
            asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=2))
        collective_rpc.assert_not_awaited()
        assert admin._nccl_initialization_state == "uninitialized"

        with pytest.raises(RuntimeError, match="ready NCCL initialization"):
            asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

        asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=1))
        asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

    assert collective_rpc.await_args_list[0].kwargs["args"] == ["trainer", 29501, 0, 1, 10, False, "default"]
    assert collective_rpc.await_args_list[1].kwargs["args"] == [(tmp_path / "step_1").as_posix()]
    assert [call.args[1] for call in post.await_args_list] == ["/pause", "/resume"]
    assert admin._nccl_initialization_state == "ready"
    asyncio.run(admin.aclose())


def test_dynamo_delegates_non_nccl_weight_updates(tmp_path):
    admin = admin_for(worker(1))

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()) as ensure_topology,
        patch.object(AdminPlane, "update_weights", new=AsyncMock()) as update_weights,
    ):
        asyncio.run(admin.update_weights(tmp_path, transport="filesystem", step=1))

    ensure_topology.assert_awaited_once_with()
    update_weights.assert_awaited_once_with(tmp_path, transport="filesystem", step=1, on_paused=None)
    asyncio.run(admin.aclose())


def test_dynamo_nccl_initialization_failure_is_terminal():
    admin = admin_for(worker(1))

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_collective_rpc", new=AsyncMock(side_effect=ValueError("unexpected"))),
        pytest.raises(RuntimeError, match="must restart"),
    ):
        asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=1))

    assert admin._nccl_initialization_state == "terminal"
    asyncio.run(admin.aclose())


def test_dynamo_nccl_update_failure_stays_paused_and_terminal(tmp_path):
    admin = admin_for(worker(1))
    admin._nccl_initialization_state = "ready"

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock()) as post,
        patch.object(admin, "_collective_rpc", new=AsyncMock(side_effect=RuntimeError("failed"))),
        pytest.raises(RuntimeError, match="engines remain paused"),
    ):
        asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

    assert [call.args[1] for call in post.await_args_list] == ["/pause"]
    assert admin._nccl_initialization_state == "terminal"
    asyncio.run(admin.aclose())


def test_parse_dynamo_python_worker_uses_system_admin_routes():
    discovered_worker = parse_dynamo_worker(
        {
            "namespace": "prime-rl-test",
            "workers": [
                {
                    "instance_id": 3,
                    "system_url": "http://worker:8081",
                    "routes": [
                        "pause_generation",
                        "resume_generation",
                        "init_weights_update_group",
                        "update_weights_from_distributed",
                    ],
                    "model": MODEL,
                }
            ],
        },
        MODEL,
        expected_admin_host="worker",
    )

    assert discovered_worker.admin_base_url == "http://worker:8081"
    assert discovered_worker.admin_protocol == "engine_routes"

    native_worker = parse_dynamo_worker(
        snapshot(worker(3, admin_base_url="http://worker:8081")),
        MODEL,
        expected_admin_host="worker",
    )

    assert topology_fingerprint(discovered_worker) != topology_fingerprint(native_worker)
    loopback_worker = parse_dynamo_worker(
        snapshot(
            {
                "instance_id": 4,
                "system_url": "http://127.0.0.1:8081",
                "routes": [
                    "pause_generation",
                    "resume_generation",
                    "init_weights_update_group",
                    "update_weights_from_distributed",
                ],
                "model": MODEL,
            }
        ),
        MODEL,
        expected_admin_host="localhost",
    )
    assert loopback_worker.admin_base_url == "http://127.0.0.1:8081"


def test_parse_dynamo_python_worker_requires_weight_update_routes():
    with pytest.raises(DynamoDiscoveryPending, match="required admin routes"):
        parse_dynamo_worker(
            snapshot(
                {
                    "instance_id": 3,
                    "system_url": "http://worker:8081",
                    "routes": ["pause_generation", "resume_generation"],
                    "model": MODEL,
                }
            ),
            MODEL,
            expected_admin_host="worker",
        )


def test_dynamo_python_worker_translates_collective_rpc_to_engine_route():
    config = ClientConfig(
        base_url="http://worker:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, MODEL, poll_interval=0)
    admin._admin_protocol = "engine_routes"
    response = MagicMock()

    response.json.return_value = {"status": "ok"}
    client = AsyncMock()
    client.post.return_value = response

    asyncio.run(
        admin._collective_rpc(
            client,
            method="init_broadcaster",
            timeout=10,
            args=["trainer", 29501, 0, 1, 10, False, "default"],
        )
    )

    client.post.assert_awaited_once()
    assert client.post.await_args.args[0] == "/engine/init_weights_update_group"
    assert client.post.await_args.kwargs["json"] == {
        "engine_rpc": "init_broadcaster",
        "host": "trainer",
        "port": 29501,
        "rank_offset": 0,
        "inference_world_size": 1,
        "timeout": 10,
        "quantize_in_weight_transfer": False,
        "session_id": "default",
    }


def test_dynamo_nixl_lifecycle_uses_collective_rpc():
    admin = admin_for(worker(1))

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_collective_rpc", new=AsyncMock()) as collective_rpc,
        patch.object(admin, "_set_generation_paused", new=AsyncMock()) as set_paused,
    ):
        asyncio.run(
            admin.initialize_nixl(
                host="trainer",
                port=8001,
                timeout=10,
                inference_world_size=1,
                session_id="test-session",
            )
        )
        asyncio.run(admin.update_weights(None, transport="nixl", step=1))

    assert [call.kwargs["method"] for call in collective_rpc.await_args_list] == [
        "init_broadcaster",
        "update_weights_from_path",
    ]
    assert collective_rpc.await_args_list[0].kwargs["args"][-1] == "test-session"
    assert collective_rpc.await_args_list[1].kwargs["args"] == [None]
    assert [call.args[0] for call in set_paused.await_args_list] == [True, False]
    asyncio.run(admin.aclose())


def test_dynamo_nixl_update_failure_is_terminal():
    admin = admin_for(worker(1))

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_collective_rpc", new=AsyncMock(side_effect=[None, RuntimeError("failed")])),
        patch.object(admin, "_set_generation_paused", new=AsyncMock()),
    ):
        asyncio.run(
            admin.initialize_nixl(
                host="trainer",
                port=8001,
                timeout=10,
                inference_world_size=1,
                session_id="test-session",
            )
        )
        with pytest.raises(RuntimeError, match="restart is required"):
            asyncio.run(admin.update_weights(None, transport="nixl", step=1))

    assert admin._terminal
    asyncio.run(admin.aclose())
