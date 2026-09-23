import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.inference.dynamo import (
    DynamoAdminPlane,
    DynamoDiscoveryPending,
    parse_dynamo_worker,
    parse_dynamo_workers,
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
    world_size: int = 1,
) -> dict:
    return {
        "instance_id": instance_id,
        "admin_base_url": admin_base_url or f"http://worker:{8200 + instance_id}",
        "world_size": world_size,
        "model": model,
    }


def python_worker(
    *,
    host: str = "10.0.0.8",
    world_size: int = 4,
    routes: list[str] | None = None,
    enable_lora: bool = False,
    component: str = "backend",
    instance_id: int = 7,
    system_port: int = 8081,
) -> dict:
    worker_routes = routes or [
        "pause_generation",
        "resume_generation",
        "init_weights_update_group",
        "update_weights_from_distributed",
    ]
    if enable_lora:
        worker_routes = [*worker_routes, "load_lora", "unload_lora"]
    return {
        "component": component,
        "endpoint": "rl",
        "instance_id": instance_id,
        "system_url": f"http://{host}:{system_port}",
        "world_size": world_size,
        "model": MODEL,
        "routes": worker_routes,
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
    discovered = parsed(*workers)
    workers = (discovered,)
    admin._bind(workers, admin._topology_fingerprint(workers), [AsyncMock()])
    return admin


def python_admin() -> DynamoAdminPlane:
    config = ClientConfig(
        base_url="http://frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://frontend:8001"},
    )
    discovered = parse_dynamo_worker(snapshot(python_worker()), MODEL, expected_admin_host="frontend")
    admin = DynamoAdminPlane(config, MODEL, poll_interval=0)
    workers = (discovered,)
    admin._bind(workers, admin._topology_fingerprint(workers), [AsyncMock()])
    return admin


def python_pd_admin() -> tuple[DynamoAdminPlane, AsyncMock, AsyncMock]:
    config = ClientConfig(
        base_url="http://frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://frontend:8001"},
    )
    workers = parse_dynamo_workers(
        snapshot(
            python_worker(enable_lora=True, component="prefill", instance_id=8, system_port=8083),
            python_worker(enable_lora=True, component="backend", instance_id=9, system_port=8082),
        ),
        MODEL,
        expected_admin_host="frontend",
    )
    admin = DynamoAdminPlane(config, MODEL, poll_interval=0)
    decode_client = AsyncMock()
    prefill_client = AsyncMock()
    admin._bind(workers, admin._topology_fingerprint(workers), [decode_client, prefill_client])
    return admin, decode_client, prefill_client


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


def test_parse_dynamo_workers_accepts_prefill_decode_pair_in_stable_order():
    decode = python_worker(
        enable_lora=True,
        component="backend",
        instance_id=9,
        system_port=8082,
    )
    prefill = python_worker(
        enable_lora=True,
        component="prefill",
        instance_id=8,
        system_port=8083,
    )

    discovered = parse_dynamo_workers(
        snapshot(prefill, decode),
        MODEL,
        expected_admin_host="frontend",
    )

    assert [(worker.component, worker.instance_id) for worker in discovered] == [
        ("backend", 9),
        ("prefill", 8),
    ]


@pytest.mark.parametrize(
    ("worker_update", "error", "match"),
    [
        ({"admin_base_url": None}, DynamoDiscoveryPending, "admin_base_url"),
        ({"world_size": 0}, ValueError, "greater than 0"),
    ],
)
def test_parse_dynamo_worker_validates_required_metadata(worker_update, error, match):
    payload = {**worker(1), **worker_update}
    with pytest.raises(error, match=match):
        parse_dynamo_worker(snapshot(payload), MODEL, expected_admin_host="worker")


def test_parse_dynamo_python_worker_accepts_tp4_concrete_pod_ip():
    discovered = parse_dynamo_worker(snapshot(python_worker()), MODEL, expected_admin_host="frontend")

    assert discovered.admin_base_url == "http://10.0.0.8:8081"
    assert discovered.world_size == 4
    assert discovered.admin_contract == "engine_routes"


@pytest.mark.parametrize(
    ("payload", "error", "match"),
    [
        (python_worker(host="127.0.0.1"), ValueError, "loopback"),
        (python_worker(host="0.0.0.0"), ValueError, "wildcard"),
        (python_worker(host="worker.example"), ValueError, "IP address"),
        (python_worker(host="169.254.169.254"), ValueError, "unsafe"),
        (python_worker(host="[::ffff:127.0.0.1]"), ValueError, "loopback"),
        (
            python_worker(routes=["pause_generation"]),
            DynamoDiscoveryPending,
            "required admin routes",
        ),
    ],
)
def test_parse_dynamo_python_worker_rejects_unsafe_or_incomplete_admin(payload, error, match):
    with pytest.raises(error, match=match):
        parse_dynamo_worker(snapshot(payload), MODEL, expected_admin_host="frontend")


def test_dynamo_admin_plane_factory_pins_two_identical_snapshots():
    discovered_worker = parsed(worker(1))
    discover = AsyncMock(side_effect=[(discovered_worker,), (discovered_worker,)])
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


@pytest.mark.parametrize(("api_key", "authorization"), [("secret", "Bearer secret"), ("EMPTY", None)])
def test_dynamo_worker_client_propagates_admin_headers(monkeypatch, api_key, authorization):
    monkeypatch.setenv("DYNAMO_HEADER", "from-env")
    monkeypatch.setenv("DYNAMO_API_KEY", api_key)
    config = ClientConfig(
        base_url="http://worker:8000/v1",
        headers={"X-Static": "static"},
        headers_from_env={"X-Environment": "DYNAMO_HEADER"},
        api_key_var="DYNAMO_API_KEY",
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, MODEL)
    client = admin._make_worker_client(parsed(worker(1)))

    assert client.headers["X-Static"] == "static"
    assert client.headers["X-Environment"] == "from-env"
    assert client.headers.get("Authorization") == authorization

    asyncio.run(client.aclose())
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
        patch.object(admin, "_discover", new=AsyncMock(side_effect=[(changed,), (changed,)])),
        pytest.raises(RuntimeError, match="topology changed"),
    ):
        asyncio.run(admin.ensure_topology_current())

    asyncio.run(admin.aclose())


def test_dynamo_nccl_lifecycle_initializes_and_updates_weights(tmp_path):
    admin = admin_for(worker(3, world_size=4))

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_collective_rpc", new=AsyncMock()) as collective_rpc,
        patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock()) as post,
    ):
        with pytest.raises(ValueError, match="world size"):
            asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=2))
        collective_rpc.assert_not_awaited()
        assert admin._nccl_initialization_state == "uninitialized"

        with pytest.raises(RuntimeError, match="ready NCCL initialization"):
            asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

        asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=4))
        asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

    assert collective_rpc.await_args_list[0].kwargs["args"] == ["trainer", 29501, 0, 4, 10, "default"]
    assert collective_rpc.await_args_list[1].kwargs["args"] == [(tmp_path / "step_1").as_posix()]
    assert [call.args[1] for call in post.await_args_list] == ["/pause", "/resume"]
    assert admin._nccl_initialization_state == "ready"
    asyncio.run(admin.aclose())


def test_dynamo_collective_rpc_requires_one_result_per_inference_rank():
    admin = admin_for(worker(3, world_size=4))
    response = Mock()
    response.json.return_value = {"results": [None, None, None, None]}
    client = AsyncMock()
    client.post.return_value = response

    asyncio.run(admin._collective_rpc(client, method="init_broadcaster", timeout=10, args=[]))

    response.raise_for_status.assert_called_once_with()
    response.json.return_value = {"results": [None]}
    with pytest.raises(ValueError, match="invalid collective RPC response"):
        asyncio.run(admin._collective_rpc(client, method="init_broadcaster", timeout=10, args=[]))

    response.json.return_value = {"results": [None, None, "unexpected", None]}
    with pytest.raises(ValueError, match="invalid collective RPC response"):
        asyncio.run(admin._collective_rpc(client, method="init_broadcaster", timeout=10, args=[]))

    asyncio.run(admin.aclose())


def test_dynamo_python_routes_initialize_and_update_tp4(tmp_path):
    admin = python_admin()
    response = AsyncMock()
    response.raise_for_status = lambda: None
    response.json = lambda: {"status": "ok"}
    admin.clients[0].post.return_value = response

    with patch.object(admin, "ensure_topology_current", new=AsyncMock()):
        asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=4))
        asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

    calls = admin.clients[0].post.await_args_list
    assert [call.args[0] for call in calls] == [
        "/engine/init_weights_update_group",
        "/engine/pause_generation",
        "/engine/update_weights_from_distributed",
        "/engine/resume_generation",
    ]
    assert calls[0].kwargs["json"] == {
        "engine_rpc": "init_broadcaster",
        "host": "trainer",
        "port": 29501,
        "rank_offset": 0,
        "inference_world_size": 4,
        "timeout": 10,
        "session_id": "default",
    }
    assert calls[1].kwargs["json"] == {"mode": "keep", "clear_cache": False}
    assert calls[2].kwargs["json"] == {
        "engine_rpc": "update_weights_from_path",
        "weight_dir": (tmp_path / "step_1").as_posix(),
        "weight_version": 1,
    }
    assert calls[3].kwargs["json"] == {}
    asyncio.run(admin.aclose())


def test_dynamo_python_routes_reject_configured_world_size_mismatch_before_mutation():
    admin = python_admin()

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        pytest.raises(ValueError, match="world size"),
    ):
        asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=2))

    admin.clients[0].post.assert_not_awaited()
    assert admin._nccl_initialization_state == "uninitialized"
    asyncio.run(admin.aclose())


def test_dynamo_python_routes_reject_multi_worker_nccl():
    admin, decode_client, prefill_client = python_pd_admin()

    with pytest.raises(ValueError, match="exactly one inference worker"):
        asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=2))

    decode_client.post.assert_not_awaited()
    prefill_client.post.assert_not_awaited()
    assert admin._nccl_initialization_state == "uninitialized"
    asyncio.run(admin.aclose())


def test_dynamo_bind_requires_one_client_per_worker():
    admin, _, _ = python_pd_admin()

    with pytest.raises(ValueError, match="counts must match"):
        admin._bind(admin._workers, admin._topology_fingerprint(admin._workers), [AsyncMock()])

    asyncio.run(admin.aclose())


def test_dynamo_python_worker_does_not_receive_frontend_credentials(monkeypatch):
    monkeypatch.setenv("DYNAMO_TEST_API_KEY", "secret")
    config = ClientConfig(
        base_url="http://frontend:8000/v1",
        skip_model_check=True,
        api_key_var="DYNAMO_TEST_API_KEY",
        headers={"X-Frontend-Only": "value"},
        wait_for_ready_timeout=2,
        dynamo={"discovery_url": "http://frontend:8001"},
    )
    discovered = parse_dynamo_worker(snapshot(python_worker()), MODEL, expected_admin_host="frontend")
    admin = DynamoAdminPlane(config, MODEL, poll_interval=0)
    client = admin._make_worker_client(discovered)

    assert "authorization" not in client.headers
    assert "x-frontend-only" not in client.headers
    asyncio.run(client.aclose())
    asyncio.run(admin.aclose())


def test_dynamo_python_pause_retries_transient_transport_failure():
    admin = python_admin()
    response = AsyncMock()
    response.raise_for_status = lambda: None
    response.json = lambda: {"status": "ok"}
    admin.clients[0].post.side_effect = [httpx.ConnectError("reset"), response]

    with patch("prime_rl.inference.dynamo.asyncio.sleep", new=AsyncMock()) as sleep:
        asyncio.run(admin._set_generation_paused(True))

    assert admin.clients[0].post.await_count == 2
    sleep.assert_awaited_once()
    asyncio.run(admin.aclose())


@pytest.mark.parametrize("transport", ["filesystem", "nixl"])
def test_dynamo_python_worker_rejects_unsupported_weight_transports(tmp_path, transport):
    admin = python_admin()

    with pytest.raises(ValueError, match="only supports NCCL"):
        asyncio.run(admin.update_weights(tmp_path, transport=transport))

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


def test_dynamo_python_worker_loads_versioned_filesystem_lora(tmp_path):
    discovered = parse_dynamo_worker(snapshot(python_worker(enable_lora=True)), MODEL, expected_admin_host="frontend")
    admin = python_admin()
    workers = (discovered,)
    admin._bind(workers, admin._topology_fingerprint(workers), [admin.clients[0]])
    response = AsyncMock()
    response.raise_for_status = lambda: None
    response.json = lambda: {
        "status": "success",
        "lora_name": "prime-rl-policy-v1-test",
        "lora_id": 17,
    }
    admin.clients[0].post.return_value = response
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_lora_name", return_value="prime-rl-policy-v1-test"),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()) as check_model,
    ):
        active_model = asyncio.run(admin.load_lora_adapter(MODEL, adapter, step=1))

    assert active_model == "prime-rl-policy-v1-test"
    admin.clients[0].post.assert_awaited_once_with(
        "/engine/load_lora",
        json={
            "lora_name": "prime-rl-policy-v1-test",
            "source": {"uri": adapter.as_uri()},
        },
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )
    check_model.assert_awaited_once()
    asyncio.run(admin.aclose())


def test_dynamo_python_workers_load_decode_before_prefill(tmp_path):
    admin, decode_client, prefill_client = python_pd_admin()
    call_order: list[str] = []

    def response(name: str, lora_name: str):
        result = AsyncMock()
        result.raise_for_status = lambda: None
        result.json = lambda: {"status": "success", "lora_name": lora_name, "lora_id": 17}
        call_order.append(name)
        return result

    decode_client.post.side_effect = lambda *args, **kwargs: response("decode", kwargs["json"]["lora_name"])
    prefill_client.post.side_effect = lambda *args, **kwargs: response("prefill", kwargs["json"]["lora_name"])
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_lora_name", return_value="prime-rl-policy-v1-test"),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
    ):
        active_model = asyncio.run(admin.load_lora_adapter(MODEL, adapter, step=1))

    assert active_model == "prime-rl-policy-v1-test"
    assert call_order == ["decode", "prefill"]
    asyncio.run(admin.aclose())


def test_dynamo_python_workers_roll_back_partial_lora_load(tmp_path):
    admin, decode_client, prefill_client = python_pd_admin()
    call_order: list[tuple[str, str]] = []

    def response(name: str, operation: str, lora_name: str):
        result = AsyncMock()
        result.raise_for_status = lambda: None
        result.json = lambda: {"status": "success", "lora_name": lora_name, "lora_id": 17}
        call_order.append((name, operation))
        return result

    decode_client.post.side_effect = lambda path, **kwargs: response(
        "decode", path.rsplit("/", 1)[-1], kwargs["json"]["lora_name"]
    )
    prefill_client.post.side_effect = RuntimeError("prefill load failed")
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_lora_name", return_value="prime-rl-policy-v1-test"),
        pytest.raises(RuntimeError, match="prefill load failed"),
    ):
        asyncio.run(admin.load_lora_adapter(MODEL, adapter, step=1))

    assert call_order == [("decode", "load_lora"), ("decode", "unload_lora")]
    asyncio.run(admin.aclose())


def test_dynamo_python_workers_retry_partial_lora_unload():
    admin, decode_client, prefill_client = python_pd_admin()
    events: list[tuple[str, str]] = []

    def result(payload: dict):
        response = AsyncMock()
        response.raise_for_status = lambda: None
        response.json = lambda: payload
        return response

    def prefill_response(path, **kwargs):
        events.append(("prefill", path.rsplit("/", 1)[-1]))
        if len([event for event in events if event[0] == "prefill"]) == 1:
            return result({"status": "success", "lora_name": kwargs["json"]["lora_name"], "lora_id": 17})
        return result({"status": "error", "message": "LoRA adapter not found"})

    def decode_response(path, **kwargs):
        events.append(("decode", path.rsplit("/", 1)[-1]))
        if len([event for event in events if event[0] == "decode"]) == 1:
            raise RuntimeError("decode unload failed")
        return result({"status": "success", "lora_name": kwargs["json"]["lora_name"], "lora_id": 17})

    prefill_client.post.side_effect = prefill_response
    decode_client.post.side_effect = decode_response
    body = {"lora_name": "prime-rl-policy-v0-test"}

    with pytest.raises(RuntimeError, match="decode unload failed"):
        asyncio.run(admin._post_lora_all("unload_lora", body))
    asyncio.run(admin._post_lora_all("unload_lora", body))

    assert events == [
        ("prefill", "unload_lora"),
        ("decode", "unload_lora"),
        ("prefill", "unload_lora"),
        ("decode", "unload_lora"),
    ]
    asyncio.run(admin.aclose())


def test_dynamo_python_worker_requires_lora_routes(tmp_path):
    admin = python_admin()
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    with pytest.raises(RuntimeError, match="required LoRA routes"):
        asyncio.run(admin.load_lora_adapter(MODEL, adapter, step=1))

    admin.clients[0].post.assert_not_awaited()
    asyncio.run(admin.aclose())
