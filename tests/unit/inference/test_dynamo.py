import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.inference.dynamo import (
    DynamoAdminPlane,
    _discovery_headers,
    discover_dynamo_workers,
    topology_fingerprint,
)
from prime_rl.inference.dynamo import parse_dynamo_workers as _parse_dynamo_workers
from prime_rl.orchestrator.clients import setup_admin_plane

DEFAULT_ADMIN_HOST_ALLOWLIST = ("worker", "worker-1", "worker-2", "worker-3", "worker-9")


def dynamo_config() -> dict:
    return {
        "discovery_url": "http://dynamo-frontend:8001",
        "expected_namespace": "dynamo",
        "admin_host_allowlist": list(DEFAULT_ADMIN_HOST_ALLOWLIST),
    }


def parse_dynamo_workers(
    payload,
    model_name,
    *,
    expected_namespace="dynamo",
    admin_host_allowlist=DEFAULT_ADMIN_HOST_ALLOWLIST,
    admin_origin_allowlist=None,
):
    kwargs = {
        "expected_namespace": expected_namespace,
        "admin_host_allowlist": admin_host_allowlist,
    }
    if admin_origin_allowlist is not None:
        kwargs["admin_origin_allowlist"] = admin_origin_allowlist
    return _parse_dynamo_workers(payload, model_name, **kwargs)


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
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)
    admin.workers = parse_dynamo_workers(snapshot(*worker_specs), "Qwen/Qwen3-0.6B")
    return admin


@pytest.mark.parametrize(
    "dynamo,match",
    [
        ({**dynamo_config(), "api_key_var": "DISCOVERY_TOKEN"}, "use HTTPS"),
        (
            {
                **dynamo_config(),
                "discovery_url": "https://dynamo-frontend:8001",
                "admin_api_key_var": "ADMIN_TOKEN",
            },
            "admin_origin_allowlist",
        ),
        (
            {
                **dynamo_config(),
                "discovery_url": "https://dynamo-frontend:8001",
                "admin_origin_allowlist": ["http://worker:8120"],
                "admin_api_key_var": "ADMIN_TOKEN",
            },
            "HTTPS or loopback",
        ),
    ],
)
def test_dynamo_admin_plane_validates_credential_origins(dynamo, match):
    with pytest.raises(ValueError, match=match):
        DynamoAdminPlane(ClientConfig(dynamo=dynamo), "Qwen/Qwen3-0.6B")


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


def test_parse_dynamo_workers_requires_world_size():
    incomplete = worker(1, admin_base_url="http://worker-1:8120", world_size=1)
    incomplete.pop("world_size")

    with pytest.raises(RuntimeError, match="missing required RL metadata"):
        parse_dynamo_workers(snapshot(incomplete), "Qwen/Qwen3-0.6B")


def test_admin_plane_factory_selects_dynamo():
    admin = setup_admin_plane(ClientConfig(dynamo=dynamo_config()), "Qwen/Qwen3-0.6B")

    assert isinstance(admin, DynamoAdminPlane)
    asyncio.run(admin.aclose())


@pytest.mark.parametrize(
    "payload",
    [
        {"protocol_version": 2, "namespace": "dynamo", "workers": []},
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=0)),
        snapshot(worker(1, admin_base_url="file:///tmp/admin", world_size=1)),
    ],
)
def test_parse_dynamo_workers_rejects_invalid_contract(payload):
    with pytest.raises(ValueError):
        parse_dynamo_workers(payload, "Qwen/Qwen3-0.6B")


@pytest.mark.parametrize(
    "payload",
    [
        {**snapshot(), "namespace": "other"},
        snapshot({**worker(1, admin_base_url="http://worker-1:8120", world_size=1), "namespace": "other"}),
    ],
)
def test_parse_dynamo_workers_rejects_unexpected_namespace(payload):
    with pytest.raises(ValueError, match="namespace"):
        parse_dynamo_workers(payload, "Qwen/Qwen3-0.6B")


def test_parse_dynamo_workers_rejects_admin_host_outside_allowlist():
    with pytest.raises(ValueError, match="allowlist"):
        parse_dynamo_workers(
            snapshot(worker(1, admin_base_url="http://untrusted:8120", world_size=1)),
            "Qwen/Qwen3-0.6B",
        )


@pytest.mark.parametrize(
    "admin_base_url",
    [
        "http://user:password@worker-1:8120",
        "http://worker-1:8120/admin",
        "http://worker-1:8120?redirect=worker-2",
        "http://worker-1:8120#fragment",
        "http://worker-1:70000",
    ],
)
def test_parse_dynamo_workers_rejects_malformed_admin_url(admin_base_url):
    with pytest.raises(ValueError, match="admin_base_url"):
        parse_dynamo_workers(
            snapshot(worker(1, admin_base_url=admin_base_url, world_size=1)),
            "Qwen/Qwen3-0.6B",
        )


def test_parse_dynamo_workers_rejects_admin_origin_mismatch():
    with pytest.raises(ValueError, match="origin allowlist"):
        parse_dynamo_workers(
            snapshot(worker(1, admin_base_url="https://worker-1:8121", world_size=1)),
            "Qwen/Qwen3-0.6B",
            admin_origin_allowlist=("https://worker-1:8120",),
        )


@pytest.mark.parametrize("error", ["line one\nforged line", "status \u202ereversed"])
def test_parse_dynamo_workers_rejects_untrusted_worker_error(error):
    with pytest.raises(ValueError, match="worker error"):
        parse_dynamo_workers(
            snapshot({**worker(1, admin_base_url="http://worker-1:8120", world_size=1), "error": error}),
            "Qwen/Qwen3-0.6B",
        )


def test_parse_dynamo_workers_rejects_duplicate_admin_urls():
    with pytest.raises(ValueError, match="duplicate worker admin endpoints"):
        parse_dynamo_workers(
            snapshot(
                worker(1, admin_base_url="http://WORKER", world_size=1),
                worker(2, admin_base_url="http://worker:80/", world_size=1),
            ),
            "Qwen/Qwen3-0.6B",
        )


def test_discovery_headers_do_not_inherit_data_plane_credentials(monkeypatch):
    monkeypatch.setenv("DATA_API_KEY", "data-secret")
    monkeypatch.setenv("DATA_HEADER", "data-header-secret")
    monkeypatch.setenv("DISCOVERY_API_KEY", "discovery-secret")
    monkeypatch.setenv("DISCOVERY_HEADER", "discovery-header")
    config = ClientConfig(
        api_key_var="DATA_API_KEY",
        headers={"X-Data": "data-secret"},
        headers_from_env={"X-Data-Env": "DATA_HEADER"},
        dynamo={
            **dynamo_config(),
            "discovery_url": "https://dynamo-frontend:8001",
            "api_key_var": "DISCOVERY_API_KEY",
            "headers_from_env": {"X-Discovery": "DISCOVERY_HEADER"},
        },
    )

    headers = _discovery_headers(config)
    assert headers["Authorization"] == "Bearer discovery-secret"
    assert headers["X-Discovery"] == "discovery-header"
    assert headers["Accept-Encoding"] == "identity"
    assert "X-Data" not in headers
    assert "X-Data-Env" not in headers


class ChunkedDiscoveryResponse:
    headers: dict[str, str] = {}

    def raise_for_status(self):
        return None

    async def aiter_raw(self):
        yield b"x" * (512 * 1024)
        yield b"x" * (512 * 1024 + 1)


class SlowDiscoveryResponse(ChunkedDiscoveryResponse):
    async def aiter_raw(self):
        await asyncio.sleep(0.05)
        yield b"{}"


class EncodedDiscoveryResponse(ChunkedDiscoveryResponse):
    headers = {"Content-Encoding": "gzip"}

    async def aiter_raw(self):
        yield b"{}"


class DiscoveryStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class StreamingDiscoveryClient:
    def __init__(self, response=None):
        self.response = response or ChunkedDiscoveryResponse()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    def stream(self, method, url):
        return DiscoveryStreamContext(self.response)


@pytest.mark.parametrize(
    "discovery_url",
    [
        "dynamo-frontend:8001",
        "ftp://dynamo-frontend:8001",
        "http://user:password@dynamo-frontend:8001",
        "http://dynamo-frontend:8001?namespace=other",
        "http://dynamo-frontend:70000",
    ],
)
def test_discover_dynamo_workers_rejects_invalid_url(discovery_url):
    with pytest.raises(ValueError, match="discovery_url"):
        asyncio.run(
            discover_dynamo_workers(
                discovery_url,
                "Qwen/Qwen3-0.6B",
                headers={},
                timeout=2,
                expected_namespace="dynamo",
                admin_host_allowlist=("worker-1",),
            )
        )


def test_discover_dynamo_workers_caps_response_body():
    with (
        patch("prime_rl.inference.dynamo.httpx.AsyncClient", return_value=StreamingDiscoveryClient()),
        pytest.raises(ValueError, match="response body exceeds"),
    ):
        asyncio.run(
            discover_dynamo_workers(
                "http://dynamo-frontend:8001",
                "Qwen/Qwen3-0.6B",
                headers={},
                timeout=2,
                expected_namespace="dynamo",
                admin_host_allowlist=("worker-1",),
            )
        )


def test_discover_dynamo_workers_rejects_encoded_response():
    with (
        patch(
            "prime_rl.inference.dynamo.httpx.AsyncClient",
            return_value=StreamingDiscoveryClient(EncodedDiscoveryResponse()),
        ),
        pytest.raises(ValueError, match="Content-Encoding"),
    ):
        asyncio.run(
            discover_dynamo_workers(
                "http://dynamo-frontend:8001",
                "Qwen/Qwen3-0.6B",
                headers={},
                timeout=2,
                expected_namespace="dynamo",
                admin_host_allowlist=("worker-1",),
            )
        )


def test_discover_dynamo_workers_has_total_request_deadline():
    with (
        patch(
            "prime_rl.inference.dynamo.httpx.AsyncClient",
            return_value=StreamingDiscoveryClient(SlowDiscoveryResponse()),
        ),
        pytest.raises(TimeoutError, match="discovery request"),
    ):
        asyncio.run(
            discover_dynamo_workers(
                "http://dynamo-frontend:8001",
                "Qwen/Qwen3-0.6B",
                headers={},
                timeout=0.01,
                expected_namespace="dynamo",
                admin_host_allowlist=("worker-1",),
            )
        )


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
    admin = DynamoAdminPlane(
        ClientConfig(
            base_url="http://dynamo-frontend:8000/v1",
            skip_model_check=True,
            wait_for_ready_timeout=2,
            dynamo=dynamo_config(),
        ),
        "Qwen/Qwen3-0.6B",
        poll_interval=0,
    )

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
        patch.object(admin, "_check_admin_capabilities", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))

    assert discover.await_count == 4
    assert str(admin.clients[0].base_url) == "http://worker-1:8120"
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_fails_closed_on_topology_drift():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    changed = parse_dynamo_workers(
        snapshot(worker(2, admin_base_url="http://worker-2:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    admin._fingerprint = topology_fingerprint(admin.workers)

    with (
        patch.object(admin, "_discover", new=AsyncMock(side_effect=[changed, changed])),
        pytest.raises(RuntimeError, match="topology changed"),
    ):
        asyncio.run(admin.ensure_topology_current())

    asyncio.run(admin.aclose())


def test_dynamo_direct_admin_clients_use_only_dedicated_credentials(monkeypatch):
    monkeypatch.setenv("DATA_API_KEY", "data-secret")
    monkeypatch.setenv("DISCOVERY_API_KEY", "discovery-secret")
    monkeypatch.setenv("ADMIN_API_KEY", "admin-secret")
    monkeypatch.setenv("ADMIN_HEADER", "admin-header")
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        api_key_var="DATA_API_KEY",
        headers={"X-Data": "data-secret"},
        dynamo={
            **dynamo_config(),
            "discovery_url": "https://dynamo-frontend:8001",
            "api_key_var": "DISCOVERY_API_KEY",
            "admin_api_key_var": "ADMIN_API_KEY",
            "admin_headers_from_env": {"X-Admin": "ADMIN_HEADER"},
            "admin_origin_allowlist": ["https://worker-1:8120"],
        },
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B")
    workers = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="https://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    admin._bind(workers, topology_fingerprint(workers))

    assert admin.clients[0].headers["Authorization"] == "Bearer admin-secret"
    assert admin.clients[0].headers["X-Admin"] == "admin-header"
    assert "X-Data" not in admin.clients[0].headers
    assert admin._frontend_clients[0].headers["Authorization"] == "Bearer data-secret"
    assert "X-Admin" not in admin._frontend_clients[0].headers
    assert _discovery_headers(config)["Authorization"] == "Bearer discovery-secret"
    assert "X-Admin" not in _discovery_headers(config)
    asyncio.run(admin.aclose())


def test_dynamo_capability_probe_requires_collective_rpc_on_every_worker():
    admin = dynamo_admin(
        worker(1, admin_base_url="http://worker-1:8120", world_size=1),
        worker(2, admin_base_url="http://worker-2:8120", world_size=1),
    )

    with (
        patch.object(admin, "_collective_rpc", new=AsyncMock(side_effect=[None, RuntimeError("missing route")])) as rpc,
        pytest.raises(RuntimeError, match="required vLLM development admin contract"),
    ):
        asyncio.run(admin._check_admin_capabilities([AsyncMock(), AsyncMock()], admin.workers, timeout=1))

    assert rpc.await_count == 2
    asyncio.run(admin.aclose())


@pytest.mark.parametrize(
    "payload",
    [{"results": []}, {"results": ["unexpected"]}, {"results": [None, None]}, {"other": [None]}],
)
def test_dynamo_collective_rpc_rejects_invalid_result_contract(payload):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    response = successful_response()
    response.json.return_value = payload

    with (
        patch("prime_rl.inference.dynamo._bounded_request", new=AsyncMock(return_value=response)),
        pytest.raises(ValueError, match="invalid collective RPC response"),
    ):
        asyncio.run(
            admin._collective_rpc(
                AsyncMock(),
                method="liveness_probe",
                timeout=1,
                args=[],
                expected_result_count=1,
            )
        )

    asyncio.run(admin.aclose())


def test_dynamo_collective_rpc_accepts_null_results():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    response = successful_response()
    response.json.return_value = {"results": [None]}

    with patch("prime_rl.inference.dynamo._bounded_request", new=AsyncMock(return_value=response)):
        asyncio.run(
            admin._collective_rpc(
                AsyncMock(),
                method="liveness_probe",
                timeout=1,
                args=[],
                expected_result_count=1,
            )
        )

    asyncio.run(admin.aclose())


def test_dynamo_nccl_initialization_assigns_rank_offsets_and_validates_results():
    admin = dynamo_admin(
        worker(3, admin_base_url="http://worker-3:8120", world_size=2),
        worker(9, admin_base_url="http://worker-9:8120", world_size=1),
    )
    admin.clients = [AsyncMock(), AsyncMock()]

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_collective_rpc", new=AsyncMock()) as collective_rpc,
    ):
        asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=3))

    assert [call.kwargs["args"][2] for call in collective_rpc.await_args_list] == [0, 2]
    assert [call.kwargs["expected_result_count"] for call in collective_rpc.await_args_list] == [2, 1]
    asyncio.run(admin.aclose())


def test_dynamo_nccl_initialization_is_one_shot():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin.clients = [AsyncMock()]

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_collective_rpc", new=AsyncMock()),
    ):
        asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=1))
        with pytest.raises(RuntimeError, match="only be initialized once"):
            asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=1))

    asyncio.run(admin.aclose())


def test_dynamo_nccl_partial_initialization_requires_restart():
    admin = dynamo_admin(
        worker(1, admin_base_url="http://worker-1:8120", world_size=1),
        worker(2, admin_base_url="http://worker-2:8120", world_size=1),
    )
    admin.clients = [AsyncMock(), AsyncMock()]

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch.object(admin, "_collective_rpc", new=AsyncMock(side_effect=[RuntimeError("failed"), None])),
        pytest.raises(RuntimeError, match="must restart"),
    ):
        asyncio.run(admin.initialize_nccl(host="trainer", port=29501, timeout=10, inference_world_size=2))

    assert admin._nccl_initialization_state == "terminal"
    asyncio.run(admin.aclose())


def test_dynamo_nccl_update_requires_initialization(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin.clients = [AsyncMock()]

    with pytest.raises(RuntimeError, match="ready NCCL initialization"):
        asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

    asyncio.run(admin.aclose())


def test_dynamo_nccl_update_uses_collective_rpc_and_returns_to_ready(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin.clients = [AsyncMock()]
    admin._nccl_initialization_state = "ready"

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock(return_value=successful_response())) as post,
        patch.object(admin, "_collective_rpc", new=AsyncMock()) as collective_rpc,
    ):
        asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

    assert [call.args[1] for call in post.await_args_list] == ["/pause", "/resume"]
    collective_rpc.assert_awaited_once()
    assert admin._nccl_initialization_state == "ready"
    assert admin._control_terminal is False
    asyncio.run(admin.aclose())


def test_dynamo_nccl_update_failure_is_terminal_and_skips_resume(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin.clients = [AsyncMock()]
    admin._nccl_initialization_state = "ready"

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock(return_value=successful_response())) as post,
        patch.object(admin, "_collective_rpc", new=AsyncMock(side_effect=RuntimeError("receive failed"))),
        pytest.raises(RuntimeError, match="engines remain paused"),
    ):
        asyncio.run(admin.update_weights(tmp_path / "step_1", transport="nccl", step=1))

    assert [call.args[1] for call in post.await_args_list] == ["/pause"]
    assert admin._nccl_initialization_state == "terminal"
    assert admin._control_terminal is True
    asyncio.run(admin.aclose())
