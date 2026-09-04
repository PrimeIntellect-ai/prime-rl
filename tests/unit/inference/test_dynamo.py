import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prime_rl.configs.shared import ClientConfig
from prime_rl.inference.dynamo import (
    DynamoAdminPlane,
    DynamoDiscoveryPending,
    _discovery_headers,
    discover_dynamo_workers,
    topology_fingerprint,
)
from prime_rl.inference.dynamo import (
    parse_dynamo_workers as _parse_dynamo_workers,
)
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
        (
            {
                **dynamo_config(),
                "api_key_var": "DYNAMO_DISCOVERY_TOKEN",
            },
            "use HTTPS",
        ),
        (
            {
                **dynamo_config(),
                "discovery_url": "https://dynamo-frontend:8001",
                "admin_api_key_var": "DYNAMO_ADMIN_TOKEN",
            },
            "admin_origin_allowlist",
        ),
        (
            {
                **dynamo_config(),
                "discovery_url": "https://dynamo-frontend:8001",
                "admin_origin_allowlist": ["http://worker:8120"],
                "admin_api_key_var": "DYNAMO_ADMIN_TOKEN",
            },
            "HTTPS or loopback",
        ),
    ],
)
def test_dynamo_admin_plane_validates_credential_origins(dynamo, match):
    config = ClientConfig(dynamo=dynamo)

    with pytest.raises(ValueError, match=match):
        DynamoAdminPlane(config, "Qwen/Qwen3-0.6B")


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


def test_parse_dynamo_workers_allows_protocol_v1_without_unused_system_url():
    worker_without_system_url = worker(1, admin_base_url="http://worker-1:8120", world_size=1)
    worker_without_system_url.pop("system_url")

    parsed = parse_dynamo_workers(snapshot(worker_without_system_url), "Qwen/Qwen3-0.6B")

    assert parsed[0].system_url is None
    assert topology_fingerprint(parsed)[0][6] is None


def test_parse_dynamo_workers_requires_world_size_for_admin_worker():
    filesystem_worker = worker(1, admin_base_url="http://worker-1:8120", world_size=1)
    filesystem_worker.pop("world_size")

    with pytest.raises(DynamoDiscoveryPending, match="missing required RL metadata"):
        parse_dynamo_workers(snapshot(filesystem_worker), "Qwen/Qwen3-0.6B")


def test_admin_plane_factory_selects_dynamo():
    config = ClientConfig(dynamo=dynamo_config())

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
                worker(1, admin_base_url="http://WORKER", world_size=1),
                worker(2, admin_base_url="http://worker:80/", world_size=1),
            ),
            "Qwen/Qwen3-0.6B",
        )


@pytest.mark.parametrize("path", ["/admin/prefill", "/admin/replica/../decode"])
def test_parse_dynamo_workers_rejects_admin_paths(path):
    with pytest.raises(ValueError, match="admin_base_url must be an origin without a path"):
        parse_dynamo_workers(
            snapshot(worker(1, admin_base_url=f"http://worker{path}", world_size=1)),
            "Qwen/Qwen3-0.6B",
        )


def test_parse_dynamo_workers_treats_incomplete_matching_worker_as_pending():
    incomplete_worker = worker(1, admin_base_url="http://worker-1:8120", world_size=1)
    incomplete_worker.pop("admin_base_url")

    with pytest.raises(DynamoDiscoveryPending, match="missing required RL metadata"):
        parse_dynamo_workers(snapshot(incomplete_worker), "Qwen/Qwen3-0.6B")


@pytest.mark.parametrize(
    "payload",
    [
        {**snapshot(), "namespace": "other"},
        snapshot(
            {
                **worker(1, admin_base_url="http://worker-1:8120", world_size=1),
                "namespace": "other",
            }
        ),
    ],
)
def test_parse_dynamo_workers_rejects_unexpected_namespace(payload):
    with pytest.raises(ValueError, match="namespace"):
        parse_dynamo_workers(
            payload,
            "Qwen/Qwen3-0.6B",
            expected_namespace="dynamo",
            admin_host_allowlist=("worker-1",),
        )


@pytest.mark.parametrize(
    ("admin_base_url", "admin_host_allowlist"),
    [
        ("http://localhost:8120", ("localhost",)),
        ("http://WORKER-1:8120", ("worker-1",)),
        ("http://10.0.0.12:8120", ("10.0.0.12",)),
        ("http://10.0.0.12:8120", ("10.0.0.0/24",)),
    ],
)
def test_parse_dynamo_workers_accepts_allowlisted_admin_hosts(admin_base_url, admin_host_allowlist):
    parsed = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url=admin_base_url, world_size=1)),
        "Qwen/Qwen3-0.6B",
        expected_namespace="dynamo",
        admin_host_allowlist=admin_host_allowlist,
    )

    assert len(parsed) == 1


def test_parse_dynamo_workers_rejects_admin_host_outside_allowlist():
    with pytest.raises(ValueError, match="allowlist"):
        parse_dynamo_workers(
            snapshot(worker(1, admin_base_url="http://untrusted:8120", world_size=1)),
            "Qwen/Qwen3-0.6B",
            expected_namespace="dynamo",
            admin_host_allowlist=("worker-1",),
        )


def test_parse_dynamo_workers_rejects_admin_port_outside_origin_allowlist():
    with pytest.raises(ValueError, match="origin allowlist"):
        parse_dynamo_workers(
            snapshot(worker(1, admin_base_url="http://worker-1:8121", world_size=1)),
            "Qwen/Qwen3-0.6B",
            expected_namespace="dynamo",
            admin_host_allowlist=("worker-1",),
            admin_origin_allowlist=("http://worker-1:8120",),
        )


def test_parse_dynamo_workers_accepts_canonical_allowlisted_admin_origin():
    parsed = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://WORKER-1", world_size=1)),
        "Qwen/Qwen3-0.6B",
        expected_namespace="dynamo",
        admin_host_allowlist=("worker-1",),
        admin_origin_allowlist=("http://worker-1:80",),
    )

    assert len(parsed) == 1


def test_parse_dynamo_workers_rejects_canonical_duplicate_admin_origins():
    with pytest.raises(ValueError, match="duplicate worker admin endpoints"):
        parse_dynamo_workers(
            snapshot(
                worker(1, admin_base_url="http://WORKER", world_size=1),
                worker(2, admin_base_url="http://worker:80", world_size=1),
            ),
            "Qwen/Qwen3-0.6B",
            expected_namespace="dynamo",
            admin_host_allowlist=("worker",),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {**snapshot(), "workers": [{}] * 129},
        snapshot({**worker(1, admin_base_url="http://worker-1:8120", world_size=1), "component": "x" * 257}),
        snapshot({**worker(1, admin_base_url="http://worker-1:8120", world_size=1), "routes": ["route"] * 129}),
        snapshot({**worker(1, admin_base_url="http://worker-1:8120", world_size=1), "routes": ["x" * 257]}),
        snapshot(
            {
                **worker(1, admin_base_url="http://worker-1:8120", world_size=1),
                "transport": {"nats_tcp": "x" * 2049},
            }
        ),
    ],
)
def test_parse_dynamo_workers_caps_discovery_metadata(payload):
    with pytest.raises(ValueError):
        parse_dynamo_workers(
            payload,
            "Qwen/Qwen3-0.6B",
            expected_namespace="dynamo",
            admin_host_allowlist=("worker-1",),
        )


@pytest.mark.parametrize(
    "error",
    [123, "bad\nforged", "bad\u202eforged", "bad\u2066forged", "bad\u2028forged", "bad\u2029forged", "é" * 1025],
)
def test_parse_dynamo_workers_rejects_untrusted_worker_error_before_interpolation(error):
    with pytest.raises(ValueError, match="worker error"):
        parse_dynamo_workers(
            snapshot(
                {
                    **worker(1, admin_base_url="http://worker-1:8120", world_size=1),
                    "error": error,
                }
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
        headers={"X-Data-Static": "data-secret"},
        headers_from_env={"X-Data-Env": "DATA_HEADER"},
        dynamo={
            **dynamo_config(),
            "discovery_url": "https://dynamo-frontend:8001",
            "admin_host_allowlist": ["worker-1"],
            "api_key_var": "DISCOVERY_API_KEY",
            "headers_from_env": {"X-Discovery-Env": "DISCOVERY_HEADER"},
        },
    )

    assert _discovery_headers(config) == {
        "X-Discovery-Env": "discovery-header",
        "Authorization": "Bearer discovery-secret",
        "Accept-Encoding": "identity",
    }


def test_discovery_headers_force_identity_encoding(monkeypatch):
    monkeypatch.setenv("DISCOVERY_ENCODING", "gzip")
    config = ClientConfig(
        dynamo={
            **dynamo_config(),
            "discovery_url": "https://dynamo-frontend:8001",
            "headers_from_env": {"accept-encoding": "DISCOVERY_ENCODING"},
        }
    )

    headers = httpx.Headers(_discovery_headers(config))

    assert headers.get_list("accept-encoding") == ["identity"]


class ChunkedDiscoveryResponse:
    headers: dict[str, str] = {}

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        yield b"x" * (512 * 1024)
        yield b"x" * (512 * 1024 + 1)

    async def aiter_raw(self):
        yield b"x" * (512 * 1024)
        yield b"x" * (512 * 1024 + 1)


class RawOnlyChunkedDiscoveryResponse(ChunkedDiscoveryResponse):
    async def aiter_bytes(self):
        raise AssertionError("decoded discovery iterator must not be used")
        yield b""


class EncodedDiscoveryResponse(ChunkedDiscoveryResponse):
    headers = {"Content-Encoding": "gzip"}

    async def aiter_bytes(self):
        yield b"{}"

    async def aiter_raw(self):
        yield b"{}"


class SlowDiscoveryResponse(ChunkedDiscoveryResponse):
    async def aiter_bytes(self):
        await asyncio.sleep(0.05)
        yield b"{}"

    async def aiter_raw(self):
        await asyncio.sleep(0.05)
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


def test_discover_dynamo_workers_streams_and_caps_response_body():
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


def test_discover_dynamo_workers_caps_raw_response_body():
    with (
        patch(
            "prime_rl.inference.dynamo.httpx.AsyncClient",
            return_value=StreamingDiscoveryClient(RawOnlyChunkedDiscoveryResponse()),
        ),
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


def test_discover_dynamo_workers_rejects_non_identity_content_encoding():
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


@pytest.mark.parametrize(
    "contract_error",
    [
        "snapshot namespace does not match",
        "worker admin host is not in the configured allowlist",
        "unsupported protocol_version",
    ],
)
def test_dynamo_admin_plane_fails_fast_on_invalid_startup_snapshot(monkeypatch, contract_error):
    workers = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    discover = AsyncMock(side_effect=[ValueError(contract_error), workers, workers])
    monkeypatch.setattr("prime_rl.inference.dynamo.discover_dynamo_workers", discover)
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
        pytest.raises(ValueError, match=contract_error),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))

    assert discover.await_count == 1
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
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
        patch.object(admin, "_check_admin_capabilities", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))

    assert discover.await_count == 4
    assert str(admin.clients[0].base_url) == "http://worker-1:8120"
    assert "authorization" not in admin.clients[0].headers
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_bounds_direct_worker_client_timeouts():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    fingerprint = topology_fingerprint(admin.workers)

    admin._bind(admin.workers, fingerprint)

    assert admin.clients[0].timeout.connect is not None
    assert admin.clients[0].timeout.read is not None
    assert admin.clients[0].timeout.read <= 30
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_bounds_frontend_client_timeouts():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))

    assert admin._frontend_clients[0].timeout.connect is not None
    assert admin._frontend_clients[0].timeout.read is not None
    assert admin._frontend_clients[0].timeout.read <= 30
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_has_total_frontend_readiness_deadline():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin._timeout = 0.01

    async def slow_health_check(*args, **kwargs):
        await asyncio.sleep(0.05)

    try:
        with (
            patch("prime_rl.inference.dynamo.check_health", side_effect=slow_health_check),
            patch.object(admin, "_discover", new=AsyncMock(return_value=admin.workers)),
            pytest.raises(TimeoutError, match="frontend readiness"),
        ):
            asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
    finally:
        asyncio.run(admin.aclose())


def test_dynamo_admin_plane_has_total_direct_worker_readiness_deadline():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin._timeout = 0.01

    async def slow_direct_health_check(*args, quiet=False, **kwargs):
        if quiet:
            await asyncio.sleep(0.05)

    try:
        with (
            patch("prime_rl.inference.dynamo.check_health", side_effect=slow_direct_health_check),
            patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
            patch.object(admin, "_discover", new=AsyncMock(return_value=admin.workers)),
            pytest.raises(TimeoutError, match="workers did not become ready"),
        ):
            asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
    finally:
        asyncio.run(admin.aclose())


def test_dynamo_admin_plane_bounds_slow_discovery_by_remaining_deadline():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin._timeout = 0.01
    admin._poll_interval = 0.05

    async def slow_discovery():
        await asyncio.sleep(0.05)
        return admin.workers

    try:
        with (
            patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
            patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
            patch.object(admin, "_discover", side_effect=slow_discovery),
            pytest.raises(TimeoutError, match="workers did not become ready"),
        ):
            started = time.monotonic()
            asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
        elapsed = time.monotonic() - started
    finally:
        asyncio.run(admin.aclose())

    assert elapsed < 0.04


def test_dynamo_admin_plane_bounds_retry_sleep_by_remaining_deadline():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin._timeout = 0.01
    admin._poll_interval = 0.05

    started = time.monotonic()
    try:
        with (
            patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
            patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
            patch.object(admin, "_discover", side_effect=DynamoDiscoveryPending("pending")),
            pytest.raises(TimeoutError, match="workers did not become ready"),
        ):
            asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
    finally:
        asyncio.run(admin.aclose())

    assert time.monotonic() - started < 0.04


def test_dynamo_topology_retry_sleep_uses_one_absolute_deadline():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin._timeout = 0.01
    admin._poll_interval = 0.05
    admin._fingerprint = topology_fingerprint(admin.workers)

    started = time.monotonic()
    try:
        with (
            patch.object(admin, "_discover", side_effect=httpx.ConnectError("unavailable")),
            pytest.raises(RuntimeError, match="pinned Dynamo topology"),
        ):
            asyncio.run(admin.ensure_topology_current())
    finally:
        asyncio.run(admin.aclose())

    assert time.monotonic() - started < 0.04


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
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
        patch.object(admin, "_check_admin_capabilities", new=AsyncMock()),
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
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
        patch.object(admin, "_check_admin_capabilities", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
        asyncio.run(admin.ensure_topology_current())

    assert discover.await_count == 4
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_fails_fast_on_invalid_topology_snapshot(monkeypatch):
    pinned = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="http://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )
    discover = AsyncMock(side_effect=[pinned, pinned, ValueError("worker admin host is not allowlisted"), pinned])
    monkeypatch.setattr("prime_rl.inference.dynamo.discover_dynamo_workers", discover)
    config = ClientConfig(
        base_url="http://dynamo-frontend:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
        patch.object(admin, "_check_admin_capabilities", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
        with pytest.raises(ValueError, match="allowlisted"):
            asyncio.run(admin.ensure_topology_current())

    assert discover.await_count == 3
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
            "admin_headers_from_env": {"X-Admin-Env": "ADMIN_HEADER"},
            "admin_origin_allowlist": ["https://worker-1:8120"],
        },
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B")
    workers = parse_dynamo_workers(
        snapshot(worker(1, admin_base_url="https://worker-1:8120", world_size=1)),
        "Qwen/Qwen3-0.6B",
    )

    admin._bind(workers, topology_fingerprint(workers))

    direct_headers = admin.clients[0].headers
    assert direct_headers["Authorization"] == "Bearer admin-secret"
    assert direct_headers["X-Admin-Env"] == "admin-header"
    assert "X-Data" not in direct_headers
    assert admin._frontend_clients[0].headers["Authorization"] == "Bearer data-secret"
    assert "X-Admin-Env" not in admin._frontend_clients[0].headers
    assert _discovery_headers(config)["Authorization"] == "Bearer discovery-secret"
    assert "X-Admin-Env" not in _discovery_headers(config)
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
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, "Qwen/Qwen3-0.6B", poll_interval=0)

    with (
        patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()),
        patch("prime_rl.inference.dynamo.maybe_check_has_model", new=AsyncMock()),
        patch.object(admin, "_check_admin_capabilities", new=AsyncMock()),
    ):
        asyncio.run(admin.wait_for_ready("Qwen/Qwen3-0.6B"))
        asyncio.run(admin.ensure_topology_current())

    assert discover.await_count == 4
    asyncio.run(admin.aclose())


def test_dynamo_filesystem_update_checks_topology_and_uses_collective_rpc(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    client = AsyncMock()
    admin.clients = [client]
    weight_dir = tmp_path / "step_3"

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()) as topology,
        patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock(return_value=successful_response())) as post,
        patch.object(admin, "_collective_rpc", new=AsyncMock()) as collective_rpc,
    ):
        asyncio.run(admin.update_weights(weight_dir, transport="filesystem", step=3))

    topology.assert_awaited_once_with()
    assert [call.args[1] for call in post.await_args_list] == ["/pause", "/resume"]
    collective_rpc.assert_awaited_once_with(
        client,
        method="update_weights_from_path",
        timeout=720.0,
        args=[weight_dir.as_posix()],
        expected_result_count=1,
    )
    asyncio.run(admin.aclose())


def test_dynamo_filesystem_update_blocks_all_mutation_on_topology_drift(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    client = AsyncMock()
    admin.clients = [client]

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock(side_effect=RuntimeError("topology changed"))),
        patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock()) as post,
        patch.object(admin, "_collective_rpc", new=AsyncMock()) as collective_rpc,
        pytest.raises(RuntimeError, match="topology changed"),
    ):
        asyncio.run(admin.update_weights(tmp_path / "step_3", transport="filesystem", step=3))

    post.assert_not_awaited()
    collective_rpc.assert_not_awaited()
    asyncio.run(admin.aclose())


def test_dynamo_capability_probe_requires_collective_rpc_on_every_worker():
    admin = dynamo_admin(
        worker(1, admin_base_url="http://worker-1:8120", world_size=1),
        worker(2, admin_base_url="http://worker-2:8120", world_size=1),
    )
    clients = [AsyncMock(), AsyncMock()]

    with (
        patch.object(admin, "_collective_rpc", new=AsyncMock(side_effect=[None, RuntimeError("missing route")])) as rpc,
        pytest.raises(RuntimeError, match="required vLLM development admin contract"),
    ):
        asyncio.run(admin._check_admin_capabilities(clients, admin.workers, timeout=1))

    assert rpc.await_count == 2
    asyncio.run(admin.aclose())


@pytest.mark.parametrize(
    "payload",
    [
        {"results": []},
        {"results": ["unexpected"]},
        {"results": [None, None]},
        {"other": [None]},
    ],
)
def test_dynamo_collective_rpc_rejects_invalid_result_contract(payload):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    response = successful_response()
    response.json.return_value = payload

    with (
        patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock(return_value=response)),
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


def test_dynamo_collective_rpc_accepts_nonempty_null_results():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    response = successful_response()
    response.json.return_value = {"results": [None]}

    with patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock(return_value=response)):
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


def test_dynamo_pause_callback_failure_leaves_control_terminal(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin.clients = [AsyncMock()]

    def fail_callback():
        raise RuntimeError("callback failed")

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()),
        patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock(return_value=successful_response())) as post,
        pytest.raises(RuntimeError, match="engines remain paused"),
    ):
        asyncio.run(
            admin.update_weights(
                tmp_path / "step_3",
                transport="filesystem",
                step=3,
                on_paused=fail_callback,
            )
        )

    assert admin._control_terminal is True
    assert [call.args[1] for call in post.await_args_list] == ["/pause"]
    asyncio.run(admin.aclose())


def test_dynamo_cancelled_pause_leaves_control_terminal(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))
    admin.clients = [AsyncMock()]

    async def scenario():
        entered_pause = asyncio.Event()

        async def blocking_post(client, path, **kwargs):
            entered_pause.set()
            await asyncio.Event().wait()

        with (
            patch.object(admin, "ensure_topology_current", new=AsyncMock()),
            patch("prime_rl.inference.dynamo._admin_post", new=AsyncMock(side_effect=blocking_post)),
        ):
            task = asyncio.create_task(admin.update_weights(tmp_path / "step_3", transport="filesystem", step=3))
            await entered_pause.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())

    assert admin._control_terminal is True
    asyncio.run(admin.aclose())


def test_dynamo_rejects_lora_without_admin_mutation(tmp_path):
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))

    with pytest.raises(ValueError, match="does not support LoRA"):
        asyncio.run(admin.load_lora_adapter("adapter", tmp_path))

    asyncio.run(admin.aclose())


def test_dynamo_rejects_in_memory_initialization_without_admin_mutation():
    admin = dynamo_admin(worker(1, admin_base_url="http://worker-1:8120", world_size=1))

    with pytest.raises(ValueError, match="does not support NCCL"):
        asyncio.run(
            admin.initialize_nccl(
                host="localhost",
                port=29501,
                timeout=10,
                inference_world_size=1,
            )
        )
    with pytest.raises(ValueError, match="does not support NIXL"):
        asyncio.run(
            admin.initialize_nixl(
                host="localhost",
                port=8001,
                timeout=10,
                inference_world_size=1,
                session_id="test",
            )
        )

    asyncio.run(admin.aclose())
