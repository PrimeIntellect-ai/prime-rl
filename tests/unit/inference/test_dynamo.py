import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from prime_rl.configs.shared import PolicyClientConfig
from prime_rl.inference.dynamo import (
    DynamoAdminPlane,
    DynamoDiscoveryPending,
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


def snapshot(*workers: dict) -> dict:
    return {"protocol_version": 1, "workers": list(workers)}


def parsed(*workers: dict):
    return parse_dynamo_workers(snapshot(*workers), MODEL, expected_admin_host="worker")


def unbound_admin() -> DynamoAdminPlane:
    config = PolicyClientConfig(
        base_url="http://worker:8000/v1",
        skip_model_check=True,
        wait_for_ready_timeout=2,
        dynamo=dynamo_config(),
    )
    return DynamoAdminPlane(config, MODEL, poll_interval=0)


def admin_for(*workers: dict) -> DynamoAdminPlane:
    admin = unbound_admin()
    admin._workers = parsed(*workers)
    admin.clients = [AsyncMock() for _ in workers]
    return admin


def test_parse_dynamo_workers_filters_model_and_checks_admin_host():
    [discovered_worker] = parse_dynamo_workers(
        snapshot(
            worker(3),
            worker(1, model="other"),
        ),
        MODEL,
        expected_admin_host="worker",
    )

    assert discovered_worker.admin_base_url == "http://worker:8203"

    with pytest.raises(ValueError, match="does not match discovery host"):
        parse_dynamo_workers(
            snapshot(worker(1, admin_base_url="http://other-worker:8201")),
            MODEL,
            expected_admin_host="worker",
        )


def test_parse_dynamo_workers_orders_workers_by_instance_id():
    workers = parse_dynamo_workers(snapshot(worker(2), worker(1, world_size=4)), MODEL, expected_admin_host="worker")

    assert [(discovered.instance_id, discovered.world_size) for discovered in workers] == [(1, 4), (2, 1)]
    with pytest.raises(ValueError, match="duplicate worker instance ids"):
        parse_dynamo_workers(
            snapshot(worker(1), worker(1, admin_base_url="http://worker:8300")), MODEL, expected_admin_host="worker"
        )


@pytest.mark.parametrize(
    ("worker_update", "error", "match"),
    [
        ({"admin_base_url": None}, DynamoDiscoveryPending, "admin_base_url"),
        ({"world_size": 0}, ValueError, "positive world_size"),
    ],
)
def test_parse_dynamo_workers_validates_required_metadata(worker_update, error, match):
    payload = {**worker(1), **worker_update}
    with pytest.raises(error, match=match):
        parse_dynamo_workers(snapshot(payload), MODEL, expected_admin_host="worker")


def test_dynamo_admin_plane_factory_pins_two_identical_snapshots():
    discovered_worker = parsed(worker(1))
    discover = AsyncMock(side_effect=[discovered_worker, discovered_worker])
    admin = setup_admin_plane(
        PolicyClientConfig(
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
    config = PolicyClientConfig(
        base_url="http://worker:8000/v1",
        headers={"X-Static": "static"},
        headers_from_env={"X-Environment": "DYNAMO_HEADER"},
        api_key_var="DYNAMO_API_KEY",
        dynamo=dynamo_config(),
    )
    admin = DynamoAdminPlane(config, MODEL)
    client = admin._make_worker_client(parsed(worker(1))[0])

    assert client.headers["X-Static"] == "static"
    assert client.headers["X-Environment"] == "from-env"
    assert client.headers.get("Authorization") == authorization

    asyncio.run(client.aclose())
    asyncio.run(admin.aclose())


def test_dynamo_admin_plane_derives_discovery_url_from_client_port():
    admin = setup_admin_plane(
        PolicyClientConfig(
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
        PolicyClientConfig(
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
            PolicyClientConfig(
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


def test_dynamo_delegates_filesystem_weight_updates(tmp_path):
    admin = admin_for(worker(1))

    with (
        patch.object(admin, "ensure_topology_current", new=AsyncMock()) as ensure_topology,
        patch.object(AdminPlane, "update_weights", new=AsyncMock()) as update_weights,
    ):
        asyncio.run(admin.update_weights(tmp_path, step=1))

    ensure_topology.assert_awaited_once_with()
    update_weights.assert_awaited_once_with(tmp_path, step=1, on_paused=None)
    asyncio.run(admin.aclose())


class FakeEngine:
    """Admin client of one Dynamo worker; records its admin calls into a shared log.

    Like the ModelExpress worker extension, it keeps its client across repeated ``init_broadcaster`` calls under the
    same worker id and cannot install a version without one."""

    def __init__(self, url: str, log: list, world_size: int = 1, fail_install: bool = False):
        self.base_url = httpx.URL(url)
        self.log = log
        self.world_size = world_size
        self.fail_install = fail_install
        self.closed = False
        self.client_id: str | None = None
        self.inits = 0

    def restart(self):
        self.client_id = None

    async def post(self, path, *, json=None, **kwargs):
        call = path if json is None else json["method"]
        self.log.append((self.base_url.port, call, json["args"] if json else None))
        request = httpx.Request("POST", self.base_url.join(path))
        if call == "init_broadcaster" and self.client_id != json["args"][5]:
            self.client_id = json["args"][5]
            self.inits += 1
        if call == "update_weights_from_modelexpress" and (self.fail_install or self.client_id is None):
            return httpx.Response(500, request=request)
        return httpx.Response(200, request=request, json={"results": [None] * self.world_size})

    async def aclose(self):
        self.closed = True


def modelexpress_admin(log: list, *workers: dict, failing: set[int] = frozenset()):
    """A Dynamo admin plane whose discovery reports ``workers`` until the test changes ``discovered``."""
    admin = unbound_admin()
    discovered = list(workers)
    engines = {
        8200 + raw["instance_id"]: FakeEngine(
            f"http://worker:{8200 + raw['instance_id']}", log, raw["world_size"], raw["instance_id"] in failing
        )
        for raw in workers
    }

    def make_worker_client(dynamo_worker):
        port = httpx.URL(dynamo_worker.admin_base_url).port
        return engines.setdefault(port, FakeEngine(dynamo_worker.admin_base_url, log, dynamo_worker.world_size))

    admin._discover = AsyncMock(side_effect=lambda: parsed(*discovered))
    admin._make_worker_client = make_worker_client
    return admin, discovered, engines


def initialize(admin):
    asyncio.run(
        admin.initialize_modelexpress(
            "trainer", 8001, 60, inference_world_size=1, staging_buffer_bytes=None, staging_buffers_count=1
        )
    )


def update(admin, step: int):
    asyncio.run(admin.update_modelexpress_weights(version_uid=f"v{step}", step=step))


def test_dynamo_modelexpress_initializes_every_worker_under_its_instance_id_and_updates_all_of_them():
    log = []
    admin, _, engines = modelexpress_admin(log, worker(2), worker(1, world_size=2))

    with patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()):
        initialize(admin)
        init_args = {port: args for port, call, args in log if call == "init_broadcaster"}
        log.clear()
        update(admin, 1)

    session_id = admin._modelexpress.session_id
    assert init_args == {
        8201: ["trainer", 8001, 0, 2, 60, f"{session_id}-1", None, 1],
        8202: ["trainer", 8001, 0, 1, 60, f"{session_id}-2", None, 1],
    }
    # Every update re-sends the same initialization, which the engines' running clients absorb. Every worker
    # pauses before any installs, and resumes only after all installed.
    assert [call for _, call, _ in log] == (
        ["init_broadcaster"] * 2 + ["/pause"] * 2 + ["update_weights_from_modelexpress"] * 2 + ["/resume"] * 2
    )
    assert {port: args for port, call, args in log if call == "init_broadcaster"} == init_args
    assert [engine.inits for engine in engines.values()] == [1, 1]
    asyncio.run(admin.aclose())


def test_dynamo_modelexpress_updates_follow_workers_that_join_and_leave():
    log = []
    admin, discovered, engines = modelexpress_admin(log, worker(1), worker(2))

    with patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()):
        initialize(admin)
        discovered[:] = [worker(2), worker(3)]
        log.clear()
        update(admin, 1)

    assert {port for port, call, _ in log if call == "update_weights_from_modelexpress"} == {8202, 8203}
    assert all(port != 8201 for port, _, _ in log)
    assert engines[8201].closed
    assert {port: engine.inits for port, engine in engines.items()} == {8201: 1, 8202: 1, 8203: 1}
    assert [engine.base_url.port for engine in admin.clients] == [8202, 8203]
    asyncio.run(admin.aclose())


def test_dynamo_modelexpress_reinitializes_an_engine_that_restarted_behind_the_same_worker():
    log = []
    admin, _, engines = modelexpress_admin(log, worker(1), worker(2))

    with patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()):
        initialize(admin)
        update(admin, 1)
        engines[8201].restart()
        update(admin, 2)

    assert {port: engine.inits for port, engine in engines.items()} == {8201: 2, 8202: 1}
    assert {port for port, call, args in log if call == "update_weights_from_modelexpress" and args == ["v2"]} == {
        8201,
        8202,
    }
    asyncio.run(admin.aclose())


def test_dynamo_modelexpress_install_failure_keeps_workers_paused():
    log = []
    admin, _, _ = modelexpress_admin(log, worker(1), worker(2), failing={2})

    with patch("prime_rl.inference.dynamo.check_health", new=AsyncMock()):
        initialize(admin)
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(admin.update_modelexpress_weights(version_uid="v1", step=1))

    assert "/resume" not in [call for _, call, _ in log]
    asyncio.run(admin.aclose())
