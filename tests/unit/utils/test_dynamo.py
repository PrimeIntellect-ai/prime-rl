import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prime_rl.configs.shared import ClientConfig, ElasticConfig
from prime_rl.utils.client import _rank_offsets
from prime_rl.utils.dynamo import DynamoInferencePool, _parse_dynamo_workers

MODEL = "Qwen/Qwen3-0.6B"


def worker(**updates):
    value = {
        "component": "backend",
        "instance_id": 10,
        "model": MODEL,
        "admin_base_url": "http://decode:8120",
        "system_url": "http://decode:8181",
        "world_size": 2,
        "system_routes": ["update/load_lora"],
    }
    return {**value, **updates}


def payload(*workers):
    return {"protocol_version": 1, "namespace": "training", "workers": list(workers)}


def response(body):
    result = MagicMock()
    result.raise_for_status = MagicMock()
    result.json.return_value = body
    return result


def pool_with_clients(*, admin, system=None, frontend=None):
    pool = DynamoInferencePool.__new__(DynamoInferencePool)
    pool._admin_clients = [admin]
    pool._lora_update_clients = [] if system is None else [system]
    pool._frontend_model_clients = [] if frontend is None else [frontend]
    pool._nccl_initialized = False
    pool._skip_model_check = False
    pool._wait_for_ready_timeout = 1
    pool._readiness_deadline = None
    pool._scorer = MagicMock()
    pool._scorer.aclose = AsyncMock()
    return pool


def test_parse_workers_orders_identity_and_preserves_topology():
    workers = _parse_dynamo_workers(
        payload(
            worker(
                component="prefill",
                instance_id=20,
                admin_base_url="http://prefill:8121",
                system_url="http://prefill:8182",
            ),
            worker(),
        ),
        MODEL,
    )

    assert [(item.component, item.instance_id) for item in workers] == [
        ("backend", 10),
        ("prefill", 20),
    ]
    assert [item.world_size for item in workers] == [2, 2]
    assert all("update/load_lora" in item.system_routes for item in workers)


def test_parse_workers_ignores_workers_for_other_models():
    workers = _parse_dynamo_workers(
        payload(
            worker(model="other/model", error="other model is still starting"),
            worker(),
        ),
        MODEL,
    )

    assert [(item.component, item.instance_id) for item in workers] == [("backend", 10)]


@pytest.mark.parametrize(
    "workers",
    [
        [],
        [worker(error="probe timed out")],
        [worker(admin_base_url=None)],
        [worker(world_size=0)],
        [worker(model="other/model")],
        [worker(), worker(instance_id=11)],
        [worker(), worker(admin_base_url="http://other:8120")],
        [worker(), worker(component="prefill", instance_id=20, system_url=None)],
    ],
)
def test_parse_workers_rejects_incomplete_or_duplicate_snapshots(workers):
    with pytest.raises(ValueError):
        _parse_dynamo_workers(payload(*workers), MODEL)


@pytest.mark.parametrize(
    "admin_base_url",
    [
        "https://decode:8120",
        "http://user:password@decode:8120",
        "http://decode:8120/admin",
        "http://decode:8120?target=other",
    ],
)
def test_parse_workers_rejects_unsafe_admin_urls(admin_base_url):
    with pytest.raises(ValueError, match="admin_base_url"):
        _parse_dynamo_workers(payload(worker(admin_base_url=admin_base_url)), MODEL)


@pytest.mark.parametrize("protocol_version", [None, 0, 2, "1", True, 1.0])
def test_parse_workers_rejects_unknown_protocol_version(protocol_version):
    discovery = payload(worker())
    if protocol_version is None:
        discovery.pop("protocol_version")
    else:
        discovery["protocol_version"] = protocol_version

    with pytest.raises(ValueError, match="protocol version"):
        _parse_dynamo_workers(discovery, MODEL)


@pytest.mark.parametrize("discovery", [None, [], "not-an-object", 1, True])
def test_parse_workers_rejects_non_object_snapshot(discovery):
    with pytest.raises(ValueError, match="workers list"):
        _parse_dynamo_workers(discovery, MODEL)


def test_rank_offsets_support_heterogeneous_managed_world_sizes():
    assert _rank_offsets([1, 2], inference_world_size=3) == [0, 1]
    assert _rank_offsets([2, 2, 2, 2], inference_world_size=8) == [0, 2, 4, 6]
    with pytest.raises(ValueError):
        _rank_offsets([2, 2], inference_world_size=3)


@pytest.mark.parametrize(
    "conflict",
    [
        {"admin_base_url": ["http://worker:8120"]},
        {"elastic": ElasticConfig(hostname="workers")},
    ],
)
def test_discovery_config_rejects_other_pool_modes(conflict):
    with pytest.raises(ValueError, match="dynamo_discovery_url"):
        ClientConfig(dynamo_discovery_url="http://frontend:8001", **conflict)


def test_discovery_retries_until_expected_world_size_is_complete():
    transient = MagicMock()
    transient.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Service unavailable",
        request=httpx.Request("GET", "http://frontend:8001/v1/rl/workers"),
        response=httpx.Response(503),
    )
    discovery_client = AsyncMock()
    discovery_client.get.side_effect = [
        transient,
        response(payload(worker())),
        response(
            payload(
                worker(),
                worker(
                    component="prefill",
                    instance_id=20,
                    admin_base_url="http://prefill:8121",
                    system_url="http://prefill:8182",
                ),
            )
        ),
    ]
    context = AsyncMock()
    context.__aenter__.return_value = discovery_client

    class DiscoveryOnlyPool(DynamoInferencePool):
        def __init__(self, _config, workers, **_kwargs):
            self.workers = workers

    with patch("prime_rl.utils.dynamo.AsyncClient", return_value=context):
        pool = asyncio.run(
            DiscoveryOnlyPool.from_config(
                ClientConfig(
                    base_url=["http://frontend:8000/v1"],
                    dynamo_discovery_url="http://frontend:8001",
                    wait_for_ready_timeout=1,
                ),
                model_name=MODEL,
                expected_inference_world_size=4,
            )
        )

    assert discovery_client.get.await_count == 3
    assert all(0 < call.kwargs["timeout"].connect <= 1 for call in discovery_client.get.await_args_list)
    assert [item.component for item in pool.workers] == ["backend", "prefill"]


def test_discovery_requires_expected_world_size():
    with pytest.raises(ValueError, match="expected_inference_world_size"):
        asyncio.run(
            DynamoInferencePool.from_config(
                ClientConfig(
                    base_url=["http://frontend:8000/v1"],
                    dynamo_discovery_url="http://frontend:8001",
                ),
                model_name=MODEL,
            )
        )


def test_discovered_control_clients_do_not_receive_frontend_credentials(monkeypatch):
    monkeypatch.setenv("PRIME_TEST_API_KEY", "secret")
    config = ClientConfig(
        base_url=["http://frontend:8000/v1"],
        dynamo_discovery_url="http://frontend:8001",
        api_key_var="PRIME_TEST_API_KEY",
        headers={"X-Frontend-Secret": "secret"},
    )
    pool = DynamoInferencePool(config, _parse_dynamo_workers(payload(worker()), MODEL), model_name=MODEL)

    assert "authorization" not in pool.admin_clients[0].headers
    assert "x-frontend-secret" not in pool.admin_clients[0].headers

    asyncio.run(pool.stop())


def test_wait_for_ready_retries_frontend_model_publication():
    admin = AsyncMock()
    frontend = AsyncMock()
    admin.get.side_effect = lambda path: response({"data": [{"id": MODEL}]})
    frontend.get.side_effect = [
        response({"data": []}),
        response({"data": [{"id": MODEL}]}),
    ]
    pool = pool_with_clients(admin=admin, frontend=frontend)

    asyncio.run(pool.wait_for_ready(MODEL))

    assert [call.args[0] for call in admin.get.await_args_list] == ["/health", "/v1/models"]
    assert [call.args[0] for call in frontend.get.await_args_list] == ["/v1/models", "/v1/models"]
    assert all(0 < call.kwargs["timeout"].connect <= 1 for call in frontend.get.await_args_list)


def test_wait_for_ready_clears_expired_discovery_deadline_before_retry():
    admin = AsyncMock()
    frontend = AsyncMock()
    admin.get.side_effect = lambda path: response({"data": [{"id": MODEL}]})
    frontend.get.return_value = response({"data": [{"id": MODEL}]})
    pool = pool_with_clients(admin=admin, frontend=frontend)
    pool._readiness_deadline = 0.0

    with pytest.raises(TimeoutError):
        asyncio.run(pool.wait_for_ready(MODEL))

    assert pool._readiness_deadline is None
    asyncio.run(pool.wait_for_ready(MODEL))


def test_lora_update_uses_system_route_and_resumes_after_publication():
    admin = AsyncMock()
    system = AsyncMock()
    frontend = AsyncMock()
    admin.post.return_value = response({"status": "ok"})
    system.post.return_value = response({"status": "ok"})
    frontend.get.return_value = response({"data": [{"id": "math-r8"}]})
    pool = pool_with_clients(admin=admin, system=system, frontend=frontend)

    asyncio.run(pool.update_weights(Path("/shared/adapter/step_1"), lora_name="math-r8", step=1))

    system.post.assert_awaited_once_with(
        "/v1/loras",
        json={
            "lora_name": "math-r8",
            "source": {"uri": "file:///shared/adapter/step_1"},
            "load_inplace": True,
        },
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )
    assert [call.args[0] for call in admin.post.await_args_list] == ["/pause", "/resume"]
    frontend.get.assert_awaited_once()
    assert frontend.get.await_args.args == ("/v1/models",)
    assert 0 < frontend.get.await_args.kwargs["timeout"].connect <= 1


def test_lora_update_resumes_after_failure():
    admin = AsyncMock()
    system = AsyncMock()
    admin.post.return_value = response({"status": "ok"})
    system.post.side_effect = RuntimeError("LoRA update failed")
    pool = pool_with_clients(admin=admin, system=system)

    with pytest.raises(RuntimeError, match="LoRA update failed"):
        asyncio.run(pool.update_weights(Path("/shared/adapter"), lora_name="math-r8", step=1))

    assert [call.args[0] for call in admin.post.await_args_list] == ["/pause", "/resume"]


def test_full_weight_update_uses_native_collective_rpc(tmp_path):
    admin = AsyncMock()
    admin.post.return_value = response({"status": "ok"})
    pool = pool_with_clients(admin=admin)
    pool._nccl_initialized = True

    asyncio.run(pool.update_weights(tmp_path, step=1))

    assert [call.args[0] for call in admin.post.await_args_list] == [
        "/pause",
        "/collective_rpc",
        "/resume",
    ]
    assert admin.post.await_args_list[1].kwargs["json"] == {
        "method": "update_weights_from_path",
        "args": [str(tmp_path)],
    }


def test_full_weight_update_requires_initialized_nccl(tmp_path):
    admin = AsyncMock()
    pool = pool_with_clients(admin=admin)

    with pytest.raises(RuntimeError, match="init_nccl_broadcast"):
        asyncio.run(pool.update_weights(tmp_path, step=1))

    admin.post.assert_not_awaited()


def test_stop_closes_every_client_owned_by_dynamo_pool():
    admin = AsyncMock()
    system = AsyncMock()
    frontend = AsyncMock()
    pool = pool_with_clients(admin=admin, system=system, frontend=frontend)

    asyncio.run(pool.stop())

    pool._scorer.aclose.assert_awaited_once()
    admin.aclose.assert_awaited_once()
    system.aclose.assert_awaited_once()
    frontend.aclose.assert_awaited_once()
