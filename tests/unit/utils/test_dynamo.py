import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prime_rl.configs.shared import ClientConfig, ElasticConfig
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.dynamo import DynamoDiscoveryPending, DynamoInferencePool, _parse_dynamo_workers

MODEL = "Qwen/Qwen3-0.6B"


def worker(**updates):
    value = {
        "component": "backend",
        "instance_id": 10,
        "model": MODEL,
        "admin_base_url": "http://worker:8120",
        "world_size": 1,
    }
    return {**value, **updates}


def payload(*workers):
    return {"protocol_version": 1, "workers": list(workers)}


def response(body):
    result = MagicMock()
    result.raise_for_status = MagicMock()
    result.json.return_value = body
    return result


def test_parse_workers_accepts_complete_snapshot_and_rejects_duplicates():
    workers = _parse_dynamo_workers(payload(worker()), MODEL)
    assert [(item.admin_base_url, item.world_size) for item in workers] == [("http://worker:8120", 1)]

    with pytest.raises(ValueError, match="duplicate"):
        _parse_dynamo_workers(payload(worker(), worker(instance_id=11)), MODEL)


def test_parse_workers_treats_missing_model_as_pending():
    with pytest.raises(DynamoDiscoveryPending):
        _parse_dynamo_workers(payload(worker(model=None)), MODEL)


@pytest.mark.parametrize(
    "conflict",
    [
        {"admin_base_url": ["http://worker:8120"]},
        {"elastic": ElasticConfig(hostname="workers")},
    ],
)
def test_discovery_config_rejects_other_pool_modes(conflict):
    with pytest.raises(ValueError, match="dynamo_discovery_url"):
        ClientConfig(dynamo_discovery_url="http://frontend:8001", dynamo_expected_world_size=1, **conflict)


def test_discovery_config_requires_expected_world_size():
    with pytest.raises(ValueError, match="dynamo_expected_world_size"):
        ClientConfig(dynamo_discovery_url="http://frontend:8001")


def test_discovery_retries_until_expected_world_size_is_complete():
    discovery_client = AsyncMock()
    discovery_client.get.side_effect = [
        response(payload()),
        response(payload(worker())),
    ]

    class DiscoveryOnlyPool(DynamoInferencePool):
        def __init__(self, _config, workers, **_kwargs):
            self.workers = workers

    with patch("prime_rl.utils.dynamo.setup_admin_clients", return_value=[discovery_client]):
        pool = asyncio.run(
            DiscoveryOnlyPool.from_config(
                ClientConfig(
                    base_url=["http://frontend:8000/v1"],
                    dynamo_discovery_url="http://frontend:8001",
                    dynamo_expected_world_size=1,
                    wait_for_ready_timeout=1,
                ),
                model_name=MODEL,
            )
        )

    assert discovery_client.get.await_count == 2
    assert pool.workers[0].admin_base_url == "http://worker:8120"


def test_setup_inference_pool_selects_dynamo_pool():
    config = ClientConfig(
        base_url=["http://frontend:8000/v1"],
        dynamo_discovery_url="http://frontend:8001",
        dynamo_expected_world_size=1,
    )
    expected = MagicMock()
    with patch("prime_rl.utils.dynamo.DynamoInferencePool.from_config", new=AsyncMock(return_value=expected)):
        pool = asyncio.run(setup_inference_pool(config, model_name=MODEL))

    assert pool is expected


def test_discovered_admin_clients_do_not_inherit_frontend_credentials(monkeypatch):
    monkeypatch.setenv("PRIME_API_KEY", "frontend-token")
    config = ClientConfig(
        base_url=["http://frontend:8000/v1"],
        api_key_var="PRIME_API_KEY",
        headers={"X-Frontend-Secret": "secret"},
        dynamo_discovery_url="http://frontend:8001",
        dynamo_expected_world_size=1,
    )
    pool = DynamoInferencePool(config, _parse_dynamo_workers(payload(worker()), MODEL), model_name=MODEL)

    assert "X-Frontend-Secret" not in pool.admin_clients[0].headers
    assert "Authorization" not in pool.admin_clients[0].headers
    assert pool._router_clients[0].headers["X-Frontend-Secret"] == "secret"
    assert pool._router_clients[0].headers["Authorization"] == "Bearer frontend-token"

    asyncio.run(pool.stop())
