import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prime_rl.configs.shared import ClientConfig, ElasticConfig
from prime_rl.utils.dynamo import DynamoInferencePool, _parse_dynamo_workers

MODEL = "Qwen/Qwen3-0.6B"


def worker(**updates):
    value = {
        "component": "backend",
        "instance_id": 10,
        "model": MODEL,
        "admin_base_url": "http://decode:8120",
        "world_size": 2,
    }
    return {**value, **updates}


def payload(*workers):
    return {"protocol_version": 1, "workers": list(workers)}


def response(body):
    result = MagicMock()
    result.raise_for_status = MagicMock()
    result.json.return_value = body
    return result


def test_parse_workers_orders_identity_and_preserves_topology():
    workers = _parse_dynamo_workers(
        payload(
            worker(component="prefill", instance_id=20, admin_base_url="http://prefill:8121"),
            worker(),
        ),
        MODEL,
    )

    assert [(item.component, item.instance_id) for item in workers] == [
        ("backend", 10),
        ("prefill", 20),
    ]
    assert [item.world_size for item in workers] == [2, 2]


@pytest.mark.parametrize(
    "workers",
    [
        [],
        [worker(error="probe timed out")],
        [worker(admin_base_url=None)],
        [worker(world_size=0)],
        [worker(model="other/model")],
        [worker(), worker(instance_id=11)],
        [worker(), worker(component="prefill", instance_id=20, admin_base_url="http://decode:8120")],
    ],
)
def test_parse_workers_rejects_incomplete_or_duplicate_snapshots(workers):
    with pytest.raises(ValueError):
        _parse_dynamo_workers(payload(*workers), MODEL)


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
                worker(component="prefill", instance_id=20, admin_base_url="http://prefill:8121"),
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
    assert [item.component for item in pool.workers] == ["backend", "prefill"]
