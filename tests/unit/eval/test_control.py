import pytest
from fastapi.testclient import TestClient

from prime_rl.configs.orchestrator import ConcurrencyConfig
from prime_rl.eval.control import create_app
from prime_rl.orchestrator.concurrency import ConcurrencyController, EngineLoadSample

INFLIGHT = 1800
PREEMPTED = EngineLoadSample(
    engine_id="engine-0",
    role=None,
    kv_capacity_tokens=1_000_000,
    max_model_len=8192,
    kv_usage=0.95,
    running=INFLIGHT,
    waiting=0,
    waiting_capacity=0,
    preemptions_delta=4,
)


@pytest.fixture
def limits():
    return []


@pytest.fixture
def concurrency(limits):
    config = ConcurrencyConfig(initial_inflight=2000, min_inflight=1, max_inflight=2000)
    concurrency = ConcurrencyController(config, fallback_cost=8192)
    concurrency.bind(set_limit=limits.append, get_inflight=lambda: INFLIGHT)
    return concurrency


@pytest.fixture
def client(concurrency):
    return TestClient(create_app(concurrency))


@pytest.mark.parametrize(
    ("targets", "expected", "expected_limits"),
    [
        ([1500], {"target": 1500, "cap": 1500, "inflight": INFLIGHT}, [1500]),
        ([2500], {"target": 2500, "cap": 2500, "inflight": INFLIGHT}, [2500]),
        ([1500, None], {"target": None, "cap": 1500, "inflight": INFLIGHT}, [1500]),
    ],
    ids=["below_band", "above_band", "back_to_adaptive"],
)
def test_set_concurrency(client, limits, targets, expected, expected_limits):
    for target in targets:
        response = client.put("/v1/concurrency", json={"target": target})

    assert (response.status_code, response.json(), limits) == (200, expected, expected_limits), response.text


def test_get_concurrency(client):
    response = client.get("/v1/concurrency")

    assert (response.status_code, response.json()) == (200, {"target": None, "cap": 2000, "inflight": INFLIGHT})


@pytest.mark.parametrize("body", [{"target": 0}, {}], ids=["zero", "missing"])
def test_set_concurrency_invalid(client, body):
    response = client.put("/v1/concurrency", json=body)

    assert response.status_code == 422, response.text


@pytest.mark.parametrize(("target", "expected_cap"), [(1500, 1500), (None, 1440)], ids=["pinned", "adaptive"])
def test_target_holds_under_overload(concurrency, target, expected_cap):
    if target is not None:
        concurrency.set_target(target)

    concurrency.observe([PREEMPTED])

    assert concurrency.max_inflight == expected_cap
