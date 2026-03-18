import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import httpx

from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector, parse_prometheus_text

# Real vLLM /metrics response (trimmed to the metrics we track), from a server with 2 DP engines.
VLLM_METRICS_RESPONSE = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Qwen/Qwen3-4B-Instruct-2507"} 5.0
vllm:num_requests_running{engine="1",model_name="Qwen/Qwen3-4B-Instruct-2507"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="Qwen/Qwen3-4B-Instruct-2507"} 2.0
vllm:num_requests_waiting{engine="1",model_name="Qwen/Qwen3-4B-Instruct-2507"} 0.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen3-4B-Instruct-2507"} 0.45
vllm:kv_cache_usage_perc{engine="1",model_name="Qwen/Qwen3-4B-Instruct-2507"} 0.32
"""

METRIC_NAMES = InferenceMetricsCollector.METRICS


# --- parse_prometheus_text ---


def test_parse_extracts_per_engine_gauges():
    result = parse_prometheus_text(VLLM_METRICS_RESPONSE, METRIC_NAMES)
    assert result[("vllm:num_requests_running", "0")] == 5.0
    assert result[("vllm:num_requests_running", "1")] == 3.0
    assert result[("vllm:num_requests_waiting", "0")] == 2.0
    assert result[("vllm:num_requests_waiting", "1")] == 0.0
    assert result[("vllm:kv_cache_usage_perc", "0")] == 0.45
    assert result[("vllm:kv_cache_usage_perc", "1")] == 0.32


def test_parse_ignores_counters():
    text = """\
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="Qwen/Qwen3-4B-Instruct-2507"} 1000.0
"""
    result = parse_prometheus_text(text, {"vllm:prompt_tokens_total"})
    assert result == {}


def test_parse_empty():
    assert parse_prometheus_text("", METRIC_NAMES) == {}


def test_parse_missing_metric():
    assert parse_prometheus_text(VLLM_METRICS_RESPONSE, {"vllm:nonexistent"}) == {}


# --- InferenceMetricsCollector ---


@dataclass
class FakeInferencePool:
    admin_clients: list[httpx.AsyncClient] = field(default_factory=list)


def make_mock_client(base_url: str, response_text: str) -> httpx.AsyncClient:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = httpx.URL(base_url)
    mock_response = AsyncMock()
    mock_response.text = response_text
    mock_response.raise_for_status = lambda: None
    client.get = AsyncMock(return_value=mock_response)
    return client


def test_collect_single_server():
    client = make_mock_client("http://server0:8000", VLLM_METRICS_RESPONSE)
    pool = FakeInferencePool(admin_clients=[client])
    collector = InferenceMetricsCollector(pool)

    asyncio.run(collector.collect())
    metrics = collector.get_metrics()
    assert metrics["inference/num_requests_running/server_0_engine_0"] == 5.0
    assert metrics["inference/num_requests_running/server_0_engine_1"] == 3.0
    assert metrics["inference/kv_cache_usage_perc/server_0_engine_0"] == 0.45
    assert metrics["inference/kv_cache_usage_perc/server_0_engine_1"] == 0.32


def test_collect_two_servers():
    server1_metrics = VLLM_METRICS_RESPONSE.replace("5.0\n", "10.0\n").replace("3.0\n", "7.0\n")
    pool = FakeInferencePool(
        admin_clients=[
            make_mock_client("http://server0:8000", VLLM_METRICS_RESPONSE),
            make_mock_client("http://server1:8000", server1_metrics),
        ]
    )
    collector = InferenceMetricsCollector(pool)

    asyncio.run(collector.collect())
    metrics = collector.get_metrics()
    assert metrics["inference/num_requests_running/server_0_engine_0"] == 5.0
    assert metrics["inference/num_requests_running/server_0_engine_1"] == 3.0
    assert metrics["inference/num_requests_running/server_1_engine_0"] == 10.0
    assert metrics["inference/num_requests_running/server_1_engine_1"] == 7.0


def test_collect_running_average():
    """Values are averaged over multiple collects."""
    response_a = VLLM_METRICS_RESPONSE  # engine 0 running = 5.0
    response_b = VLLM_METRICS_RESPONSE.replace("5.0\n", "15.0\n")  # engine 0 running = 15.0

    client = make_mock_client("http://server0:8000", response_a)
    pool = FakeInferencePool(admin_clients=[client])
    collector = InferenceMetricsCollector(pool, window_size=20)

    asyncio.run(collector.collect())

    # Switch to response_b
    mock_response_b = AsyncMock()
    mock_response_b.text = response_b
    mock_response_b.raise_for_status = lambda: None
    client.get = AsyncMock(return_value=mock_response_b)

    asyncio.run(collector.collect())

    metrics = collector.get_metrics()
    assert metrics["inference/num_requests_running/server_0_engine_0"] == 10.0  # avg(5, 15)


def test_collect_window_size_limit():
    """Old values are dropped when window is full."""
    pool = FakeInferencePool(admin_clients=[make_mock_client("http://server0:8000", VLLM_METRICS_RESPONSE)])
    collector = InferenceMetricsCollector(pool, window_size=2)

    # Collect twice with value 5.0
    asyncio.run(collector.collect())
    asyncio.run(collector.collect())

    # Switch to value 15.0
    response_new = VLLM_METRICS_RESPONSE.replace("5.0\n", "15.0\n")
    mock_response = AsyncMock()
    mock_response.text = response_new
    mock_response.raise_for_status = lambda: None
    pool.admin_clients[0].get = AsyncMock(return_value=mock_response)

    asyncio.run(collector.collect())  # window now [5.0, 15.0] (oldest 5.0 dropped)
    asyncio.run(collector.collect())  # window now [15.0, 15.0]

    metrics = collector.get_metrics()
    assert metrics["inference/num_requests_running/server_0_engine_0"] == 15.0


def test_collect_clears_stale_servers():
    client0 = make_mock_client("http://server0:8000", VLLM_METRICS_RESPONSE)
    client1 = make_mock_client("http://server1:8000", VLLM_METRICS_RESPONSE)
    pool = FakeInferencePool(admin_clients=[client0, client1])
    collector = InferenceMetricsCollector(pool)

    asyncio.run(collector.collect())
    assert any("server_1" in k for k in collector.get_metrics())

    # Remove server1 from pool
    pool.admin_clients = [client0]
    asyncio.run(collector.collect())
    assert not any("server_1" in k for k in collector.get_metrics())


def test_collect_handles_server_failure():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = httpx.URL("http://server0:8000")
    client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    pool = FakeInferencePool(admin_clients=[client])

    collector = InferenceMetricsCollector(pool)
    asyncio.run(collector.collect())
    assert collector.get_metrics() == {}
