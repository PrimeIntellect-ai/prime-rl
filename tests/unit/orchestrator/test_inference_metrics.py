import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

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
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="Qwen/Qwen3-4B-Instruct-2507"} 5000.0
vllm:prompt_tokens_total{engine="1",model_name="Qwen/Qwen3-4B-Instruct-2507"} 3000.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="Qwen/Qwen3-4B-Instruct-2507"} 10000.0
vllm:generation_tokens_total{engine="1",model_name="Qwen/Qwen3-4B-Instruct-2507"} 8000.0
"""

GAUGE_NAMES = InferenceMetricsCollector.GAUGE_METRICS
COUNTER_NAMES = InferenceMetricsCollector.COUNTER_METRICS


# --- parse_prometheus_text ---


def test_parse_extracts_per_engine_gauges():
    gauges, counters = parse_prometheus_text(VLLM_METRICS_RESPONSE, GAUGE_NAMES, COUNTER_NAMES)
    assert gauges[("vllm:num_requests_running", "0")] == 5.0
    assert gauges[("vllm:num_requests_running", "1")] == 3.0
    assert gauges[("vllm:num_requests_waiting", "0")] == 2.0
    assert gauges[("vllm:num_requests_waiting", "1")] == 0.0
    assert gauges[("vllm:kv_cache_usage_perc", "0")] == 0.45
    assert gauges[("vllm:kv_cache_usage_perc", "1")] == 0.32


def test_parse_extracts_per_engine_counters():
    gauges, counters = parse_prometheus_text(VLLM_METRICS_RESPONSE, GAUGE_NAMES, COUNTER_NAMES)
    assert counters[("vllm:prompt_tokens", "0")] == 5000.0
    assert counters[("vllm:prompt_tokens", "1")] == 3000.0
    assert counters[("vllm:generation_tokens", "0")] == 10000.0
    assert counters[("vllm:generation_tokens", "1")] == 8000.0


def test_parse_empty():
    gauges, counters = parse_prometheus_text("", GAUGE_NAMES, COUNTER_NAMES)
    assert gauges == {}
    assert counters == {}


def test_parse_missing_metric():
    gauges, counters = parse_prometheus_text(VLLM_METRICS_RESPONSE, {"vllm:nonexistent"}, set())
    assert gauges == {}
    assert counters == {}


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


def set_client_response(client, response_text: str):
    mock_response = AsyncMock()
    mock_response.text = response_text
    mock_response.raise_for_status = lambda: None
    client.get = AsyncMock(return_value=mock_response)


def test_collect_gauges():
    client = make_mock_client("http://server0:8000", VLLM_METRICS_RESPONSE)
    pool = FakeInferencePool(admin_clients=[client])
    collector = InferenceMetricsCollector(pool)

    asyncio.run(collector.collect())
    metrics = collector.get_metrics()
    assert metrics["inference/num_requests_running/server_0_engine_0"] == 5.0
    assert metrics["inference/num_requests_running/server_0_engine_1"] == 3.0
    assert metrics["inference/kv_cache_usage_perc/server_0_engine_0"] == 0.45


def test_collect_counter_rates():
    """Counter rates are computed between consecutive polls."""
    client = make_mock_client("http://server0:8000", VLLM_METRICS_RESPONSE)
    pool = FakeInferencePool(admin_clients=[client])
    collector = InferenceMetricsCollector(pool)

    # First poll at t=0: stores baseline, no rate yet
    with patch("prime_rl.orchestrator.inference_metrics.time") as mock_time:
        mock_time.monotonic.return_value = 0.0
        asyncio.run(collector.collect())

    metrics = collector.get_metrics()
    assert "inference/prompt_throughput_tps/server_0_engine_0" not in metrics

    # Second poll at t=5: prompt_tokens went 5000->7500 (+2500 in 5s = 500 tps)
    response_2 = VLLM_METRICS_RESPONSE.replace("5000.0", "7500.0").replace("10000.0", "15000.0")
    set_client_response(client, response_2)

    with patch("prime_rl.orchestrator.inference_metrics.time") as mock_time:
        mock_time.monotonic.return_value = 5.0
        asyncio.run(collector.collect())

    metrics = collector.get_metrics()
    assert metrics["inference/prompt_throughput_tps/server_0_engine_0"] == 500.0  # 2500/5
    assert metrics["inference/generation_throughput_tps/server_0_engine_0"] == 1000.0  # 5000/5


def test_collect_clears_stale_servers():
    client0 = make_mock_client("http://server0:8000", VLLM_METRICS_RESPONSE)
    client1 = make_mock_client("http://server1:8000", VLLM_METRICS_RESPONSE)
    pool = FakeInferencePool(admin_clients=[client0, client1])
    collector = InferenceMetricsCollector(pool)

    asyncio.run(collector.collect())
    assert any("server_1" in k for k in collector.get_metrics())

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
