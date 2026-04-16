import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector, parse_prometheus_text

# Realistic vLLM Prometheus output
PROMETHEUS_TEXT_SERVER_1 = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="model"} 5.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="model"} 3.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="model"} 0.85
# HELP vllm:gpu_prefix_cache_hit_rate GPU prefix cache hit rate.
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate{model_name="model"} 0.60
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="model"} 50000.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="model"} 30000.0
# HELP vllm:request_success_total Number of successfully completed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="model"} 100.0
# HELP vllm:nixl_xfer_time_seconds Histogram of NIXL transfer times.
# TYPE vllm:nixl_xfer_time_seconds histogram
vllm:nixl_xfer_time_seconds_bucket{le="0.001"} 10
vllm:nixl_xfer_time_seconds_bucket{le="0.01"} 50
vllm:nixl_xfer_time_seconds_bucket{le="+Inf"} 100
vllm:nixl_xfer_time_seconds_sum 0.5
vllm:nixl_xfer_time_seconds_count 100
"""

PROMETHEUS_TEXT_SERVER_2 = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="model"} 8.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="model"} 1.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="model"} 0.40
# HELP vllm:gpu_prefix_cache_hit_rate GPU prefix cache hit rate.
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate{model_name="model"} 0.90
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="model"} 70000.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="model"} 40000.0
# HELP vllm:request_success_total Number of successfully completed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{model_name="model"} 150.0
# HELP vllm:nixl_xfer_time_seconds Histogram of NIXL transfer times.
# TYPE vllm:nixl_xfer_time_seconds histogram
vllm:nixl_xfer_time_seconds_bucket{le="0.001"} 20
vllm:nixl_xfer_time_seconds_bucket{le="0.01"} 80
vllm:nixl_xfer_time_seconds_bucket{le="+Inf"} 200
vllm:nixl_xfer_time_seconds_sum 1.0
vllm:nixl_xfer_time_seconds_count 200
"""


def test_parse_prometheus_text_gauges():
    gauges, _, _ = parse_prometheus_text(PROMETHEUS_TEXT_SERVER_1)
    assert gauges["vllm:num_requests_running"] == 5.0
    assert gauges["vllm:num_requests_waiting"] == 3.0
    assert gauges["vllm:gpu_cache_usage_perc"] == 0.85
    assert gauges["vllm:gpu_prefix_cache_hit_rate"] == 0.60


def test_parse_prometheus_text_counters():
    _, counters, _ = parse_prometheus_text(PROMETHEUS_TEXT_SERVER_1)
    assert counters["vllm:prompt_tokens"] == 50000.0
    assert counters["vllm:generation_tokens"] == 30000.0
    assert counters["vllm:request_success"] == 100.0


def test_parse_prometheus_text_histograms():
    _, _, histograms = parse_prometheus_text(PROMETHEUS_TEXT_SERVER_1)
    h_sum, h_count = histograms["vllm:nixl_xfer_time_seconds"]
    assert h_sum == 0.5
    assert h_count == 100


def test_parse_ignores_unknown_metrics():
    text = """\
# HELP some_other_metric A metric we don't care about.
# TYPE some_other_metric gauge
some_other_metric{label="foo"} 42.0
"""
    gauges, counters, histograms = parse_prometheus_text(text)
    assert gauges == {}
    assert counters == {}
    assert histograms == {}


def _make_mock_client(response_text):
    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = "http://test:8000"
    response = MagicMock()
    response.text = response_text
    response.raise_for_status = MagicMock()
    client.get = AsyncMock(return_value=response)
    return client


def _make_failing_client():
    client = AsyncMock(spec=httpx.AsyncClient)
    client.base_url = "http://dead:8000"
    client.get = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    return client


def test_collect_aggregates_sum_gauges():
    """Sum gauges (requests running/waiting) are summed across servers."""
    clients = [_make_mock_client(PROMETHEUS_TEXT_SERVER_1), _make_mock_client(PROMETHEUS_TEXT_SERVER_2)]
    collector = InferenceMetricsCollector(clients)

    async def run():
        with patch("prime_rl.orchestrator.inference_metrics.wandb"):
            await collector._collect_and_log()

    asyncio.run(run())

    assert collector._gauge_history["num_requests_running"][0] == 13.0  # 5 + 8
    assert collector._gauge_history["num_requests_waiting"][0] == 4.0  # 3 + 1


def test_collect_aggregates_dual_gauges():
    """KV cache metrics produce both max and mean across servers."""
    clients = [_make_mock_client(PROMETHEUS_TEXT_SERVER_1), _make_mock_client(PROMETHEUS_TEXT_SERVER_2)]
    collector = InferenceMetricsCollector(clients)

    async def run():
        with patch("prime_rl.orchestrator.inference_metrics.wandb"):
            await collector._collect_and_log()

    asyncio.run(run())

    # gpu_cache_usage_perc: server1=0.85, server2=0.40
    assert collector._gauge_history["gpu_cache_usage_perc_max"][0] == 0.85
    assert collector._gauge_history["gpu_cache_usage_perc_mean"][0] == pytest.approx(0.625)
    # gpu_prefix_cache_hit_rate: server1=0.60, server2=0.90
    assert collector._gauge_history["gpu_prefix_cache_hit_rate_max"][0] == 0.90
    assert collector._gauge_history["gpu_prefix_cache_hit_rate_mean"][0] == pytest.approx(0.75)


def test_collect_computes_counter_rates():
    """Counter metrics are converted to rates after two polls."""
    client = _make_mock_client(PROMETHEUS_TEXT_SERVER_1)
    collector = InferenceMetricsCollector([client])

    async def run():
        with patch("prime_rl.orchestrator.inference_metrics.wandb"):
            await collector._collect_and_log()
            assert len(collector._rate_history) == 0

            # Update counters (simulating time passing)
            updated_text = PROMETHEUS_TEXT_SERVER_1.replace("50000.0", "55000.0")
            updated_text = updated_text.replace("30000.0", "32000.0")
            updated_text = updated_text.replace(
                'vllm:request_success_total{model_name="model"} 100.0',
                'vllm:request_success_total{model_name="model"} 110.0',
            )
            response = MagicMock()
            response.text = updated_text
            response.raise_for_status = MagicMock()
            client.get = AsyncMock(return_value=response)

            await collector._collect_and_log()

    asyncio.run(run())

    assert collector._rate_history["prefill_throughput_tps"][0] > 0
    assert collector._rate_history["decode_throughput_tps"][0] > 0
    assert collector._rate_history["completed_requests_per_s"][0] > 0


def test_collect_skips_counter_reset():
    """Counter resets (server restart) should be skipped."""
    client = _make_mock_client(PROMETHEUS_TEXT_SERVER_1)
    collector = InferenceMetricsCollector([client])

    async def run():
        with patch("prime_rl.orchestrator.inference_metrics.wandb"):
            await collector._collect_and_log()

            # Simulate counter reset — lower values
            reset_text = PROMETHEUS_TEXT_SERVER_1.replace("50000.0", "100.0").replace("30000.0", "50.0")
            response = MagicMock()
            response.text = reset_text
            response.raise_for_status = MagicMock()
            client.get = AsyncMock(return_value=response)

            await collector._collect_and_log()

    asyncio.run(run())

    assert len(collector._rate_history.get("prefill_throughput_tps", [])) == 0
    assert len(collector._rate_history.get("decode_throughput_tps", [])) == 0


def test_collect_computes_histogram_avg_latency():
    """Histogram metrics are converted to average latency in ms."""
    client = _make_mock_client(PROMETHEUS_TEXT_SERVER_1)
    collector = InferenceMetricsCollector([client])

    async def run():
        with patch("prime_rl.orchestrator.inference_metrics.wandb"):
            await collector._collect_and_log()

            updated_text = PROMETHEUS_TEXT_SERVER_1.replace(
                "vllm:nixl_xfer_time_seconds_sum 0.5", "vllm:nixl_xfer_time_seconds_sum 1.0"
            ).replace("vllm:nixl_xfer_time_seconds_count 100", "vllm:nixl_xfer_time_seconds_count 200")
            response = MagicMock()
            response.text = updated_text
            response.raise_for_status = MagicMock()
            client.get = AsyncMock(return_value=response)

            await collector._collect_and_log()

    asyncio.run(run())

    # avg latency = (1.0 - 0.5) / (200 - 100) * 1000 = 5.0 ms
    assert collector._rate_history["nixl_xfer_time_seconds_avg_ms"][0] == pytest.approx(5.0)


def test_collect_handles_failing_server():
    """If one server fails, metrics from the other still get collected."""
    collector = InferenceMetricsCollector([_make_mock_client(PROMETHEUS_TEXT_SERVER_1), _make_failing_client()])

    async def run():
        with patch("prime_rl.orchestrator.inference_metrics.wandb"):
            await collector._collect_and_log()

    asyncio.run(run())

    assert collector._gauge_history["num_requests_running"][0] == 5.0


def test_collect_noop_when_all_servers_fail():
    """When all servers fail, no metrics are recorded."""
    collector = InferenceMetricsCollector([_make_failing_client(), _make_failing_client()])

    async def run():
        with patch("prime_rl.orchestrator.inference_metrics.wandb"):
            await collector._collect_and_log()

    asyncio.run(run())

    assert len(collector._gauge_history) == 0


def test_collect_logs_to_wandb_with_timestamp():
    """Metrics are logged to wandb with an inference_wall_time key."""
    clients = [_make_mock_client(PROMETHEUS_TEXT_SERVER_1), _make_mock_client(PROMETHEUS_TEXT_SERVER_2)]
    collector = InferenceMetricsCollector(clients)

    async def run():
        with patch("prime_rl.orchestrator.inference_metrics.wandb") as mock_wandb:
            await collector._collect_and_log()
        return mock_wandb

    mock_wandb = asyncio.run(run())

    mock_wandb.log.assert_called_once()
    logged = mock_wandb.log.call_args[0][0]
    assert "inference_wall_time" in logged
    assert "inference/num_requests_running" in logged
    assert "inference/gpu_cache_usage_perc_max" in logged
    assert "inference/gpu_cache_usage_perc_mean" in logged


def test_sliding_window_smoothing():
    """Gauge values are smoothed via sliding window average."""
    client = _make_mock_client(PROMETHEUS_TEXT_SERVER_1)
    collector = InferenceMetricsCollector([client])

    async def run():
        with patch("prime_rl.orchestrator.inference_metrics.wandb") as mock_wandb:
            await collector._collect_and_log()

            updated_text = PROMETHEUS_TEXT_SERVER_1.replace(
                'vllm:num_requests_running{model_name="model"} 5.0',
                'vllm:num_requests_running{model_name="model"} 15.0',
            )
            response = MagicMock()
            response.text = updated_text
            response.raise_for_status = MagicMock()
            client.get = AsyncMock(return_value=response)

            await collector._collect_and_log()
        return mock_wandb

    mock_wandb = asyncio.run(run())

    assert len(collector._gauge_history["num_requests_running"]) == 2
    last_call = mock_wandb.log.call_args_list[-1][0][0]
    assert last_call["inference/num_requests_running"] == pytest.approx(10.0)  # (5 + 15) / 2
