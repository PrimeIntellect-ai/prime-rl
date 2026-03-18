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
