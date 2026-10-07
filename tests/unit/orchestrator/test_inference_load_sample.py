from prime_rl.orchestrator.inference_metrics import (
    EngineSample,
    InferenceMetricsCollector,
    MetricsEndpoint,
    TimedSnapshot,
    parse_prometheus_text,
)

VLLM_METRICS = """\
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{{engine="0",model_name="m"}} {gen}
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{{engine="0",model_name="m"}} {pre}
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{engine="0",model_name="m"}} 12
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{engine="0",model_name="m"}} 3
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} 0.25
"""


def engine_sample(endpoint, t, gen, pre):
    snapshot = parse_prometheus_text(VLLM_METRICS.format(gen=gen, pre=pre))["0"]
    return EngineSample(endpoint=endpoint, engine_label="0", timestamp=t, snapshot=snapshot)


def test_load_sample_carries_generation_token_delta():
    collector = InferenceMetricsCollector([])
    endpoint = MetricsEndpoint(client=None, role=None, key="http://x", name="server0")
    first = engine_sample(endpoint, 0.0, gen=1000.0, pre=2.0)
    load = collector.build_load_sample(first)
    assert load.generation_tokens_delta == 0.0  # no baseline yet
    collector.previous[first.key] = TimedSnapshot(timestamp=first.timestamp, snapshot=first.snapshot)
    second = engine_sample(endpoint, 5.0, gen=6000.0, pre=5.0)
    load = collector.build_load_sample(second)
    assert load.generation_tokens_delta == 5000.0
    assert load.preemptions_delta == 3
    assert load.running == 12 and load.waiting == 3 and load.kv_usage == 0.25


def test_counter_reset_reads_as_no_signal():
    collector = InferenceMetricsCollector([])
    endpoint = MetricsEndpoint(client=None, role=None, key="http://x", name="server0")
    first = engine_sample(endpoint, 0.0, gen=9000.0, pre=0.0)
    collector.previous[first.key] = TimedSnapshot(timestamp=first.timestamp, snapshot=first.snapshot)
    load = collector.build_load_sample(engine_sample(endpoint, 5.0, gen=100.0, pre=0.0))
    assert load.generation_tokens_delta == 0.0
