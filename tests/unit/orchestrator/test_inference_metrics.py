import pytest

from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector, parse_prometheus_text
from tests.unit.orchestrator.fakes import RecordingHooks, RecordingMonitors

EXPOSITION = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0"} 12.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0"} 3.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0"} 0.42
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0"} PREEMPTIONS
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{engine="0",kv_cache_size_tokens="80000",block_size="16"} 1.0
"""


class FakeResponse:
    def __init__(self, text="", json=None):
        self.text = text
        self._json = json or {}

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class FakeAdminClient:
    base_url = "http://engine:8000"

    def __init__(self):
        self.preemptions = 0

    async def get(self, path, timeout=None):
        if path == "/metrics":
            return FakeResponse(text=EXPOSITION.replace("PREEMPTIONS", str(self.preemptions)))
        return FakeResponse(json={"data": [{"max_model_len": 4096}]})


def test_parse_exposition_keeps_gauges_counters_and_cache_config():
    engines = parse_prometheus_text(EXPOSITION.replace("PREEMPTIONS", "2"))
    snapshot = engines["0"]
    assert snapshot.gauges["num_requests_running"] == 12.0
    assert snapshot.counters["num_preemptions_total"] == 2.0
    assert snapshot.cache_config["kv_cache_size_tokens"] == "80000"


@pytest.mark.asyncio
async def test_collect_feeds_load_samples_and_logs_metrics():
    client = FakeAdminClient()
    collector = InferenceMetricsCollector([client])
    hooks, monitors = RecordingHooks(), RecordingMonitors()
    collector.bind(on_load=hooks.record("on_load"), monitors=monitors)
    await collector.collect_and_log()
    ((samples,),) = hooks["on_load"]
    (sample,) = samples
    assert sample.engine_id == "server0.0"
    assert sample.kv_capacity_tokens == 80000 and sample.max_model_len == 4096
    assert sample.running == 12 and sample.waiting == 3 and sample.kv_usage == 0.42
    assert sample.preemptions_delta == 0  # no baseline yet
    assert monitors.metrics[0][1] is None
    assert monitors.metrics[0][0]["inference/server0.0/num_requests_running"] == 12.0
    # a counter delta shows up on the next poll
    client.preemptions = 5
    await collector.collect_and_log()
    assert hooks["on_load"][1][0][0].preemptions_delta == 5


@pytest.mark.asyncio
async def test_log_false_still_feeds_the_controller():
    collector = InferenceMetricsCollector([FakeAdminClient()], log=False)
    hooks, monitors = RecordingHooks(), RecordingMonitors()
    collector.bind(on_load=hooks.record("on_load"), monitors=monitors)
    assert await collector.probe(attempts=1)
    assert len(hooks["on_load"]) == 1 and monitors.metrics == []
