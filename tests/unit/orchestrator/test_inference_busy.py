import httpx

from prime_rl.orchestrator.inference_metrics import EngineSample, EngineSnapshot, InferenceMetricsCollector


def test_collector_reports_engine_busy_time_since_the_last_poll():
    collector = InferenceMetricsCollector([httpx.AsyncClient(base_url="http://server")], log_metrics=False)
    endpoint = collector.endpoints[0]
    sample = EngineSample(endpoint=endpoint, engine_label="0", timestamp=0.0, snapshot=EngineSnapshot())
    for busy, expected in ((10.0, None), (13.5, 3.5), (1.0, None), (2.0, 1.0)):  # first poll, delta, restart, delta
        collector.busy_by_endpoint[endpoint.key] = {"0": busy}
        assert collector.busy_delta(sample) == expected
    collector.busy_by_endpoint[endpoint.key] = {}
    assert collector.busy_delta(sample) is None  # not reported
