import httpx

from prime_rl.orchestrator.inference_metrics import EngineSample, EngineSnapshot, InferenceMetricsCollector


def test_collector_reports_engine_steps_since_the_last_poll():
    collector = InferenceMetricsCollector([httpx.AsyncClient(base_url="http://server")], log_metrics=False)
    endpoint = collector.endpoints[0]
    sample = EngineSample(endpoint=endpoint, engine_label="0", timestamp=0.0, snapshot=EngineSnapshot())
    collector.busy_by_endpoint[endpoint.key] = {"0": {"busy_s": 10.0, "steps": 100, "max_num_seqs": 256}}
    assert collector.step_deltas(sample) is None  # first poll: no baseline
    collector.busy_by_endpoint[endpoint.key] = {"0": {"busy_s": 13.5, "steps": 150, "max_num_seqs": 256}}
    assert collector.step_deltas(sample) == {"busy_s": 3.5, "steps": 50, "max_num_seqs": 256}
    collector.busy_by_endpoint[endpoint.key] = {"0": {"busy_s": 1.0, "steps": 10, "max_num_seqs": 256}}
    assert collector.step_deltas(sample) is None  # the engine restarted
    collector.busy_by_endpoint[endpoint.key] = {}
    assert collector.step_deltas(sample) is None  # not reported
