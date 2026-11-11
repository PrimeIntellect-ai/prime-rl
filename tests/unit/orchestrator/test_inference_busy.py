from pathlib import Path

import httpx

from prime_rl.inference.vllm.engine_busy import read_busy, write_busy
from prime_rl.orchestrator.inference_metrics import EngineSample, EngineSnapshot, InferenceMetricsCollector


def test_busy_files_round_trip(tmp_path: Path):
    write_busy(tmp_path, 0, 1.5)
    write_busy(tmp_path, 3, 2.25)
    write_busy(tmp_path, 0, 4.0)
    assert read_busy(tmp_path) == {"0": 4.0, "3": 2.25}
    assert read_busy(tmp_path / "missing") == {}


def test_collector_reports_busy_seconds_since_the_last_poll():
    collector = InferenceMetricsCollector([httpx.AsyncClient(base_url="http://server")], log_metrics=False)
    endpoint = collector.endpoints[0]
    sample = EngineSample(endpoint=endpoint, engine_label="0", timestamp=0.0, snapshot=EngineSnapshot())
    collector.busy_by_endpoint[endpoint.key] = {"0": 10.0}
    assert collector.busy_delta(sample) is None  # first poll: no baseline
    collector.busy_by_endpoint[endpoint.key] = {"0": 13.5}
    assert collector.busy_delta(sample) == 3.5
    collector.busy_by_endpoint[endpoint.key] = {"0": 1.0}
    assert collector.busy_delta(sample) is None  # the engine restarted
    collector.busy_by_endpoint[endpoint.key] = {}
    assert collector.busy_delta(sample) is None  # not reported
