import httpx
import pytest

from prime_rl.orchestrator.inference_metrics import (
    EngineSample,
    EngineSnapshot,
    InferenceMetricsCollector,
    TimedSnapshot,
    engine_values,
    ratio_deltas,
)


def test_speculative_ratios_pool_counts_and_skip_resets():
    collector = InferenceMetricsCollector([httpx.AsyncClient(base_url="http://localhost:8000")])
    endpoint = collector.endpoints[0]
    names = (
        "spec_decode_num_accepted_tokens_total",
        "spec_decode_num_draft_tokens_total",
        "spec_decode_num_drafts_total",
    )
    samples = []
    for engine, counts in enumerate(((140, 200, 100), (190, 400, 200), (90, 200, 200))):
        sample = EngineSample(endpoint, str(engine), 10.0, EngineSnapshot(counters=dict(zip(names, counts))))
        collector.previous[sample.key] = TimedSnapshot(5.0, EngineSnapshot(counters=dict.fromkeys(names, 100)))
        samples.append(sample)
    metrics = collector.build_metrics(samples)
    assert metrics["inference/agg/spec_decode_acceptance_rate/pooled"] == pytest.approx(130 / 400)
    assert metrics["inference/agg/spec_decode_acceptance_length/pooled"] == pytest.approx(1 + 90 / 100)
    assert metrics["inference/server0.0/spec_decode_acceptance_rate"] == pytest.approx(0.4)
    assert "inference/server0.0/spec_decode_acceptance_length" not in metrics
    assert "inference/server0.2/spec_decode_acceptance_rate" not in metrics


@pytest.mark.parametrize("previous_counts", [None, {}, {"spec_decode_num_drafts_total": 10}])
def test_speculative_ratios_require_two_complete_scrapes(previous_counts):
    collector = InferenceMetricsCollector([httpx.AsyncClient(base_url="http://localhost:8000")])
    snapshot = EngineSnapshot(
        counters={"spec_decode_num_drafts_total": 20, "spec_decode_num_accepted_tokens_total": 40}
    )
    sample = EngineSample(collector.endpoints[0], "0", 10.0, snapshot)
    previous = TimedSnapshot(5.0, EngineSnapshot(counters=previous_counts)) if previous_counts is not None else None
    assert ratio_deltas(sample, previous) == {}
    assert "spec_decode_acceptance_length" not in engine_values(sample, previous)


def test_speculative_zero_acceptance_includes_bonus_token():
    collector = InferenceMetricsCollector([httpx.AsyncClient(base_url="http://localhost:8000")])
    snapshot = EngineSnapshot(counters={"spec_decode_num_drafts": 10, "spec_decode_num_accepted_tokens": 0})
    sample = EngineSample(collector.endpoints[0], "0", 10.0, snapshot)
    previous = TimedSnapshot(5.0, EngineSnapshot(counters=dict.fromkeys(snapshot.counters, 0)))
    assert engine_values(sample, previous)["spec_decode_acceptance_length"] == 1.0
