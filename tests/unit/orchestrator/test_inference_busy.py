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


def test_step_contents_counts_first_step_decodes():
    from types import SimpleNamespace

    from prime_rl.inference.vllm.engine_busy import step_contents

    step = SimpleNamespace(
        # New: one prefilling 300 tokens, one decoding on its first step (P/D decode engine, 900 tokens of context)
        scheduled_new_reqs=[
            SimpleNamespace(req_id="p", num_computed_tokens=0),
            SimpleNamespace(req_id="d", num_computed_tokens=900),
        ],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["c"], num_computed_tokens=[500]),
        num_scheduled_tokens={"p": 300, "d": 1, "c": 1},
        scheduled_spec_decode_tokens={},
    )
    assert step_contents(step) == (2, 1400, 300)
