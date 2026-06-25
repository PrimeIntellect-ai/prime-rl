import asyncio

import pytest

import benchmarks.scripts.inference_backend_benchmark as benchmark
from benchmarks.scripts.inference_backend_benchmark import (
    BackendSummary,
    BenchmarkScenario,
    RegressionGate,
    RequestSample,
    ScenarioResult,
    build_suite_markdown,
    evaluate_regression_gate,
    load_scenarios,
    metrics_url_from_base_url,
    parse_backend,
    parse_metrics_text,
    percentile,
    rollup_delta,
    summarize_backend,
)


def test_percentile_interpolates_sorted_values():
    assert percentile([10.0, 20.0, 30.0], 0.50) == 20.0
    assert percentile([10.0, 20.0, 30.0, 40.0], 0.95) == pytest.approx(38.5)
    assert percentile([], 0.95) is None


def test_parse_backend_sets_metrics_url_from_openai_base_url():
    backend = parse_backend("dynamo=http://localhost:9000/v1", "PRIME_API_KEY")

    assert backend.label == "dynamo"
    assert backend.base_url == "http://localhost:9000/v1"
    assert backend.metrics_url == "http://localhost:9000/metrics"
    assert backend.api_key_var == "PRIME_API_KEY"


def test_metrics_url_from_base_url_without_v1_suffix():
    assert metrics_url_from_base_url("http://localhost:8000") == "http://localhost:8000/metrics"


def test_summarize_backend_computes_latency_and_throughput():
    samples = [
        RequestSample(
            backend="vllm",
            request_index=0,
            session_id="s0",
            ok=True,
            status_code=200,
            latency_s=1.0,
            ttft_s=0.2,
            prompt_tokens=100,
            output_tokens=20,
            total_tokens=120,
            output_chars=80,
        ),
        RequestSample(
            backend="vllm",
            request_index=1,
            session_id="s1",
            ok=True,
            status_code=200,
            latency_s=2.0,
            ttft_s=0.4,
            prompt_tokens=100,
            output_tokens=30,
            total_tokens=130,
            output_chars=120,
        ),
        RequestSample(
            backend="vllm",
            request_index=2,
            session_id="s2",
            ok=False,
            status_code=500,
            latency_s=0.5,
            ttft_s=None,
            prompt_tokens=None,
            output_tokens=None,
            total_tokens=None,
            output_chars=0,
            error="boom",
        ),
    ]

    summary = summarize_backend("vllm", samples, wall_time_s=5.0)

    assert summary.succeeded == 2
    assert summary.failed == 1
    assert summary.error_rate == pytest.approx(1 / 3)
    assert summary.requests_per_second == pytest.approx(0.4)
    assert summary.output_tokens_per_second == pytest.approx(10.0)
    assert summary.latency_p50_s == pytest.approx(1.5)
    assert summary.ttft_p95_s == pytest.approx(0.39)


def test_rollup_delta_computes_prometheus_counter_deltas():
    before = parse_metrics_text(
        """
        # HELP vllm:prompt_tokens_total Count
        # TYPE vllm:prompt_tokens_total counter
        vllm:prompt_tokens_total{engine="0"} 100
        vllm:generation_tokens_total{engine="0"} 25
        vllm:prefix_cache_queries_total{engine="0"} 10
        vllm:prefix_cache_hits_total{engine="0"} 4
        """
    )
    after = parse_metrics_text(
        """
        # HELP vllm:prompt_tokens_total Count
        # TYPE vllm:prompt_tokens_total counter
        vllm:prompt_tokens_total{engine="0"} 250
        vllm:generation_tokens_total{engine="0"} 75
        vllm:prefix_cache_queries_total{engine="0"} 30
        vllm:prefix_cache_hits_total{engine="0"} 19
        """
    )

    delta = rollup_delta(before, after)

    assert delta["prompt_tokens_total"] == 150
    assert delta["generation_tokens_total"] == 50
    assert delta["prefix_cache_queries"] == 20
    assert delta["prefix_cache_hits"] == 15
    assert delta["prefix_cache_hit_rate"] == pytest.approx(0.75)


def test_run_backend_excludes_warmup_from_summary(monkeypatch):
    calls = []

    async def fake_fetch_metrics(client, url):
        return None

    async def fake_request(
        client,
        backend,
        model,
        request_index,
        session_id,
        prompt_words,
        max_tokens,
        temperature,
        stream,
    ):
        calls.append(request_index)
        return RequestSample(
            backend=backend.label,
            request_index=request_index,
            session_id=session_id,
            ok=True,
            status_code=200,
            latency_s=0.1,
            ttft_s=0.02,
            prompt_tokens=8,
            output_tokens=4,
            total_tokens=12,
            output_chars=16,
        )

    monkeypatch.setattr(benchmark, "fetch_metrics", fake_fetch_metrics)
    monkeypatch.setattr(benchmark, "run_one_request", fake_request)

    backend = parse_backend("candidate=http://localhost:9000/v1", "PRIME_API_KEY")
    summary, samples = asyncio.run(
        benchmark.run_backend(
            backend=backend,
            model="test-model",
            requests=3,
            warmup_requests=2,
            concurrency=2,
            sessions=2,
            prompt_words=16,
            max_tokens=8,
            temperature=0.0,
            stream=True,
        )
    )

    assert calls[:2] == [-1, -2]
    assert [sample.request_index for sample in samples] == [0, 1, 2]
    assert summary.requests == 3
    assert summary.succeeded == 3


def test_evaluate_regression_gate_reports_candidate_failures():
    baseline = BackendSummary(
        label="vllm",
        requests=10,
        succeeded=10,
        failed=0,
        error_rate=0.0,
        wall_time_s=10.0,
        requests_per_second=10.0,
        output_tokens_per_second=100.0,
        output_chars_per_second=400.0,
        latency_p50_s=0.5,
        latency_p95_s=1.0,
        latency_p99_s=1.2,
        ttft_p50_s=0.1,
        ttft_p95_s=0.2,
        ttft_p99_s=0.25,
        mean_output_tokens=10.0,
        mean_output_chars=40.0,
        metrics_delta={},
    )
    candidate = BackendSummary(
        label="dynamo",
        requests=10,
        succeeded=9,
        failed=1,
        error_rate=0.1,
        wall_time_s=10.0,
        requests_per_second=8.0,
        output_tokens_per_second=70.0,
        output_chars_per_second=320.0,
        latency_p50_s=0.7,
        latency_p95_s=1.5,
        latency_p99_s=1.7,
        ttft_p50_s=0.15,
        ttft_p95_s=0.3,
        ttft_p99_s=0.35,
        mean_output_tokens=8.0,
        mean_output_chars=32.0,
        metrics_delta={},
    )

    failures = evaluate_regression_gate(
        [baseline, candidate],
        RegressionGate(
            min_request_throughput_ratio=0.9,
            min_output_throughput_ratio=0.8,
            max_latency_p95_ratio=1.2,
            max_error_rate=0.05,
        ),
    )

    assert failures == [
        "dynamo request throughput ratio 0.800 is below 0.900",
        "dynamo output token throughput ratio 0.700 is below 0.800",
        "dynamo p95 latency ratio 1.500 is above 1.200",
        "dynamo error rate 10.0% is above 5.0%",
    ]


def test_load_scenarios_applies_suite_model_and_gate_overrides(tmp_path):
    scenario_file = tmp_path / "suite.json"
    scenario_file.write_text(
        """
        {
          "model": "Qwen/Qwen3-4B-Instruct-2507",
          "scenarios": [
            {
              "name": "long_context",
              "requests": 32,
              "prompt_words": 4096,
              "gate": {
                "max_latency_p95_ratio": 1.15
              }
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    defaults = BenchmarkScenario(
        name="default",
        model="default-model",
        requests=8,
        warmup_requests=2,
        concurrency=4,
        sessions=4,
        prompt_words=128,
        max_tokens=32,
        temperature=0.0,
        stream=True,
        gate=RegressionGate(
            min_request_throughput_ratio=1.05,
            min_output_throughput_ratio=None,
            max_latency_p95_ratio=None,
            max_error_rate=0.01,
        ),
    )

    scenarios = load_scenarios(scenario_file, defaults)

    assert len(scenarios) == 1
    assert scenarios[0].name == "long_context"
    assert scenarios[0].model == "Qwen/Qwen3-4B-Instruct-2507"
    assert scenarios[0].requests == 32
    assert scenarios[0].warmup_requests == 2
    assert scenarios[0].prompt_words == 4096
    assert scenarios[0].gate.min_request_throughput_ratio == 1.05
    assert scenarios[0].gate.max_latency_p95_ratio == 1.15
    assert scenarios[0].gate.max_error_rate == 0.01


def test_build_suite_markdown_includes_each_scenario_and_gate_failure():
    summary = BackendSummary(
        label="candidate",
        requests=4,
        succeeded=4,
        failed=0,
        error_rate=0.0,
        wall_time_s=1.0,
        requests_per_second=4.0,
        output_tokens_per_second=40.0,
        output_chars_per_second=160.0,
        latency_p50_s=0.1,
        latency_p95_s=0.2,
        latency_p99_s=0.25,
        ttft_p50_s=0.02,
        ttft_p95_s=0.04,
        ttft_p99_s=0.05,
        mean_output_tokens=10.0,
        mean_output_chars=40.0,
        metrics_delta={"prefix_cache_hit_rate": 0.5},
    )

    markdown = build_suite_markdown(
        [
            ScenarioResult(
                name="session_cache_reuse",
                model="test-model",
                summaries=[summary],
                gate_failures=["candidate error rate 2.0% is above 1.0%"],
                samples=[],
            )
        ]
    )

    assert "# Inference Backend Benchmark Suite" in markdown
    assert "## session_cache_reuse" in markdown
    assert "candidate error rate 2.0% is above 1.0%" in markdown
