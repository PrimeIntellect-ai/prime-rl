import pytest

from scripts.analyze_wandb_production_gate import (
    InferenceSummary,
    LengthSummary,
    ProgressSummary,
    RunReport,
    TrainerSummary,
    classify_comparison,
    compact_float,
    compare_reports,
    print_comparisons,
    ratio_roofline,
    summarize_inference,
    summarize_trainer,
)


def _report(
    *,
    label: str,
    step_mean_s: float,
    wait_mean_s: float,
    wait_fraction: float,
    fwd_mean_s: float,
    broadcast_mean_s: float,
    throughput_mean: float,
    implied_decode_mean: float,
) -> RunReport:
    return RunReport(
        label=label,
        run_id=f"{label}-id",
        name=label,
        state="finished",
        created_at="2026-06-26T00:00:00",
        runtime_s=None,
        history_rows=1,
        trainer=TrainerSummary(
            source_rows=1,
            skipped_rows=0,
            rows=1,
            step_sum_s=step_mean_s,
            step_mean_s=step_mean_s,
            wait_sum_s=wait_mean_s,
            wait_mean_s=wait_mean_s,
            wait_fraction=wait_fraction,
            forward_backward_mean_s=fwd_mean_s,
            broadcast_mean_s=broadcast_mean_s,
            mfu_mean=None,
            throughput_mean=None,
        ),
        inference=InferenceSummary(
            rows=1,
            active_rows=1,
            running_cap=256.0,
            saturated_fraction=1.0,
            waiting_positive_fraction=1.0,
            running_median=256.0,
            waiting_median=64.0,
            kv_cache_mean_median=0.2,
            kv_cache_max_median=0.4,
            queue_time_median_s=10.0,
            prefix_cache_hit_rate_mean=0.5,
            throughput_mean=throughput_mean,
            throughput_median=throughput_mean,
            implied_decode_tokens_s_mean=implied_decode_mean,
            implied_decode_tokens_s_median=implied_decode_mean,
        ),
        progress=ProgressSummary(
            rows=0,
            kept_decode_tokens_s_mean=None,
            kept_decode_tokens_s_median=None,
        ),
        lengths=LengthSummary(
            rows=0,
            decode_len_mean_median=None,
            decode_len_max_max=None,
            seq_len_mean_median=None,
            generation_mean_median_s=None,
        ),
        roofline_by_serving_ratio={},
    )


def test_compact_float_rejects_nonfinite_values() -> None:
    assert compact_float("1.5") == 1.5
    assert compact_float(None) is None
    assert compact_float("nan") is None
    assert compact_float(float("inf")) is None


def test_trainer_summary_uses_sum_weighted_wait_fraction() -> None:
    rows = [
        {
            "time/step": 100.0,
            "time/wait_for_batch": 50.0,
            "time/forward_backward": 40.0,
            "time/broadcast_weights": 5.0,
            "perf/mfu": 0.2,
            "perf/throughput": 1000.0,
        },
        {
            "time/step": 300.0,
            "time/wait_for_batch": 60.0,
            "time/forward_backward": 220.0,
            "time/broadcast_weights": 7.0,
            "perf/mfu": 0.4,
            "perf/throughput": 2000.0,
        },
        {"time/step": 999.0},
    ]

    summary = summarize_trainer(rows)

    assert summary.source_rows == 2
    assert summary.skipped_rows == 0
    assert summary.rows == 2
    assert summary.step_sum_s == 400.0
    assert summary.wait_sum_s == 110.0
    assert summary.wait_fraction == pytest.approx(0.275)
    assert summary.forward_backward_mean_s == pytest.approx(130.0)
    assert summary.broadcast_mean_s == pytest.approx(6.0)
    assert summary.mfu_mean == pytest.approx(0.3)
    assert summary.throughput_mean == pytest.approx(1500.0)


def test_trainer_summary_can_skip_startup_rows_and_limit_window() -> None:
    rows = [
        {
            "time/step": 999.0,
            "time/wait_for_batch": 900.0,
            "time/forward_backward": 90.0,
            "time/broadcast_weights": 9.0,
        },
        {
            "time/step": 200.0,
            "time/wait_for_batch": 80.0,
            "time/forward_backward": 100.0,
            "time/broadcast_weights": 10.0,
        },
        {
            "time/step": 220.0,
            "time/wait_for_batch": 88.0,
            "time/forward_backward": 110.0,
            "time/broadcast_weights": 11.0,
        },
        {
            "time/step": 500.0,
            "time/wait_for_batch": 250.0,
            "time/forward_backward": 200.0,
            "time/broadcast_weights": 20.0,
        },
    ]

    summary = summarize_trainer(rows, skip_rows=1, max_rows=2)

    assert summary.source_rows == 4
    assert summary.skipped_rows == 1
    assert summary.rows == 2
    assert summary.step_mean_s == pytest.approx(210.0)
    assert summary.wait_fraction == pytest.approx(168.0 / 420.0)
    assert summary.forward_backward_mean_s == pytest.approx(105.0)
    assert summary.broadcast_mean_s == pytest.approx(10.5)


def test_inference_summary_filters_active_rows_and_computes_decode_proxy() -> None:
    rows = [
        {
            "inference/agg/running_requests": 10.0,
            "inference/agg/waiting_requests": 0.0,
            "inference/agg/avg_tpot_seconds": 0.1,
            "inference/agg/throughput": 100.0,
        },
        {
            "inference/agg/running_requests": 160.0,
            "inference/agg/waiting_requests": 1.0,
            "inference/agg/kv_cache_usage_mean": 0.1,
            "inference/agg/kv_cache_usage_max": 0.2,
            "inference/agg/avg_queue_time_seconds": 2.0,
            "inference/agg/prefix_cache_hit_rate": 0.3,
            "inference/agg/avg_tpot_seconds": 0.02,
            "inference/agg/throughput": 6400.0,
        },
        {
            "inference/agg/running_requests": 256.0,
            "inference/agg/waiting_requests": 0.0,
            "inference/agg/kv_cache_usage_mean": 0.3,
            "inference/agg/kv_cache_usage_max": 0.4,
            "inference/agg/avg_queue_time_seconds": 4.0,
            "inference/agg/prefix_cache_hit_rate": 0.5,
            "inference/agg/avg_tpot_seconds": 0.01,
            "inference/agg/throughput": 12800.0,
        },
    ]

    summary = summarize_inference(rows)

    assert summary.rows == 3
    assert summary.active_rows == 2
    assert summary.running_cap == 256.0
    assert summary.saturated_fraction == pytest.approx(0.5)
    assert summary.waiting_positive_fraction == pytest.approx(0.5)
    assert summary.running_median == pytest.approx(208.0)
    assert summary.waiting_median == pytest.approx(0.5)
    assert summary.kv_cache_mean_median == pytest.approx(0.2)
    assert summary.kv_cache_max_median == pytest.approx(0.3)
    assert summary.queue_time_median_s == pytest.approx(3.0)
    assert summary.prefix_cache_hit_rate_mean == pytest.approx(0.4)
    assert summary.throughput_mean == pytest.approx(9600.0)
    assert summary.implied_decode_tokens_s_mean == pytest.approx(16800.0)


def test_roofline_and_ab_comparison_ratios() -> None:
    assert ratio_roofline(0.4, 1.25) == pytest.approx(1.087, rel=1e-3)

    baseline = _report(
        label="native",
        step_mean_s=200.0,
        wait_mean_s=80.0,
        wait_fraction=0.4,
        fwd_mean_s=100.0,
        broadcast_mean_s=20.0,
        throughput_mean=10000.0,
        implied_decode_mean=9000.0,
    )
    candidate = _report(
        label="patched",
        step_mean_s=180.0,
        wait_mean_s=60.0,
        wait_fraction=1.0 / 3.0,
        fwd_mean_s=100.0,
        broadcast_mean_s=18.0,
        throughput_mean=12500.0,
        implied_decode_mean=9900.0,
    )

    comparison = compare_reports(
        baseline,
        candidate,
        min_trainer_rows=1,
        min_active_inference_rows=1,
    )

    assert comparison.decision == "pass"
    assert comparison.reasons == ["E2E 1.111x >= 1.080x"]
    assert comparison.baseline_trainer_rows == 1
    assert comparison.candidate_trainer_rows == 1
    assert comparison.baseline_active_inference_rows == 1
    assert comparison.candidate_active_inference_rows == 1
    assert comparison.step_speed_ratio == pytest.approx(200.0 / 180.0)
    assert comparison.serving_throughput_ratio == pytest.approx(1.25)
    assert comparison.implied_decode_ratio == pytest.approx(1.1)
    assert comparison.wait_fraction_delta == pytest.approx(-1.0 / 15.0)
    assert comparison.wait_mean_ratio == pytest.approx(80.0 / 60.0)
    assert comparison.forward_backward_ratio == pytest.approx(1.0)
    assert comparison.broadcast_ratio == pytest.approx(20.0 / 18.0)


def test_comparison_classification_distinguishes_weak_fail_and_mixed() -> None:
    assert (
        classify_comparison(
            baseline_trainer_rows=2,
            candidate_trainer_rows=2,
            baseline_active_inference_rows=10,
            candidate_active_inference_rows=10,
            step_speed_ratio=1.05,
            serving_throughput_ratio=1.2,
            e2e_pass_ratio=1.08,
            e2e_fail_ratio=1.03,
            serving_pass_ratio=1.10,
            min_trainer_rows=2,
            min_active_inference_rows=10,
        )[0]
        == "weak_positive"
    )
    assert (
        classify_comparison(
            baseline_trainer_rows=2,
            candidate_trainer_rows=2,
            baseline_active_inference_rows=10,
            candidate_active_inference_rows=10,
            step_speed_ratio=1.02,
            serving_throughput_ratio=1.2,
            e2e_pass_ratio=1.08,
            e2e_fail_ratio=1.03,
            serving_pass_ratio=1.10,
            min_trainer_rows=2,
            min_active_inference_rows=10,
        )[0]
        == "fail"
    )
    decision, reasons = classify_comparison(
        baseline_trainer_rows=2,
        candidate_trainer_rows=2,
        baseline_active_inference_rows=10,
        candidate_active_inference_rows=10,
        step_speed_ratio=1.09,
        serving_throughput_ratio=1.05,
        e2e_pass_ratio=1.08,
        e2e_fail_ratio=1.03,
        serving_pass_ratio=1.10,
        min_trainer_rows=2,
        min_active_inference_rows=10,
    )
    assert decision == "mixed"
    assert reasons == [
        "serving throughput 1.050x < 1.100x",
        "E2E 1.090x passes but serving/supporting gate does not",
    ]


def test_comparison_classification_fails_closed_on_insufficient_rows() -> None:
    decision, reasons = classify_comparison(
        baseline_trainer_rows=1,
        candidate_trainer_rows=2,
        baseline_active_inference_rows=9,
        candidate_active_inference_rows=10,
        step_speed_ratio=1.2,
        serving_throughput_ratio=1.3,
        e2e_pass_ratio=1.08,
        e2e_fail_ratio=1.03,
        serving_pass_ratio=1.10,
        min_trainer_rows=2,
        min_active_inference_rows=10,
    )

    assert decision == "missing"
    assert reasons == [
        "baseline trainer rows 1 < 2",
        "baseline active inference rows 9 < 10",
    ]


def test_comparison_table_reports_row_counts(capsys: pytest.CaptureFixture[str]) -> None:
    baseline = _report(
        label="native",
        step_mean_s=200.0,
        wait_mean_s=80.0,
        wait_fraction=0.4,
        fwd_mean_s=100.0,
        broadcast_mean_s=20.0,
        throughput_mean=10000.0,
        implied_decode_mean=9000.0,
    )
    candidate = _report(
        label="patched",
        step_mean_s=180.0,
        wait_mean_s=60.0,
        wait_fraction=1.0 / 3.0,
        fwd_mean_s=100.0,
        broadcast_mean_s=18.0,
        throughput_mean=12500.0,
        implied_decode_mean=9900.0,
    )
    comparison = compare_reports(
        baseline,
        candidate,
        min_trainer_rows=1,
        min_active_inference_rows=1,
    )

    print_comparisons([comparison])

    output = capsys.readouterr().out
    assert "trainer rows" in output
    assert "active inf rows" in output
    assert "| native | patched | pass | 1/1 | 1/1 |" in output
