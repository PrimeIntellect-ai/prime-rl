#!/usr/bin/env python3
"""Analyze W&B runs for the sampler production gate.

This script intentionally streams unfiltered ``scan_history`` rows. For current
PrimeRL runs, filtered ``scan_history(keys=...)`` can return zero rows even when
the run has history and summary metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any

import wandb

DEFAULT_ENTITY = "jvelja-private"
DEFAULT_PROJECT = "gpqa-openended-debate-calibration"
DEFAULT_SERVING_RATIOS = (1.13, 1.24)
DEFAULT_E2E_PASS_RATIO = 1.08
DEFAULT_E2E_FAIL_RATIO = 1.03
DEFAULT_SERVING_PASS_RATIO = 1.10
DEFAULT_MIN_TRAINER_ROWS = 2
DEFAULT_MIN_ACTIVE_INFERENCE_ROWS = 10


@dataclass(frozen=True)
class TrainerSummary:
    source_rows: int
    skipped_rows: int
    rows: int
    step_sum_s: float | None
    step_mean_s: float | None
    wait_sum_s: float | None
    wait_mean_s: float | None
    wait_fraction: float | None
    forward_backward_mean_s: float | None
    broadcast_mean_s: float | None
    mfu_mean: float | None
    throughput_mean: float | None


@dataclass(frozen=True)
class InferenceSummary:
    rows: int
    active_rows: int
    running_cap: float | None
    saturated_fraction: float | None
    waiting_positive_fraction: float | None
    running_median: float | None
    waiting_median: float | None
    kv_cache_mean_median: float | None
    kv_cache_max_median: float | None
    queue_time_median_s: float | None
    prefix_cache_hit_rate_mean: float | None
    throughput_mean: float | None
    throughput_median: float | None
    implied_decode_tokens_s_mean: float | None
    implied_decode_tokens_s_median: float | None


@dataclass(frozen=True)
class ProgressSummary:
    rows: int
    kept_decode_tokens_s_mean: float | None
    kept_decode_tokens_s_median: float | None


@dataclass(frozen=True)
class LengthSummary:
    rows: int
    decode_len_mean_median: float | None
    decode_len_max_max: float | None
    seq_len_mean_median: float | None
    generation_mean_median_s: float | None


@dataclass(frozen=True)
class RunReport:
    label: str
    run_id: str
    name: str
    state: str
    created_at: str
    runtime_s: float | None
    history_rows: int
    trainer: TrainerSummary
    inference: InferenceSummary
    progress: ProgressSummary
    lengths: LengthSummary
    roofline_by_serving_ratio: dict[str, float | None]


@dataclass(frozen=True)
class ComparisonReport:
    baseline: str
    candidate: str
    decision: str
    reasons: list[str]
    baseline_trainer_rows: int
    candidate_trainer_rows: int
    baseline_active_inference_rows: int
    candidate_active_inference_rows: int
    step_speed_ratio: float | None
    serving_throughput_ratio: float | None
    implied_decode_ratio: float | None
    wait_fraction_delta: float | None
    wait_mean_ratio: float | None
    forward_backward_ratio: float | None
    broadcast_ratio: float | None


def compact_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def values(rows: list[dict[str, Any]], key: str) -> list[float]:
    output = []
    for row in rows:
        value = compact_float(row.get(key))
        if value is not None:
            output.append(value)
    return output


def safe_mean(items: list[float]) -> float | None:
    return mean(items) if items else None


def safe_median(items: list[float]) -> float | None:
    return median(items) if items else None


def safe_max(items: list[float]) -> float | None:
    return max(items) if items else None


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def ratio_roofline(wait_fraction: float | None, serving_ratio: float) -> float | None:
    if wait_fraction is None:
        return None
    return 1.0 / ((1.0 - wait_fraction) + wait_fraction / serving_ratio)


def window_rows(rows: list[dict[str, Any]], skip_rows: int, max_rows: int | None) -> list[dict[str, Any]]:
    if skip_rows < 0:
        raise ValueError("skip_rows must be >= 0")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be > 0")
    selected = rows[skip_rows:]
    return selected[:max_rows] if max_rows is not None else selected


def summarize_trainer(
    rows: list[dict[str, Any]],
    *,
    skip_rows: int = 0,
    max_rows: int | None = None,
) -> TrainerSummary:
    source_rows = [row for row in rows if compact_float(row.get("time/wait_for_batch")) is not None]
    trainer_rows = window_rows(source_rows, skip_rows=skip_rows, max_rows=max_rows)
    step = values(trainer_rows, "time/step")
    wait = values(trainer_rows, "time/wait_for_batch")
    step_sum = sum(step) if step else None
    wait_sum = sum(wait) if wait else None
    wait_fraction = wait_sum / step_sum if step_sum and wait_sum is not None else None
    return TrainerSummary(
        source_rows=len(source_rows),
        skipped_rows=min(skip_rows, len(source_rows)),
        rows=len(trainer_rows),
        step_sum_s=step_sum,
        step_mean_s=safe_mean(step),
        wait_sum_s=wait_sum,
        wait_mean_s=safe_mean(wait),
        wait_fraction=wait_fraction,
        forward_backward_mean_s=safe_mean(values(trainer_rows, "time/forward_backward")),
        broadcast_mean_s=safe_mean(values(trainer_rows, "time/broadcast_weights")),
        mfu_mean=safe_mean(values(trainer_rows, "perf/mfu")),
        throughput_mean=safe_mean(values(trainer_rows, "perf/throughput")),
    )


def summarize_inference(rows: list[dict[str, Any]]) -> InferenceSummary:
    inference_rows = [row for row in rows if compact_float(row.get("inference/agg/running_requests")) is not None]
    running = values(inference_rows, "inference/agg/running_requests")
    cap = safe_max(running)
    active_rows = (
        [row for row in inference_rows if compact_float(row.get("inference/agg/running_requests")) > 0.5 * cap]
        if cap
        else []
    )
    active_running = values(active_rows, "inference/agg/running_requests")
    waiting = values(active_rows, "inference/agg/waiting_requests")
    implied_decode = [
        running_value / tpot_value
        for row in active_rows
        if (running_value := compact_float(row.get("inference/agg/running_requests"))) is not None
        and (tpot_value := compact_float(row.get("inference/agg/avg_tpot_seconds"))) is not None
        if tpot_value
    ]
    saturated_fraction = (
        mean(1.0 if value >= 0.99 * cap else 0.0 for value in active_running) if active_running and cap else None
    )
    waiting_positive_fraction = mean(1.0 if value > 0.0 else 0.0 for value in waiting) if waiting else None
    return InferenceSummary(
        rows=len(inference_rows),
        active_rows=len(active_rows),
        running_cap=cap,
        saturated_fraction=saturated_fraction,
        waiting_positive_fraction=waiting_positive_fraction,
        running_median=safe_median(active_running),
        waiting_median=safe_median(waiting),
        kv_cache_mean_median=safe_median(values(active_rows, "inference/agg/kv_cache_usage_mean")),
        kv_cache_max_median=safe_median(values(active_rows, "inference/agg/kv_cache_usage_max")),
        queue_time_median_s=safe_median(values(active_rows, "inference/agg/avg_queue_time_seconds")),
        prefix_cache_hit_rate_mean=safe_mean(values(active_rows, "inference/agg/prefix_cache_hit_rate")),
        throughput_mean=safe_mean(values(active_rows, "inference/agg/throughput")),
        throughput_median=safe_median(values(active_rows, "inference/agg/throughput")),
        implied_decode_tokens_s_mean=safe_mean(implied_decode),
        implied_decode_tokens_s_median=safe_median(implied_decode),
    )


def summarize_progress(rows: list[dict[str, Any]]) -> ProgressSummary:
    progress_rows = [
        row
        for row in rows
        if compact_float(row.get("progress/decode_tokens")) is not None
        and compact_float(row.get("_timestamp")) is not None
    ]
    progress_rows.sort(key=lambda row: compact_float(row.get("_timestamp")) or 0.0)
    kept_decode_rates = []
    previous_timestamp = None
    first_timestamp = compact_float(progress_rows[0].get("_timestamp")) if progress_rows else None
    for row in progress_rows:
        timestamp = compact_float(row.get("_timestamp"))
        decode_tokens = compact_float(row.get("progress/decode_tokens"))
        if timestamp is None or decode_tokens is None:
            continue
        if previous_timestamp is None:
            wall_s = timestamp - first_timestamp if first_timestamp is not None else None
        else:
            wall_s = timestamp - previous_timestamp
        previous_timestamp = timestamp
        if wall_s and wall_s > 0:
            kept_decode_rates.append(decode_tokens / wall_s)
    return ProgressSummary(
        rows=len(progress_rows),
        kept_decode_tokens_s_mean=safe_mean(kept_decode_rates),
        kept_decode_tokens_s_median=safe_median(kept_decode_rates),
    )


def summarize_lengths(rows: list[dict[str, Any]]) -> LengthSummary:
    length_rows = [
        row
        for row in rows
        if compact_float(row.get("decode_len/all/mean")) is not None
        or compact_float(row.get("timing/all/generation/mean")) is not None
    ]
    return LengthSummary(
        rows=len(length_rows),
        decode_len_mean_median=safe_median(values(length_rows, "decode_len/all/mean")),
        decode_len_max_max=safe_max(values(length_rows, "decode_len/all/max")),
        seq_len_mean_median=safe_median(values(length_rows, "seq_len/all/mean")),
        generation_mean_median_s=safe_median(values(length_rows, "timing/all/generation/mean")),
    )


def parse_run_spec(raw: str) -> tuple[str | None, str]:
    if "=" not in raw:
        return None, raw
    label, run_id = raw.split("=", 1)
    return label, run_id


def fetch_report(
    api: wandb.Api,
    entity: str,
    project: str,
    run_spec: str,
    serving_ratios: tuple[float, ...],
    skip_trainer_rows: int,
    max_trainer_rows: int | None,
) -> RunReport:
    label, run_id = parse_run_spec(run_spec)
    run = api.run(f"{entity}/{project}/{run_id}")
    rows = list(run.scan_history(page_size=5000))
    trainer = summarize_trainer(
        rows,
        skip_rows=skip_trainer_rows,
        max_rows=max_trainer_rows,
    )
    return RunReport(
        label=label or run.name or run.id,
        run_id=run.id,
        name=run.name,
        state=run.state,
        created_at=run.created_at,
        runtime_s=compact_float(run.summary.get("_runtime")),
        history_rows=len(rows),
        trainer=trainer,
        inference=summarize_inference(rows),
        progress=summarize_progress(rows),
        lengths=summarize_lengths(rows),
        roofline_by_serving_ratio={
            f"{ratio:.3g}": ratio_roofline(trainer.wait_fraction, ratio) for ratio in serving_ratios
        },
    )


def classify_comparison(
    *,
    baseline_trainer_rows: int,
    candidate_trainer_rows: int,
    baseline_active_inference_rows: int,
    candidate_active_inference_rows: int,
    step_speed_ratio: float | None,
    serving_throughput_ratio: float | None,
    e2e_pass_ratio: float,
    e2e_fail_ratio: float,
    serving_pass_ratio: float,
    min_trainer_rows: int,
    min_active_inference_rows: int,
) -> tuple[str, list[str]]:
    reasons = []
    if baseline_trainer_rows < min_trainer_rows:
        reasons.append(f"baseline trainer rows {baseline_trainer_rows} < {min_trainer_rows}")
    if candidate_trainer_rows < min_trainer_rows:
        reasons.append(f"candidate trainer rows {candidate_trainer_rows} < {min_trainer_rows}")
    if baseline_active_inference_rows < min_active_inference_rows:
        reasons.append(f"baseline active inference rows {baseline_active_inference_rows} < {min_active_inference_rows}")
    if candidate_active_inference_rows < min_active_inference_rows:
        reasons.append(
            f"candidate active inference rows {candidate_active_inference_rows} < {min_active_inference_rows}"
        )
    if reasons:
        return "missing", reasons
    if step_speed_ratio is None:
        reasons.append("missing step speed ratio")
        return "missing", reasons
    if serving_throughput_ratio is None:
        reasons.append("missing serving throughput ratio")
    elif serving_throughput_ratio < serving_pass_ratio:
        reasons.append(f"serving throughput {serving_throughput_ratio:.3f}x < {serving_pass_ratio:.3f}x")

    if step_speed_ratio >= e2e_pass_ratio:
        if not reasons:
            reasons.append(f"E2E {step_speed_ratio:.3f}x >= {e2e_pass_ratio:.3f}x")
            return "pass", reasons
        reasons.append(f"E2E {step_speed_ratio:.3f}x passes but serving/supporting gate does not")
        return "mixed", reasons
    if step_speed_ratio >= e2e_fail_ratio:
        reasons.append(
            f"E2E {step_speed_ratio:.3f}x in weak-positive band [{e2e_fail_ratio:.3f}, {e2e_pass_ratio:.3f})"
        )
        return "weak_positive", reasons
    reasons.append(f"E2E {step_speed_ratio:.3f}x < {e2e_fail_ratio:.3f}x")
    return "fail", reasons


def compare_reports(
    baseline: RunReport,
    candidate: RunReport,
    *,
    e2e_pass_ratio: float = DEFAULT_E2E_PASS_RATIO,
    e2e_fail_ratio: float = DEFAULT_E2E_FAIL_RATIO,
    serving_pass_ratio: float = DEFAULT_SERVING_PASS_RATIO,
    min_trainer_rows: int = DEFAULT_MIN_TRAINER_ROWS,
    min_active_inference_rows: int = DEFAULT_MIN_ACTIVE_INFERENCE_ROWS,
) -> ComparisonReport:
    step_speed_ratio = safe_ratio(baseline.trainer.step_mean_s, candidate.trainer.step_mean_s)
    serving_throughput_ratio = safe_ratio(
        candidate.inference.throughput_mean,
        baseline.inference.throughput_mean,
    )
    decision, reasons = classify_comparison(
        baseline_trainer_rows=baseline.trainer.rows,
        candidate_trainer_rows=candidate.trainer.rows,
        baseline_active_inference_rows=baseline.inference.active_rows,
        candidate_active_inference_rows=candidate.inference.active_rows,
        step_speed_ratio=step_speed_ratio,
        serving_throughput_ratio=serving_throughput_ratio,
        e2e_pass_ratio=e2e_pass_ratio,
        e2e_fail_ratio=e2e_fail_ratio,
        serving_pass_ratio=serving_pass_ratio,
        min_trainer_rows=min_trainer_rows,
        min_active_inference_rows=min_active_inference_rows,
    )
    return ComparisonReport(
        baseline=baseline.label,
        candidate=candidate.label,
        decision=decision,
        reasons=reasons,
        baseline_trainer_rows=baseline.trainer.rows,
        candidate_trainer_rows=candidate.trainer.rows,
        baseline_active_inference_rows=baseline.inference.active_rows,
        candidate_active_inference_rows=candidate.inference.active_rows,
        step_speed_ratio=step_speed_ratio,
        serving_throughput_ratio=serving_throughput_ratio,
        implied_decode_ratio=safe_ratio(
            candidate.inference.implied_decode_tokens_s_mean,
            baseline.inference.implied_decode_tokens_s_mean,
        ),
        wait_fraction_delta=(
            candidate.trainer.wait_fraction - baseline.trainer.wait_fraction
            if candidate.trainer.wait_fraction is not None and baseline.trainer.wait_fraction is not None
            else None
        ),
        wait_mean_ratio=safe_ratio(baseline.trainer.wait_mean_s, candidate.trainer.wait_mean_s),
        forward_backward_ratio=safe_ratio(
            baseline.trainer.forward_backward_mean_s,
            candidate.trainer.forward_backward_mean_s,
        ),
        broadcast_ratio=safe_ratio(
            baseline.trainer.broadcast_mean_s,
            candidate.trainer.broadcast_mean_s,
        ),
    )


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def print_markdown(reports: list[RunReport], serving_ratios: tuple[float, ...]) -> None:
    roof_headers = " | ".join(f"E2E roof @ {ratio:.3g}x" for ratio in serving_ratios)
    print(
        "| run | id | trainer rows | wait frac | fwd s | broadcast s | inf cap | "
        "run med | wait med | queue med s | KV med/max | agg tok/s | implied decode tok/s | "
        f"{roof_headers} |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(serving_ratios))
    for report in reports:
        trainer = report.trainer
        inference = report.inference
        roof_values = " | ".join(fmt(report.roofline_by_serving_ratio.get(f"{ratio:.3g}")) for ratio in serving_ratios)
        print(
            f"| {report.label} | `{report.run_id[:8]}` | {trainer.rows} | "
            f"{fmt(trainer.wait_fraction)} | {fmt(trainer.forward_backward_mean_s, 1)} | "
            f"{fmt(trainer.broadcast_mean_s, 1)} | {fmt(inference.running_cap, 0)} | "
            f"{fmt(inference.running_median, 0)} | {fmt(inference.waiting_median, 0)} | "
            f"{fmt(inference.queue_time_median_s, 1)} | "
            f"{fmt(inference.kv_cache_mean_median)}/{fmt(inference.kv_cache_max_median)} | "
            f"{fmt(inference.throughput_mean, 0)} | "
            f"{fmt(inference.implied_decode_tokens_s_mean, 0)} | {roof_values} |"
        )


def print_comparisons(comparisons: list[ComparisonReport]) -> None:
    if not comparisons:
        return
    print()
    print(
        "| baseline | candidate | decision | trainer rows | active inf rows | step speed ratio | "
        "serving throughput ratio | implied decode ratio | wait frac delta | wait speed ratio | "
        "fwd/bwd ratio | broadcast ratio | reasons |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for comparison in comparisons:
        print(
            f"| {comparison.baseline} | {comparison.candidate} | "
            f"{comparison.decision} | "
            f"{comparison.baseline_trainer_rows}/{comparison.candidate_trainer_rows} | "
            f"{comparison.baseline_active_inference_rows}/{comparison.candidate_active_inference_rows} | "
            f"{fmt(comparison.step_speed_ratio)} | "
            f"{fmt(comparison.serving_throughput_ratio)} | "
            f"{fmt(comparison.implied_decode_ratio)} | "
            f"{fmt(comparison.wait_fraction_delta)} | "
            f"{fmt(comparison.wait_mean_ratio)} | "
            f"{fmt(comparison.forward_backward_ratio)} | "
            f"{fmt(comparison.broadcast_ratio)} | "
            f"{'; '.join(comparison.reasons)} |"
        )


def parse_compare(raw: str) -> tuple[str, str]:
    if "," not in raw:
        raise argparse.ArgumentTypeError("expected BASELINE_LABEL,CANDIDATE_LABEL")
    baseline, candidate = raw.split(",", 1)
    if not baseline or not candidate:
        raise argparse.ArgumentTypeError("expected non-empty BASELINE_LABEL,CANDIDATE_LABEL")
    return baseline, candidate


def parse_serving_ratios(raw: str) -> tuple[float, ...]:
    ratios = tuple(float(part) for part in raw.split(",") if part)
    if not ratios:
        raise argparse.ArgumentTypeError("expected at least one serving ratio")
    return ratios


def nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("expected value >= 0")
    return value


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected value > 0")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="W&B run id or label=run_id.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--serving-ratios",
        type=parse_serving_ratios,
        default=DEFAULT_SERVING_RATIOS,
        help="Comma-separated serving speedups for E2E roofline. Default: 1.13,1.24.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        type=parse_compare,
        metavar="BASELINE_LABEL,CANDIDATE_LABEL",
        help="Print A/B ratios for a labeled pair. Can be repeated.",
    )
    parser.add_argument(
        "--skip-trainer-rows",
        type=nonnegative_int,
        default=0,
        help="Drop the first N trainer metric rows from each run before computing step means.",
    )
    parser.add_argument(
        "--max-trainer-rows",
        type=positive_int,
        default=None,
        help="Use at most N trainer metric rows after --skip-trainer-rows.",
    )
    parser.add_argument("--e2e-pass-ratio", type=float, default=DEFAULT_E2E_PASS_RATIO)
    parser.add_argument("--e2e-fail-ratio", type=float, default=DEFAULT_E2E_FAIL_RATIO)
    parser.add_argument("--serving-pass-ratio", type=float, default=DEFAULT_SERVING_PASS_RATIO)
    parser.add_argument("--min-trainer-rows", type=positive_int, default=DEFAULT_MIN_TRAINER_ROWS)
    parser.add_argument(
        "--min-active-inference-rows",
        type=positive_int,
        default=DEFAULT_MIN_ACTIVE_INFERENCE_ROWS,
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api = wandb.Api(timeout=300)
    reports = [
        fetch_report(
            api,
            args.entity,
            args.project,
            run_spec,
            args.serving_ratios,
            args.skip_trainer_rows,
            args.max_trainer_rows,
        )
        for run_spec in args.runs
    ]
    reports_by_label = {report.label: report for report in reports}
    missing_labels = sorted({label for pair in args.compare for label in pair if label not in reports_by_label})
    if missing_labels:
        raise SystemExit(f"unknown --compare label(s): {', '.join(missing_labels)}")
    comparisons = [
        compare_reports(
            reports_by_label[baseline],
            reports_by_label[candidate],
            e2e_pass_ratio=args.e2e_pass_ratio,
            e2e_fail_ratio=args.e2e_fail_ratio,
            serving_pass_ratio=args.serving_pass_ratio,
            min_trainer_rows=args.min_trainer_rows,
            min_active_inference_rows=args.min_active_inference_rows,
        )
        for baseline, candidate in args.compare
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "runs": [asdict(report) for report in reports],
                    "comparisons": [asdict(comparison) for comparison in comparisons],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_markdown(reports, args.serving_ratios)
    print_comparisons(comparisons)
    if comparisons:
        print()
        print(
            "Note: W&B inference metrics are aggregate telemetry. Use vLLM "
            "per-replica logs to satisfy high-pressure row gates."
        )


if __name__ == "__main__":
    main()
