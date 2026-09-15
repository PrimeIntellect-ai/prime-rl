"""Summarize adaptive concurrency and inference metrics from prime-rl runs.

Usage:
    uv run python tools/analyze_adaptive_concurrency.py <run-dir-or-metrics.jsonl> [...]
"""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

METRICS = {
    "cap": "concurrency/max_inflight",
    "turnover": "concurrency/turnover",
    "inflight": "dispatcher/inflight/train",
    "generation_tps": "inference/agg/generation_tokens_total:rate/sum",
    "prompt_tps": "inference/agg/prompt_tokens_total:rate/sum",
    "kv_usage": "inference/agg/kv_cache_usage_perc/max",
    "waiting": "inference/agg/num_requests_waiting/sum",
    "waiting_capacity": "inference/agg/num_requests_waiting_reason_capacity/sum",
    "preemptions": "inference/agg/num_preemptions_total/sum",
    "prefix_hit_rate": "inference/agg/prefix_cache_hit_rate/min",
    "external_hits": "inference/agg/external_prefix_cache_hits_total/sum",
    "external_queries": "inference/agg/external_prefix_cache_queries_total/sum",
}

ROLE_SUFFIXES = {
    "generation_tps": "generation_tokens_total:rate/sum",
    "prompt_tps": "prompt_tokens_total:rate/sum",
    "kv_usage": "kv_cache_usage_perc/max",
    "waiting": "num_requests_waiting/sum",
    "waiting_capacity": "num_requests_waiting_reason_capacity/sum",
    "preemptions": "num_preemptions_total/sum",
}

Point = tuple[float, float]


def resolve_metrics_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (path / "monitors/file/metrics.jsonl", path / "metrics.jsonl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No metrics.jsonl found under {path}")


def wanted_metric(key: str) -> bool:
    if key in METRICS.values():
        return True
    parts = key.split("/", 2)
    return (
        len(parts) == 3
        and parts[0] == "inference"
        and parts[1] in ("prefill", "decode")
        and parts[2] in ROLE_SUFFIXES.values()
    )


def load_series(path: Path) -> dict[str, list[Point]]:
    series: dict[str, list[Point]] = defaultdict(list)
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            timestamp = float(row["time"])
            for key, value in row.items():
                if wanted_metric(key) and isinstance(value, int | float):
                    series[key].append((timestamp, float(value)))
    return dict(series)


def after(points: list[Point], timestamp: float) -> list[Point]:
    return [(time, value) for time, value in points if time >= timestamp]


def values(points: list[Point]) -> list[float]:
    return [value for _, value in points]


def percentile(items: list[float], quantile: float) -> float:
    ordered = sorted(items)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def counter_increase(points: list[Point]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    previous = points[0][1]
    for _, current in points[1:]:
        total += current - previous if current >= previous else current
        previous = current
    return total


def number(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


def sample_summary(points: list[Point]) -> str:
    if not points:
        return "n/a"
    samples = values(points)
    return (
        f"mean {number(mean(samples))}, median {number(median(samples))}, "
        f"p90 {number(percentile(samples, 0.9))}, max {number(max(samples))}"
    )


def cap_summary(points: list[Point]) -> str:
    if not points:
        return "n/a"
    changes = [points[0]]
    for point in points[1:]:
        if point[1] != changes[-1][1]:
            changes.append(point)
    increases = sum(current[1] > previous[1] for previous, current in zip(changes, changes[1:]))
    decreases = sum(current[1] < previous[1] for previous, current in zip(changes, changes[1:]))
    samples = values(points)
    return (
        f"{number(samples[0], 0)} -> {number(samples[-1], 0)} "
        f"(min {number(min(samples), 0)}, max {number(max(samples), 0)}, "
        f"{increases} increases, {decreases} decreases)"
    )


def nonzero_share(points: list[Point]) -> float | None:
    if not points:
        return None
    return sum(value > 0 for _, value in points) / len(points)


def threshold_share(points: list[Point], threshold: float) -> float | None:
    if not points:
        return None
    return sum(value >= threshold for _, value in points) / len(points)


def role_line(role: str, series: dict[str, list[Point]], start: float) -> str | None:
    prefix = f"inference/{role}/"
    role_series = {label: after(series.get(prefix + suffix, []), start) for label, suffix in ROLE_SUFFIXES.items()}
    if not any(role_series.values()):
        return None
    generation = values(role_series["generation_tps"])
    waiting = values(role_series["waiting"])
    kv_usage = values(role_series["kv_usage"])
    preemptions = counter_increase(role_series["preemptions"])
    return (
        f"| {role} | {number(mean(generation) if generation else None)} | "
        f"{number(max(waiting) if waiting else None)} | "
        f"{percentage(max(kv_usage) if kv_usage else None)} | {number(preemptions, 0)} |"
    )


def summarize(path: Path) -> str:
    metrics_path = resolve_metrics_path(path)
    series = load_series(metrics_path)
    generation_points = series.get(METRICS["generation_tps"], [])
    start = next((time for time, value in generation_points if value > 0), None)
    if start is None:
        raise ValueError(f"No positive generation throughput samples in {metrics_path}")

    selected = {label: after(series.get(key, []), start) for label, key in METRICS.items()}
    end = max((time for points in selected.values() for time, _ in points), default=start)
    generation = selected["generation_tps"]
    kv_usage = selected["kv_usage"]
    waiting = selected["waiting"]
    waiting_capacity = selected["waiting_capacity"]
    preemptions = selected["preemptions"]
    external_hits = counter_increase(selected["external_hits"])
    external_queries = counter_increase(selected["external_queries"])
    external_hit_rate = external_hits / external_queries if external_queries else None

    lines = [
        f"## {path.name}",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Analysis window: {datetime.fromtimestamp(start, UTC).isoformat()} to "
        f"{datetime.fromtimestamp(end, UTC).isoformat()} ({(end - start) / 60:.1f} minutes)",
        f"- Inference samples: {len(generation)}",
        f"- Concurrency cap: {cap_summary(selected['cap'])}",
        f"- Active episodes: {sample_summary(selected['inflight'])}",
        f"- Generation TPS: {sample_summary(generation)}",
        f"- Prompt TPS: {sample_summary(selected['prompt_tps'])}",
        f"- Peak KV usage: {percentage(max(values(kv_usage)) if kv_usage else None)}; "
        f"samples at or above 80%: {percentage(threshold_share(kv_usage, 0.8))}",
        f"- Waiting requests: peak {number(max(values(waiting)) if waiting else None)}; "
        f"nonzero samples {percentage(nonzero_share(waiting))}",
        f"- Capacity-waiting requests: peak "
        f"{number(max(values(waiting_capacity)) if waiting_capacity else None)}; "
        f"nonzero samples {percentage(nonzero_share(waiting_capacity))}",
        f"- Preemption events: {number(counter_increase(preemptions), 0)}",
        f"- Minimum prefix-cache hit rate: "
        f"{percentage(min(values(selected['prefix_hit_rate'])) if selected['prefix_hit_rate'] else None)}",
        f"- External prefix-cache hit rate: {percentage(external_hit_rate)} "
        f"({number(external_hits, 0)} hits / {number(external_queries, 0)} queries)",
    ]

    role_lines = [line for role in ("prefill", "decode") if (line := role_line(role, series, start))]
    if role_lines:
        lines.extend(
            [
                "",
                "| PD role | Mean generation TPS | Peak waiting | Peak KV usage | Preemptions |",
                "| --- | ---: | ---: | ---: | ---: |",
                *role_lines,
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="run directories or metrics.jsonl files")
    args = parser.parse_args()
    print("\n\n".join(summarize(run) for run in args.runs))


if __name__ == "__main__":
    main()
