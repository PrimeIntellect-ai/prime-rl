#!/usr/bin/env python3
"""Summarize vLLM throughput logger snapshots from inference logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

THROUGHPUT_RE = re.compile(
    r"INFO (?P<stamp>\d\d-\d\d \d\d:\d\d:\d\d) .*"
    r"Avg prompt throughput: (?P<prompt>[0-9.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%, "
    r"Prefix cache hit rate: (?P<prefix>[0-9.]+)%"
)


@dataclass(frozen=True)
class ThroughputPoint:
    stamp: str
    prompt_tokens_s: float
    generation_tokens_s: float
    running: int
    waiting: int
    kv_cache_pct: float
    prefix_cache_hit_pct: float


@dataclass(frozen=True)
class RunSummary:
    label: str
    path: str
    total_points: int
    matching_points: int
    first_n: int
    first_n_mean_generation_tokens_s: float | None
    all_matching_mean_generation_tokens_s: float | None
    matching: list[ThroughputPoint]
    first_n_points: list[ThroughputPoint]


def parse_points(path: Path) -> list[ThroughputPoint]:
    points = []
    for line in path.read_text().splitlines():
        match = THROUGHPUT_RE.search(line)
        if match is None:
            continue
        points.append(
            ThroughputPoint(
                stamp=match.group("stamp"),
                prompt_tokens_s=float(match.group("prompt")),
                generation_tokens_s=float(match.group("generation")),
                running=int(match.group("running")),
                waiting=int(match.group("waiting")),
                kv_cache_pct=float(match.group("kv")),
                prefix_cache_hit_pct=float(match.group("prefix")),
            )
        )
    return points


def infer_label(path: Path) -> str:
    parts = path.parts
    for index, part in enumerate(parts):
        if part == "sampling-kernel" and index + 2 < len(parts):
            return "/".join(parts[index + 1 : index + 3])
    return path.parent.name or path.name


def prompt_matches(
    point: ThroughputPoint,
    exact_prompt_tokens_s: float,
    min_prompt_tokens_s: float | None,
    max_prompt_tokens_s: float | None,
) -> bool:
    if min_prompt_tokens_s is None and max_prompt_tokens_s is None:
        return point.prompt_tokens_s == exact_prompt_tokens_s
    if min_prompt_tokens_s is not None and point.prompt_tokens_s < min_prompt_tokens_s:
        return False
    return max_prompt_tokens_s is None or point.prompt_tokens_s <= max_prompt_tokens_s


def summarize(
    path: Path,
    label: str,
    first_n: int,
    running: int,
    waiting: int | None,
    min_waiting: int | None,
    min_kv_cache_pct: float | None,
    max_kv_cache_pct: float | None,
    prompt_tokens_s: float,
    min_prompt_tokens_s: float | None,
    max_prompt_tokens_s: float | None,
) -> RunSummary:
    points = parse_points(path)
    matching = [
        point
        for point in points
        if prompt_matches(
            point,
            exact_prompt_tokens_s=prompt_tokens_s,
            min_prompt_tokens_s=min_prompt_tokens_s,
            max_prompt_tokens_s=max_prompt_tokens_s,
        )
        and point.running == running
        and (
            (min_waiting is not None and point.waiting >= min_waiting)
            or (min_waiting is None and waiting is not None and point.waiting == waiting)
        )
        and (min_kv_cache_pct is None or point.kv_cache_pct >= min_kv_cache_pct)
        and (max_kv_cache_pct is None or point.kv_cache_pct <= max_kv_cache_pct)
    ]
    first = matching[:first_n]
    return RunSummary(
        label=label,
        path=str(path),
        total_points=len(points),
        matching_points=len(matching),
        first_n=first_n,
        first_n_mean_generation_tokens_s=(mean(point.generation_tokens_s for point in first) if first else None),
        all_matching_mean_generation_tokens_s=(
            mean(point.generation_tokens_s for point in matching) if matching else None
        ),
        matching=matching,
        first_n_points=first,
    )


def parse_run(raw: str) -> tuple[str | None, Path]:
    if "=" not in raw:
        return None, Path(raw)
    label, path = raw.split("=", 1)
    return label, Path(path)


def print_markdown(summaries: list[RunSummary]) -> None:
    baseline_first = summaries[0].first_n_mean_generation_tokens_s if summaries else None
    baseline_all = summaries[0].all_matching_mean_generation_tokens_s if summaries else None
    print(
        "| run | matching points | first-N mean gen tok/s | first-N ratio vs first | "
        "all matching mean gen tok/s | all ratio vs first | first-N values | prefix % | KV % |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        values = ", ".join(f"{point.generation_tokens_s:.1f}" for point in summary.first_n_points)
        prefixes = ", ".join(f"{point.prefix_cache_hit_pct:.1f}" for point in summary.first_n_points)
        kvs = ", ".join(f"{point.kv_cache_pct:.1f}" for point in summary.first_n_points)
        first_mean_value = summary.first_n_mean_generation_tokens_s
        all_mean_value = summary.all_matching_mean_generation_tokens_s
        first_mean = f"{first_mean_value:.1f}" if first_mean_value is not None else ""
        all_mean = f"{all_mean_value:.1f}" if all_mean_value is not None else ""
        first_ratio = (
            f"{first_mean_value / baseline_first:.3f}x" if first_mean_value is not None and baseline_first else ""
        )
        all_ratio = f"{all_mean_value / baseline_all:.3f}x" if all_mean_value is not None and baseline_all else ""
        print(
            f"| {summary.label} | {summary.matching_points} | {first_mean} | "
            f"{first_ratio} | {all_mean} | {all_ratio} | {values} | {prefixes} | {kvs} |"
        )


def print_points(summaries: list[RunSummary]) -> None:
    print()
    print("| run | stamp | prompt tok/s | gen tok/s | running | waiting | KV % | prefix % |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for summary in summaries:
        for point in summary.matching:
            print(
                f"| {summary.label} | {point.stamp} | {point.prompt_tokens_s:.1f} | "
                f"{point.generation_tokens_s:.1f} | {point.running} | {point.waiting} | "
                f"{point.kv_cache_pct:.1f} | {point.prefix_cache_hit_pct:.1f} |"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="Log path or label=log path. Use logs/inference/node_0.log.",
    )
    parser.add_argument("--first-n", type=int, default=4)
    parser.add_argument("--running", type=int, default=256)
    parser.add_argument("--waiting", type=int, default=4344)
    parser.add_argument(
        "--min-waiting",
        type=int,
        default=None,
        help="Use waiting >= this value instead of exact --waiting.",
    )
    parser.add_argument("--min-kv-cache-pct", type=float, default=None)
    parser.add_argument("--max-kv-cache-pct", type=float, default=None)
    parser.add_argument("--prompt-tokens-s", type=float, default=0.0)
    parser.add_argument(
        "--min-prompt-tokens-s",
        type=float,
        default=None,
        help="Optional lower prompt-throughput bound. If unset with --max-prompt-tokens-s, use exact --prompt-tokens-s.",
    )
    parser.add_argument(
        "--max-prompt-tokens-s",
        type=float,
        default=None,
        help="Optional upper prompt-throughput bound. Enables range filtering instead of exact --prompt-tokens-s.",
    )
    parser.add_argument(
        "--min-matching-points",
        type=int,
        default=0,
        help="Exit nonzero if any run has fewer matching points than this.",
    )
    parser.add_argument(
        "--print-points",
        action="store_true",
        help="Print every point that survived the filters for audit/debugging.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for raw_run in args.runs:
        label, path = parse_run(raw_run)
        summaries.append(
            summarize(
                path=path,
                label=label or infer_label(path),
                first_n=args.first_n,
                running=args.running,
                waiting=args.waiting,
                min_waiting=args.min_waiting,
                min_kv_cache_pct=args.min_kv_cache_pct,
                max_kv_cache_pct=args.max_kv_cache_pct,
                prompt_tokens_s=args.prompt_tokens_s,
                min_prompt_tokens_s=args.min_prompt_tokens_s,
                max_prompt_tokens_s=args.max_prompt_tokens_s,
            )
        )

    if args.json:
        print(json.dumps([asdict(summary) for summary in summaries], indent=2))
    else:
        print_markdown(summaries)
        if args.print_points:
            print_points(summaries)

    failing = [
        summary
        for summary in summaries
        if args.min_matching_points and summary.matching_points < args.min_matching_points
    ]
    if failing:
        labels = ", ".join(f"{summary.label}={summary.matching_points}" for summary in failing)
        print(
            f"ERROR: fewer than {args.min_matching_points} matching points: {labels}",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
