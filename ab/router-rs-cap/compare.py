"""Per-variant throughput, memory and loss-parity summary of the reduce-scatter buffer cap A/B runs."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path.home() / "tmp/router-rs-cap/runs"
WAITS_DIR = Path.home() / "tmp/profiling/router-rs-cap/derived"
OVERLAP_LABEL = {"ep8": "E", "fsdp16": "D+E"}
LAYOUTS = ["fsdp16", "ep8"]
VARIANTS = ["fp32-cap1", "fp32-cap2", "fp32-cap2-r2", "fp32-main", "fp32-cap3", "bf16-cap1", "bf16-cap2"]
STEADY_STEPS = range(5, 21)
SCENARIO_STEADY_STEPS = range(10, 41)
LOSS_REFERENCE = {"fp32": "fp32-cap2", "bf16fg": "bf16fg-cap1", "bf16": "bf16-cap1"}


def load_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    by_step = defaultdict(dict)
    for line in (run_dir / "monitors/file/metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        by_step[record["step"]].update(record)
    return dict(by_step)


def find_key(metrics: dict[int, dict[str, float]], suffix: str) -> str:
    keys = {key for record in metrics.values() for key in record if key.endswith(suffix)}
    assert len(keys) == 1, f"expected one metric ending in {suffix}, got {sorted(keys)}"
    return keys.pop()


def summarize(metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    step_time_key = find_key(metrics, "time/step")
    throughput_key = find_key(metrics, "perf/throughput_per_gpu")
    memory_key = find_key(metrics, "perf/peak_memory")
    steps = [step for step in STEADY_STEPS if step in metrics]
    return {
        "steps": len(steps),
        "step_time": statistics.median(metrics[step][step_time_key] for step in steps),
        "tokens_per_gpu": statistics.median(metrics[step][throughput_key] for step in steps),
        "peak_memory": max(record[memory_key] for record in metrics.values() if memory_key in record),
    }


def paired_step_time_diff(a: dict, b: dict) -> float:
    key = find_key(a, "time/step")
    return statistics.median(b[step][key] - a[step][key] for step in STEADY_STEPS if step in a and step in b)


def max_loss_diff(a: dict, b: dict) -> float:
    return max(abs(b[step]["loss/mean"] - a[step]["loss/mean"]) for step in set(a) & set(b))


def traced_metrics(prefix, layout, variant):
    """Rank 0 and rank 8 means of the comm_waits.py summaries, if the traced run exists."""
    summaries = []
    for rank in (0, 8):
        path = WAITS_DIR / f"waits-{prefix}{layout}-rank{rank}.json"
        if path.exists() and variant in (results := json.loads(path.read_text())):
            summaries.append(results[variant]["summary"])
    if not summaries:
        return {}
    mean = lambda values: sum(values) / len(values)
    return {
        "stall_ms": mean([s["stall_ms"] for s in summaries]),
        "stall_plus_tail_ms": mean([s["stall_ms"] + s["comm_tail_ms"] for s in summaries]),
        "overlap_pct": mean([100 * s["overlap_by_label"][OVERLAP_LABEL[layout]] for s in summaries]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", default=VARIANTS)
    parser.add_argument("--csv", default=None, help="write one row per run, plus traced stall/overlap metrics")
    parser.add_argument("--scenario", action="append", default=[], help="scenario prefix(es); none is the baseline")
    args = parser.parse_args()
    variants = args.variants
    global STEADY_STEPS
    if args.scenario:
        STEADY_STEPS = SCENARIO_STEADY_STEPS
    prefixes = [f"{scenario}-" for scenario in args.scenario] or [""]
    rows = []
    for prefix, layout in [(prefix, layout) for prefix in prefixes for layout in LAYOUTS]:
        runs = {
            variant: metrics
            for variant in variants
            if (RUNS_DIR / f"{prefix}{layout}-{variant}/monitors/file/metrics.jsonl").exists()
            and STEADY_STEPS[-1] in (metrics := load_metrics(RUNS_DIR / f"{prefix}{layout}-{variant}"))
        }
        if not runs:
            continue
        print(f"\n{prefix}{layout} (steps {STEADY_STEPS.start} to {STEADY_STEPS.stop - 1})")
        header = (
            f"{'variant':<14} {'steps':>5} {'s/step (lower)':>14} {'tok/s/GPU (higher)':>18} "
            f"{'peak GiB (lower)':>16} {'paired ds/step vs fp32-cap2':>27} {'max |dloss| vs ref':>18}"
        )
        print(header)
        for variant, metrics in runs.items():
            summary = summarize(metrics)
            paired = paired_step_time_diff(runs["fp32-cap2"], metrics) if "fp32-cap2" in runs else float("nan")
            reference = runs.get(LOSS_REFERENCE[variant.split("-")[0]])
            loss_diff = max_loss_diff(reference, metrics) if reference is not None else float("nan")
            print(
                f"{variant:<14} {summary['steps']:>5} {summary['step_time']:>14.3f} {summary['tokens_per_gpu']:>18.0f} "
                f"{summary['peak_memory']:>16.1f} {paired:>+27.3f} {loss_diff:>18.2e}"
            )
            rows.append(
                {
                    "group": f"{prefix}{layout}",
                    "variant": variant,
                    "run_dir": str(RUNS_DIR / f"{prefix}{layout}-{variant}"),
                    "s_step": summary["step_time"],
                    "tok_s_gpu": summary["tokens_per_gpu"],
                    "peak_gib": summary["peak_memory"],
                    "paired_ds_vs_cap2": paired,
                    **traced_metrics(prefix, layout, variant),
                }
            )
    if args.csv:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
