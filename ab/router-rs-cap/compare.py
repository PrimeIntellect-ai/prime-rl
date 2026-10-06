"""Per-variant throughput, memory and loss-parity summary of the reduce-scatter buffer cap A/B runs."""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path.home() / "tmp/router-rs-cap/runs"
LAYOUTS = ["fsdp16", "ep8"]
VARIANTS = ["fp32-cap1", "fp32-cap2", "fp32-cap2-r2", "fp32-main", "fp32-cap3", "bf16-cap1", "bf16-cap2"]
STEADY_STEPS = range(5, 21)
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


def main():
    variants = sys.argv[1:] or VARIANTS
    for layout in LAYOUTS:
        runs = {
            variant: load_metrics(RUNS_DIR / f"{layout}-{variant}")
            for variant in variants
            if (RUNS_DIR / f"{layout}-{variant}/monitors/file/metrics.jsonl").exists()
        }
        if not runs:
            continue
        print(f"\n{layout}")
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


if __name__ == "__main__":
    main()
