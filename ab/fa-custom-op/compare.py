"""Step time, throughput, memory and loss-parity summary of the FA2/FA3 custom op A/B runs."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path.home() / "tmp/fa-custom-op/runs"
STEADY_STEPS = range(5, 21)
COMPARISONS = {
    "bf16 router, no fullgraph": ["main-bf16", "main-bf16-r2", "main-bf16-r3", "branch-bf16", "branch-bf16-r2", "branch-bf16-r3"],
    "bf16 router, fullgraph (branch only)": ["main-bf16", "branch-bf16", "branch-bf16-fg", "branch-bf16-fg-r2", "branch-bf16-fg-warm40", "branch-bf16-fg-fa2"],
    "fp32 router, no fullgraph": ["main-fp32", "branch-fp32"],
}


def load_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    by_step = defaultdict(dict)
    for line in (run_dir / "monitors/file/metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        by_step[record["step"]].update(record)
    return dict(by_step)


def summarize(metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    steps = [step for step in STEADY_STEPS if step in metrics]
    return {
        "steps": max(metrics),
        "step_time": statistics.median(metrics[step]["time/step"] for step in steps),
        "tokens_per_gpu": statistics.median(metrics[step]["perf/throughput_per_gpu"] for step in steps),
        "peak_memory": max(record["perf/peak_memory"] for record in metrics.values() if "perf/peak_memory" in record),
    }


def max_loss_diff(a: dict, b: dict) -> float:
    return max(abs(b[step]["loss/mean"] - a[step]["loss/mean"]) for step in set(a) & set(b))


def main():
    for title, variants in COMPARISONS.items():
        runs = {
            variant: load_metrics(RUNS_DIR / variant)
            for variant in variants
            if (RUNS_DIR / variant / "monitors/file/metrics.jsonl").exists()
        }
        runs = {variant: metrics for variant, metrics in runs.items() if metrics}
        if not runs:
            continue
        reference = variants[0]
        print(f"\n{title} (loss reference: {reference})")
        print(
            f"{'variant':<20} {'last step':>9} {'s/step (lower)':>14} {'tok/s/GPU (higher)':>18} "
            f"{'peak GiB (lower)':>16} {'max |dloss| vs ref':>18}"
        )
        for variant, metrics in runs.items():
            summary = summarize(metrics)
            loss_diff = max_loss_diff(runs[reference], metrics) if reference in runs else float("nan")
            print(
                f"{variant:<20} {summary['steps']:>9} {summary['step_time']:>14.3f} "
                f"{summary['tokens_per_gpu']:>18.0f} {summary['peak_memory']:>16.1f} {loss_diff:>18.3f}"
            )


if __name__ == "__main__":
    main()
