"""Compare the DSv4 dsa_backend A/B runs that `launch.sh` wrote: steady-state speed per arm and per-step parity per pair.

usage (any node; reads metrics only):
  uv run --no-sync python benchmarks/scripts/dsv4_cudnn_flashmla_e2e/compare.py SHA [--runs-dir DIR] [--warmup N]

Speed: median, min and max of `time/step`, `perf/throughput_per_gpu`, `perf/mfu` and `perf/peak_memory` over the
steps after the first `--warmup` (compile and JIT warmup). Parity: per-step absolute differences of `loss/mean` and
`optim/grad_norm` for each pair, so the cudnn_flashmla gap can be read against the same-code repeat's.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_RUNS_DIR = Path("/home/garrett/prl_output_dir/dsv4-cfm-e2e")
ARMS = ["tilelang", "cudnn_flashmla", "tilelang-r2", "cudnn_flashmla-r2"]
PAIRS = [("tilelang", "cudnn_flashmla"), ("tilelang", "tilelang-r2"), ("cudnn_flashmla", "cudnn_flashmla-r2")]
SPEED = [
    ("time/step", "step s", "lower"),
    ("perf/throughput_per_gpu", "tok/s/GPU", "higher"),
    ("perf/mfu", "MFU %", "higher"),
    ("perf/peak_memory", "peak GiB", "lower"),
]
PARITY = ["loss/mean", "optim/grad_norm"]


def load_metrics(run_dir: Path) -> dict[int, dict[str, float]]:
    by_step = defaultdict(dict)
    for line in (run_dir / "monitors/file/metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("step") is not None:
            by_step[record["step"]].update(record)
    return dict(by_step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sha")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--warmup", type=int, default=5, help="leading steps excluded from the speed medians")
    args = parser.parse_args()

    runs = {}
    for arm in ARMS:
        run_dir = args.runs_dir / f"dsv4-flash-6L-sft-65k-cp8-ep8-{arm}-{args.sha}"
        if (run_dir / "monitors/file/metrics.jsonl").exists():
            runs[arm] = load_metrics(run_dir)

    print(f"Steady state = steps after the first {args.warmup}; median (min-max).\n")
    print("| arm | steps | " + " | ".join(f"{label} ({better} is better)" for _, label, better in SPEED) + " |")
    print("|---|---|" + "---|" * len(SPEED))
    for arm, metrics in runs.items():
        steady = [step for step in sorted(metrics) if step > args.warmup]
        cells = []
        for key, _, _ in SPEED:
            values = [metrics[step][key] for step in steady if key in metrics[step]]
            cells.append(f"{statistics.median(values):.4g} ({min(values):.4g}-{max(values):.4g})" if values else "-")
        print(f"| {arm} | {len(steady)} | " + " | ".join(cells) + " |")

    print("\nPer-step parity: max |difference| over common steps, and the step it occurs at.\n")
    print("| pair | steps | " + " | ".join(f"max abs d {key}" for key in PARITY) + " |")
    print("|---|---|" + "---|" * len(PARITY))
    for left, right in PAIRS:
        if left not in runs or right not in runs:
            continue
        steps = sorted(set(runs[left]) & set(runs[right]))
        cells = []
        for key in PARITY:
            diffs = [
                (abs(runs[left][step][key] - runs[right][step][key]), step)
                for step in steps
                if key in runs[left][step] and key in runs[right][step]
            ]
            worst, at = max(diffs) if diffs else (float("nan"), "-")
            cells.append(f"{worst:.3g} (step {at})")
        print(f"| {left} vs {right} | {len(steps)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
