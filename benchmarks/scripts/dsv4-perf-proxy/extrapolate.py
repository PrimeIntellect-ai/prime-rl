"""Extrapolate step time, FLOPs and MFU of a truncated model to its full layer count.

Two runs that differ only in `model.debug.num_layers` and see identical fake rows give the per-layer cost as the
per-step difference over the layer difference. Each step's FLOPs come from the logged MFU:
`perf/mfu / 100 * time/forward_backward * peak * world_size`.

Usage:
  uv run --no-sync python extrapolate.py --low <run dir> 4 --high <run dir> 6 --target-layers 43 \
      --peak 989e12 --world-size 8 --steady-start 4
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_steps(run_dir: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = defaultdict(dict)
    for line in (run_dir / "monitors/file/metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        steps[record["step"]].update(record)
    return steps


def step_values(steps: dict[int, dict[str, float]], peak: float, world_size: int) -> dict[int, dict[str, float]]:
    values = {}
    for step, record in steps.items():
        if "time/forward_backward" not in record or "perf/mfu" not in record:
            continue
        seconds = record["time/forward_backward"]
        values[step] = {
            "seconds": seconds,
            "flops": record["perf/mfu"] / 100 * seconds * peak * world_size,
            "tokens": record["perf/throughput"] * seconds,
            "mfu": record["perf/mfu"],
            "mfu_6dn": record["perf/mfu_6dn"],
        }
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--low", nargs=2, metavar=("RUN_DIR", "LAYERS"), required=True)
    parser.add_argument("--high", nargs=2, metavar=("RUN_DIR", "LAYERS"), required=True)
    parser.add_argument("--target-layers", type=int, required=True)
    parser.add_argument("--peak", type=float, required=True, help="dense BF16 peak FLOP/s per GPU")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--steady-start", type=int, default=1, help="first step included")
    parser.add_argument("--optimizer-seconds", type=float, default=0.0, help="subtracted from every step")
    args = parser.parse_args()

    low_layers, high_layers = int(args.low[1]), int(args.high[1])
    low = step_values(load_steps(Path(args.low[0])), args.peak, args.world_size)
    high = step_values(load_steps(Path(args.high[0])), args.peak, args.world_size)
    shared = sorted(step for step in low.keys() & high.keys() if step >= args.steady_start)
    if not shared:
        raise ValueError("The runs share no steps at or after --steady-start")

    rows = []
    for step in shared:
        if abs(low[step]["tokens"] - high[step]["tokens"]) > 1e-3 * low[step]["tokens"]:
            raise ValueError(f"Step {step} has different token counts: the runs did not see identical rows")
        layer_seconds = (high[step]["seconds"] - low[step]["seconds"]) / (high_layers - low_layers)
        layer_flops = (high[step]["flops"] - low[step]["flops"]) / (high_layers - low_layers)
        extra_layers = args.target_layers - low_layers
        seconds = low[step]["seconds"] + extra_layers * layer_seconds - args.optimizer_seconds
        flops = low[step]["flops"] + extra_layers * layer_flops
        rows.append(
            {
                "step": step,
                "tokens": low[step]["tokens"],
                "layer_seconds": layer_seconds,
                "target_seconds": seconds,
                "target_mfu": 100 * flops / (seconds * args.peak * args.world_size),
                "target_tok_s_gpu": low[step]["tokens"] / seconds / args.world_size,
                "low_mfu": low[step]["mfu"],
                "high_mfu": high[step]["mfu"],
                "high_mfu_6dn": high[step]["mfu_6dn"],
            }
        )

    print("| step | tokens | s per added layer | target s | target MFU % | target tok/s/GPU |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['step']} | {row['tokens']:.0f} | {row['layer_seconds']:.3f} | {row['target_seconds']:.2f} "
            f"| {row['target_mfu']:.2f} | {row['target_tok_s_gpu']:.0f} |"
        )
    print()
    for key in ["layer_seconds", "target_seconds", "target_mfu", "target_tok_s_gpu", "low_mfu", "high_mfu", "high_mfu_6dn"]:
        series = [row[key] for row in rows]
        print(f"{key}: median {statistics.median(series):.4g}, min {min(series):.4g}, max {max(series):.4g}")


if __name__ == "__main__":
    main()
