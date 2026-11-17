#!/usr/bin/env python3
"""Compare NVFP4 ablation runs from their file-monitor metrics.

Each argument is a run's output directory (any directory holding ``monitors/file/metrics.jsonl`` below it). Prints,
per run: how far it got, the trainer-sampler mismatch KL, the reward early and late, grad norm, entropy and step
time, with the mismatch and step time relative to the first run (the baseline).

    python benchmarks/nvfp4/compare.py outputs/nvfp4-bf16-replay outputs/nvfp4-nvfp4 outputs/nvfp4-nvfp4-no-replay
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REWARD = "train/agg/all/agent/reward/mean"
MISMATCH = "mismatch_kl/all/mean"


def load(run: Path) -> dict[int, dict[str, float]]:
    """Metrics per step, merged over the run's rows (trainer and orchestrator log separately)."""
    paths = sorted(run.glob("**/monitors/file/metrics.jsonl"))
    if not paths:
        raise SystemExit(f"{run}: no monitors/file/metrics.jsonl")
    steps: dict[int, dict[str, float]] = {}
    for path in paths:
        for line in path.open():
            row = json.loads(line)
            step = row.get("step")
            if step is None:
                continue
            steps.setdefault(int(step), {}).update(
                {k: v for k, v in row.items() if isinstance(v, (int, float)) and v == v}
            )
    return steps


def series(steps: dict[int, dict[str, float]], key: str, start: int = 0, end: int | None = None) -> list[float]:
    return [m[key] for s, m in sorted(steps.items()) if key in m and s >= start and (end is None or s < end)]


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(steps: dict[int, dict[str, float]], skip: int, window: int) -> dict[str, float | None]:
    last = max(steps) if steps else 0
    return {
        "steps": last,
        "mismatch_kl": median(series(steps, MISMATCH, start=skip)),
        "mismatch_kl_max": max(series(steps, MISMATCH, start=skip), default=None),
        "reward_first": median(series(steps, REWARD, end=window)),
        "reward_last": median(series(steps, REWARD, start=last - window + 1)),
        "grad_norm": median(series(steps, "optim/grad_norm", start=skip)),
        "entropy": median(series(steps, "entropy/all/mean", start=skip)),
        "step_s": median(series(steps, "time/step", start=skip)),
        "fwd_bwd_s": median(series(steps, "time/forward_backward", start=skip)),
    }


def fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run output directories; the first is the baseline")
    parser.add_argument("--skip", type=int, default=5, help="warm-up steps left out of the medians")
    parser.add_argument("--window", type=int, default=20, help="steps in the early and late reward medians")
    args = parser.parse_args()

    rows = [(run.name, summarize(load(run), args.skip, args.window)) for run in args.runs]
    base = rows[0][1]
    header = f"{'run':<24} {'steps':>5} {'mismatch_kl':>11} {'vs base':>7} {'kl max':>8} {'reward':>15} "
    header += f"{'grad_norm':>9} {'entropy':>8} {'fwd_bwd s':>9} {'step s':>7} {'speedup':>7}"
    print(header)
    for name, row in rows:
        kl_ratio = row["mismatch_kl"] / base["mismatch_kl"] if row["mismatch_kl"] and base["mismatch_kl"] else None
        speedup = base["step_s"] / row["step_s"] if row["step_s"] and base["step_s"] else None
        reward = f"{fmt(row['reward_first'], 3)} -> {fmt(row['reward_last'], 3)}"
        print(
            f"{name:<24} {row['steps']:>5} {fmt(row['mismatch_kl']):>11} {fmt(kl_ratio, 3):>7} "
            f"{fmt(row['mismatch_kl_max'], 3):>8} {reward:>15} {fmt(row['grad_norm'], 3):>9} "
            f"{fmt(row['entropy'], 3):>8} {fmt(row['fwd_bwd_s'], 3):>9} {fmt(row['step_s'], 3):>7} "
            f"{fmt(speedup, 3):>7}"
        )


if __name__ == "__main__":
    main()
