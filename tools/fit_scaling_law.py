"""Fit a scaling ladder's W&B losses and project the loss trajectory of larger rungs.

Rungs come from ``ladder.json`` (written by ``tools/scaling_ladder.py``); each rung's W&B run is
found by its name. All rungs share one recipe and schedule shape, so at every fraction ``f`` of
training the finished rungs' losses are fitted as ``L_f(C) = E + A * C^-alpha`` over the rungs'
training compute ``C``. Evaluating each fit at an unfinished rung's compute gives its projected
loss trajectory. See docs/scaling-ladder.md.

Usage:
    uv run python tools/fit_scaling_law.py <output_dir>/ladder.json <entity/project> [--metric val/loss/<source>]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import wandb

from prime_rl.utils.scaling import fit_power_law


def find_run(api: wandb.Api, path: str, name: str) -> "wandb.apis.public.Run | None":
    runs = api.runs(path, filters={"display_name": name})
    return runs[0] if len(runs) else None


def fetch_history(run: "wandb.apis.public.Run | None", metric: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(run.scan_history(keys=["step", metric])) if run else []
    return np.array([row["step"] for row in rows]), np.array([row[metric] for row in rows])


def loss_at(steps: np.ndarray, losses: np.ndarray, step: float, window: float) -> float:
    """Mean loss over ``(step - window, step]``, or the logged loss nearest to ``step``."""
    in_window = (steps > step - window) & (steps <= step)
    if in_window.any():
        return float(losses[in_window].mean())
    return float(losses[np.abs(steps - step).argmin()])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ladder", type=Path, help="ladder.json from tools/scaling_ladder.py")
    parser.add_argument("wandb_path", help="W&B entity/project of the ladder runs")
    parser.add_argument(
        "--metric", help="W&B loss key; defaults to val/loss if every run logs it, else the train loss/mean"
    )
    parser.add_argument("--points", type=int, default=20, help="fractions of training to fit")
    parser.add_argument("--window", type=float, default=0.01, help="smoothing window, as a fraction of the steps")
    args = parser.parse_args()

    rungs = json.loads(args.ladder.read_text())
    api = wandb.Api()
    runs = {r["name"]: find_run(api, args.wandb_path, r["name"]) for r in rungs}
    found = [run for run in runs.values() if run]
    metric = args.metric or ("val/loss" if found and all("val/loss" in run.summary for run in found) else "loss/mean")
    histories = {name: fetch_history(run, metric) for name, run in runs.items()}
    finished = [r for r in rungs if runs[r["name"]] and runs[r["name"]].summary.get("step", 0) >= r["steps"]]
    targets = [r for r in rungs if r not in finished]
    if len(finished) < 3:
        raise SystemExit(f"need at least 3 finished rungs to fit, found {[r['name'] for r in finished]}")
    print(f"fit {metric} on: {', '.join(r['name'] for r in finished)}")

    compute = np.array([r["flops"] for r in finished])
    fractions = np.linspace(1 / args.points, 1, args.points)
    fits = []
    for f in fractions:
        losses = [loss_at(*histories[r["name"]], f * r["steps"], args.window * r["steps"]) for r in finished]
        fits.append(fit_power_law(compute, np.array(losses)))
    floor, coeff, alpha = fits[-1]
    print(f"final loss = {floor:.4f} + {coeff:.4g} * C^-{alpha:.4f}\n")

    for r in targets:
        steps, losses = histories[r["name"]]
        print(f"{r['name']}: {r['flops']:.2e} FLOPs, {r['tokens']:.3g} tokens")
        print(f"{'fraction':>8} {'step':>8} {'tokens':>10} {'projected':>9} {'observed':>9}")
        for f, (floor, coeff, alpha) in zip(fractions, fits):
            step = f * r["steps"]
            observed = (
                f"{loss_at(steps, losses, step, args.window * r['steps']):.4f}"
                if len(steps) and steps.max() >= step
                else ""
            )
            projected = floor + coeff * r["flops"] ** -alpha
            print(f"{f:>8.2f} {round(step):>8} {f * r['tokens']:>10.3g} {projected:>9.4f} {observed:>9}")
        print()


if __name__ == "__main__":
    main()
