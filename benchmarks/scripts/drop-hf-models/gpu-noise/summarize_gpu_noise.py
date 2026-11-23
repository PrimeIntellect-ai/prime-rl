"""Per-tensor gradient envelopes of repeated GPU steps, and how far candidate runs land outside the baseline's."""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

STATS = re.compile(r"^\S+ (?P<shape>\[[^\]]*\]) \S+ sum=(?P<sum>\S+) norm=(?P<norm>\S+)$")


def load_runs(directory: Path) -> dict[str, dict[str, list[dict]]]:
    runs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(directory.glob("*.json")):
        run = json.loads(path.read_text())
        for label in (run["worktree"], f"{run['worktree']}-gpu{run['gpu']}"):
            runs[label][run["model"]].append(run["record"])
    return runs


def tensor_stats(records: list[dict]) -> dict[str, dict[str, list[float]]]:
    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"sum": [], "norm": [], "numel": []})
    for record in records:
        for name, value in record["grads"].items():
            match = STATS.match(value)
            if match is None:
                continue
            stats[name]["sum"].append(float(match["sum"]))
            stats[name]["norm"].append(float(match["norm"]))
            stats[name]["numel"] = [math.prod(json.loads(match["shape"]))]
    return stats


def worst_spread(stats: dict[str, dict[str, list[float]]]) -> tuple[float, str, float, str]:
    """Largest norm range over the mean norm, and largest sum range over mean norm times the square root of numel."""
    worst_norm, worst_sum = (0.0, ""), (0.0, "")
    for name, values in stats.items():
        norm = sum(values["norm"]) / len(values["norm"])
        if norm == 0:
            continue
        norm_spread = (max(values["norm"]) - min(values["norm"])) / norm
        sum_spread = (max(values["sum"]) - min(values["sum"])) / (norm * math.sqrt(values["numel"][0]))
        worst_norm = max(worst_norm, (norm_spread, name))
        worst_sum = max(worst_sum, (sum_spread, name))
    return (*worst_norm, *worst_sum)


def outside_distance(value: float, low: float, high: float) -> float:
    """How far `value` lies outside [low, high], in units of the range, 0 inside it."""
    excess = max(value - high, low - value, 0.0)
    if excess == 0:
        return 0.0
    return excess / (high - low) if high > low else math.inf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, help="One JSON per run with worktree, model and record keys")
    parser.add_argument(
        "baseline", help="Label of the baseline runs: a worktree, or a worktree and GPU as <worktree>-gpu<N>"
    )
    parser.add_argument("candidate", help="Label of the candidate runs, in the same form")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    for model in sorted(runs[args.baseline]):
        baseline, candidate = runs[args.baseline][model], runs[args.candidate].get(model, [])
        outputs = {(record["logits"], record["loss"]) for record in baseline + candidate}
        print(f"{model}: {len(baseline)} {args.baseline} runs, {len(candidate)} {args.candidate} runs")
        print(f"  logits and loss bitwise identical across all runs: {len(outputs) == 1}")
        baseline_stats, candidate_stats = tensor_stats(baseline), tensor_stats(candidate)
        for label, stats in ((args.baseline, baseline_stats), (args.candidate, candidate_stats)):
            if stats:
                norm_spread, norm_name, sum_spread, sum_name = worst_spread(stats)
                print(f"  {label} worst norm spread {norm_spread:.3e} ({norm_name})")
                print(f"  {label} worst sum spread {sum_spread:.3e} ({sum_name})")
        outside = []
        for name, values in candidate_stats.items():
            envelope = baseline_stats[name]
            for stat in ("sum", "norm"):
                low, high = min(envelope[stat]), max(envelope[stat])
                distance = max(outside_distance(value, low, high) for value in values[stat])
                if distance > 0:
                    outside.append((distance, name, stat, low, high, min(values[stat]), max(values[stat])))
        checks = 2 * len(candidate_stats)
        print(
            f"  {args.candidate} values outside the {args.baseline} envelope: {len(outside)} of {checks} tensor stats"
        )
        for distance, name, stat, low, high, candidate_low, candidate_high in sorted(outside, reverse=True)[:10]:
            print(
                f"    {name} {stat}: {args.baseline} [{low:.6g}, {high:.6g}], "
                f"{args.candidate} [{candidate_low:.6g}, {candidate_high:.6g}], {distance:.2f} ranges outside"
            )


if __name__ == "__main__":
    main()
