"""Markdown tables of per-block metric means for the add-model PR report.

Reads `<run_dir>/monitors/file/metrics.jsonl` and prints one table per run: a row per block of
steps, a column per metric. Columns are the first key matching each pattern; patterns without a
match are skipped.

Usage (from the prime-rl repo):
    uv run python skills/add-model/report.py <run_dir> [<run_dir> ...] [--block 10] [--metric REGEX ...]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_METRICS = [
    r"^mismatch_kl/all/mean$",
    r"^mismatch_kl/all/max$",
    r"^entropy/all/mean$",
    r"^is_masked/mean$",
    r"^train/agg/effective/[^/]+/reward/mean$",
    r"^train/agg/effective/num_turns/mean$",
    r"^train/agg/effective/[^/]+/is_truncated/mean$",
    r"^eval/[^/]+/effective/[^/]+/avg@\d+$",
    r"^perf/mfu$",
]


def load_steps(run_dir: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = defaultdict(dict)
    for line in (run_dir / "monitors/file/metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("step") is not None:
            steps[row["step"]].update({k: v for k, v in row.items() if isinstance(v, (int, float))})
    return steps


def table(steps: dict[int, dict[str, float]], patterns: list[str], block: int) -> str:
    keys = sorted({key for values in steps.values() for key in values})
    columns = [next(key for key in keys if re.search(p, key)) for p in patterns if any(re.search(p, k) for k in keys)]
    lines = ["| steps | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    for start in range(min(steps), max(steps) + 1, block):
        rows = [steps[s] for s in range(start, start + block) if s in steps]
        cells = []
        for column in columns:
            values = [row[column] for row in rows if column in row]
            cells.append(f"{sum(values) / len(values):.4g}" if values else "")
        lines.append(f"| {start}-{start + block - 1} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--metric", action="append", help="Regex of a metric key; repeat. Replaces the defaults.")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        print(f"### {run_dir.name}\n\n{table(load_steps(run_dir), args.metric or DEFAULT_METRICS, args.block)}\n")


if __name__ == "__main__":
    main()
