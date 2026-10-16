"""Collect and summarize repeated eval runs for the nightly eval regression workflow.

    python scripts/eval_regression.py collect <run_dir> --repeat 1 --wall-s 312.4 --out result.json
    python scripts/eval_regression.py summarize results/*.json >> "$GITHUB_STEP_SUMMARY"

Stdlib only, so the summarize job needs no `uv sync`.
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Two-sided 95% Student-t quantiles by degrees of freedom. Nightly runs use a
# handful of repeats, where the normal quantile (1.96) understates the interval.
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def t95(df: int) -> float:
    return T95.get(df, 1.96)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def collect(run_dir: Path, repeat: int, wall_s: float | None) -> dict:
    index = run_dir / "monitors" / "file" / "traces" / "stream.index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("kind") == "eval"]
    if not rows:
        raise SystemExit(f"no eval episodes in {index}")

    # Rollouts of one task share a group, so the standard error clusters by group.
    rewards_by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        rewards_by_group[row["group"]].append(float(row["reward"] or 0.0))
    task_means = [statistics.fmean(rewards) for rewards in rewards_by_group.values()]
    sem = statistics.stdev(task_means) / math.sqrt(len(task_means)) if len(task_means) > 1 else 0.0

    durations = [row["duration"] for row in rows]
    return {
        "repeat": repeat,
        "env": rows[0]["env"],
        "episodes": len(rows),
        "tasks": len(task_means),
        "accuracy": statistics.fmean(task_means),
        "accuracy_sem": sem,
        "error_rate": statistics.fmean(0.0 if row["ok"] else 1.0 for row in rows),
        "truncation_rate": statistics.fmean(1.0 if row["truncated"] else 0.0 for row in rows),
        "output_tokens_mean": statistics.fmean(row["output_tokens"] for row in rows),
        "episode_p50_s": quantile(durations, 0.5),
        "episode_p90_s": quantile(durations, 0.9),
        "eval_window_s": max(row["arrival"] for row in rows) - min(row["dispatch"] for row in rows),
        "wall_s": wall_s,
    }


# (key, label, format, lower_is_better)
METRICS = [
    ("accuracy", "Accuracy", "{:.2%}", False),
    ("wall_s", "Time to finish (s)", "{:.0f}", True),
    ("eval_window_s", "Eval window (s)", "{:.0f}", True),
    ("episode_p50_s", "Episode p50 (s)", "{:.1f}", True),
    ("episode_p90_s", "Episode p90 (s)", "{:.1f}", True),
    ("output_tokens_mean", "Output tokens (mean)", "{:.0f}", True),
    ("error_rate", "Error rate", "{:.2%}", True),
    ("truncation_rate", "Truncation rate", "{:.2%}", True),
]


def fmt(value: float | None, spec: str) -> str:
    return "–" if value is None else spec.format(value)


def summarize(results: list[dict], expected: int | None) -> str:
    results = sorted(results, key=lambda r: r["repeat"])
    lines = ["## Eval regression", ""]
    if not results:
        lines.append("❌ No repeat finished. Check the job logs.")
        return "\n".join(lines) + "\n"

    first = results[0]
    k = len(results)
    status = "✅" if expected is None or k == expected else "⚠️"
    lines += [
        f"{status} **{first['env']}** · {k}{'' if expected is None else f'/{expected}'} repeats finished"
        f" · {first['tasks']} tasks × {first['episodes'] // first['tasks']} rollouts per repeat",
        "",
        "### Across repeats",
        "",
        "Mean ± 95% CI of the mean (Student-t), with the run-to-run standard deviation.",
        "",
        "| Metric | Mean | ± 95% CI | Std | Min | Max | CV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label, spec, _ in METRICS:
        values = [r[key] for r in results if r.get(key) is not None]
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        ci = t95(len(values) - 1) * std / math.sqrt(len(values)) if len(values) > 1 else None
        cv = f"{std / mean:.1%}" if mean else "–"
        lines.append(
            f"| {label} | {fmt(mean, spec)} | {fmt(ci, spec)} | {fmt(std, spec)}"
            f" | {fmt(min(values), spec)} | {fmt(max(values), spec)} | {cv} |"
        )

    # If the observed spread is much larger than the per-run SEM predicts, the
    # noise comes from the serving side (or nondeterminism), not task sampling.
    mean_sem = statistics.fmean(r["accuracy_sem"] for r in results)
    observed = statistics.stdev(r["accuracy"] for r in results) if k > 1 else None
    lines += [
        "",
        f"Within-run accuracy SEM (task sampling): **±{mean_sem:.2%}** · "
        f"observed run-to-run std: **{fmt(observed, '±{:.2%}')}**",
        "",
        "<details><summary>Per repeat</summary>",
        "",
        "| Repeat | Accuracy | ± SEM | Time to finish (s) | Eval window (s) | Episode p50 / p90 (s) | Errors | Truncated |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['repeat']} | {r['accuracy']:.2%} | {r['accuracy_sem']:.2%} | {fmt(r['wall_s'], '{:.0f}')}"
            f" | {r['eval_window_s']:.0f} | {r['episode_p50_s']:.1f} / {r['episode_p90_s']:.1f}"
            f" | {r['error_rate']:.2%} | {r['truncation_rate']:.2%} |"
        )
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Reduce one eval run dir to a result JSON.")
    p_collect.add_argument("run_dir", type=Path)
    p_collect.add_argument("--repeat", type=int, required=True)
    p_collect.add_argument("--wall-s", type=float, default=None, help="Wall-clock seconds of the eval process.")
    p_collect.add_argument("--out", type=Path, required=True)

    p_summarize = sub.add_parser("summarize", help="Render result JSONs as a Markdown summary.")
    p_summarize.add_argument("results", type=Path, nargs="*")
    p_summarize.add_argument("--expected", type=int, default=None, help="Number of repeats launched.")

    args = parser.parse_args()
    if args.cmd == "collect":
        result = collect(args.run_dir, args.repeat, args.wall_s)
        args.out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    else:
        results = [json.loads(path.read_text()) for path in args.results]
        sys.stdout.write(summarize(results, args.expected))


if __name__ == "__main__":
    main()
