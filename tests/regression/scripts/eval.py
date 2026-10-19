"""Collect, summarize and gate runs of the eval suite in the regression workflow.

    python tests/regression/scripts/eval.py collect <run_dir> --repeat 1 --wall-s 312 --out result.json
    python tests/regression/scripts/eval.py summarize results/*.json --baseline baselines/eval/gsm8k.json
    python tests/regression/scripts/eval.py baseline results/*.json \
        --out baselines/eval/gsm8k.json --report reports/eval/gsm8k.md

`summarize` compares the results against the baseline and exits 1 on a regression.
`baseline` turns two or more results into a new baseline and its report.

Stdlib only, so the summary job needs no `uv sync`.
"""

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Two-sided 95% Student-t quantiles by degrees of freedom. Baselines use a
# handful of repeats, where the normal quantile (1.96) understates the interval.
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}

# A run regresses when a gated metric is worse than the baseline mean by more
# than max(K_SIGMA * σ_pred, floor_abs, floor_rel * |mean|). σ_pred is the
# spread to expect for a fresh run against a mean of n baseline runs:
# std * sqrt(1 + 1/n). The floors keep a lucky low-variance baseline from
# turning every nightly red.
K_SIGMA = 3.0


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    spec: str
    higher_is_better: bool | None = None
    """None shows the metric without gating on it."""
    floor_abs: float = 0.0
    floor_rel: float = 0.0


METRICS = [
    Metric("accuracy", "Accuracy", "{:.2%}", True, floor_abs=0.01),
    Metric("wall_s", "Time to finish (s)", "{:.0f}", False, floor_rel=0.25),
    Metric("eval_window_s", "Eval window (s)", "{:.0f}"),
    Metric("episode_p50_s", "Episode p50 (s)", "{:.1f}"),
    Metric("episode_p90_s", "Episode p90 (s)", "{:.1f}"),
    Metric("output_tokens_mean", "Output tokens (mean)", "{:.0f}"),
    Metric("error_rate", "Error rate", "{:.2%}", False, floor_abs=0.01),
    Metric("truncation_rate", "Truncation rate", "{:.2%}", False, floor_abs=0.01),
]

# A baseline only applies to runs of the same eval.
IDENTITY = ("env", "model", "tasks", "episodes")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def fmt(value: float | None, spec: str) -> str:
    return "–" if value is None else spec.format(value)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def collect(run_dir: Path, repeat: int, wall_s: float | None) -> dict:
    index = run_dir / "monitors" / "file" / "traces" / "stream.index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("kind") == "eval"]
    if not rows:
        raise SystemExit(f"no eval episodes in {index}")
    resolved = sorted(run_dir.glob("configs/attempt_*/resolved/eval.json"))[-1]

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
        "model": json.loads(resolved.read_text())["model"],
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


def compare(results: list[dict], baseline: dict) -> tuple[list[str], bool]:
    """Render the comparison table against the baseline. Returns (lines, regressed)."""
    base = baseline["results"]
    n = len(base)
    lines = [
        "| Metric | Tonight | Baseline mean ± std | Δ | Tolerance | Status |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    regressed = False
    for m in METRICS:
        values = [r[m.key] for r in results if r.get(m.key) is not None]
        base_values = [r[m.key] for r in base if r.get(m.key) is not None]
        if not values or not base_values:
            continue
        value = statistics.fmean(values)
        base_mean, base_std = mean_std(base_values)
        delta = value - base_mean
        if m.higher_is_better is None:
            tol, status = None, "–"
        else:
            tol = max(K_SIGMA * base_std * math.sqrt(1 + 1 / n), m.floor_abs, m.floor_rel * abs(base_mean))
            worse = -delta if m.higher_is_better else delta
            if worse > tol:
                status, regressed = "❌", True
            elif -worse > tol:
                status = "🟢"
            else:
                status = "✅"
        sign = "+" if delta >= 0 else "−"
        lines.append(
            f"| {m.label} | {fmt(value, m.spec)} | {fmt(base_mean, m.spec)} ± {fmt(base_std, m.spec)}"
            f" | {sign}{fmt(abs(delta), m.spec)} | {fmt(tol, '±' + m.spec)} | {status} |"
        )
    return lines, regressed


def spread(results: list[dict]) -> list[str]:
    """Render mean, 95% CI and spread across repeats."""
    lines = [
        "| Metric | Mean | ± 95% CI | Std | Min | Max | CV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in METRICS:
        values = [r[m.key] for r in results if r.get(m.key) is not None]
        if not values:
            continue
        mean, std = mean_std(values)
        ci = T95.get(len(values) - 1, 1.96) * std / math.sqrt(len(values))
        cv = f"{std / mean:.1%}" if mean else "–"
        lines.append(
            f"| {m.label} | {fmt(mean, m.spec)} | {fmt(ci, m.spec)} | {fmt(std, m.spec)}"
            f" | {fmt(min(values), m.spec)} | {fmt(max(values), m.spec)} | {cv} |"
        )
    mean_sem = statistics.fmean(r["accuracy_sem"] for r in results)
    observed = statistics.stdev(r["accuracy"] for r in results)
    lines += [
        "",
        f"Within-run accuracy SEM (task sampling): **±{mean_sem:.2%}** · observed run-to-run std: **±{observed:.2%}**",
    ]
    return lines


def per_repeat(results: list[dict]) -> list[str]:
    lines = [
        "| Repeat | Accuracy | ± SEM | Time to finish (s) | Eval window (s) | Episode p50 / p90 (s) | Errors | Truncated |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['repeat']} | {r['accuracy']:.2%} | {r['accuracy_sem']:.2%} | {fmt(r['wall_s'], '{:.0f}')}"
            f" | {r['eval_window_s']:.0f} | {r['episode_p50_s']:.1f} / {r['episode_p90_s']:.1f}"
            f" | {r['error_rate']:.2%} | {r['truncation_rate']:.2%} |"
        )
    return lines


def summarize(results: list[dict], expected: int | None, baseline: dict | None) -> tuple[str, bool]:
    """Render the Markdown summary. Returns (markdown, regressed)."""
    lines = ["## Eval regression", ""]
    if not results:
        lines.append("❌ No repeat finished. Check the job logs.")
        return "\n".join(lines) + "\n", True

    results = sorted(results, key=lambda r: r["repeat"])
    first, k = results[0], len(results)
    lines += [
        f"**{first['env']}** · `{first['model']}` · {k}{'' if expected is None else f'/{expected}'} repeats finished"
        f" · {first['tasks']} tasks × {first['episodes'] // first['tasks']} rollouts",
        "",
    ]
    if expected is not None and k < expected:
        lines += [f"⚠️ {expected - k} repeats failed. Check the job logs.", ""]

    regressed = False
    if baseline is None:
        lines += ["⚠️ No baseline. Skipped the regression check.", ""]
    elif mismatch := [key for key in IDENTITY if first.get(key) != baseline["results"][0].get(key)]:
        lines += [
            f"⚠️ The baseline is for a different eval ({', '.join(mismatch)} differ). Skipped the regression check.",
            "",
        ]
    else:
        table, regressed = compare(results, baseline)
        verdict = "❌ **Regression detected**" if regressed else "✅ No regression"
        lines += [
            f"### {verdict}",
            "",
            f"Baseline: {len(baseline['results'])} repeats from [{baseline['date']}]({baseline['source']})."
            f" Tolerance: max({K_SIGMA:g}σ of a single run, metric floor).",
            "",
            *table,
            "",
        ]

    if k > 1:
        lines += ["### Across repeats", "", "Mean ± 95% CI of the mean (Student-t).", "", *spread(results), ""]
    lines += ["<details><summary>Per repeat</summary>", "", *per_repeat(results), "", "</details>"]
    return "\n".join(lines) + "\n", regressed


def make_baseline(results: list[dict]) -> dict:
    server, repo, run_id = (os.environ.get(v) for v in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": f"{server}/{repo}/actions/runs/{run_id}" if run_id else "local",
        "commit": os.environ.get("GITHUB_SHA"),
        "results": sorted(results, key=lambda r: r["repeat"]),
    }


def report(baseline: dict) -> str:
    """Render the error bars of a baseline as Markdown."""
    results = baseline["results"]
    first = results[0]
    lines = [
        f"# Eval baseline: {first['env']}",
        "",
        f"`{first['model']}` · {first['tasks']} tasks × {first['episodes'] // first['tasks']} rollouts"
        f" · {len(results)} repeats from [{baseline['date']}]({baseline['source']})"
        + (f" at `{baseline['commit'][:9]}`" if baseline.get("commit") else ""),
        "",
        "Mean ± 95% CI of the mean (Student-t).",
        "",
        *spread(results),
        "",
        "## Per repeat",
        "",
        *per_repeat(results),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Reduce one eval run dir to a result JSON.")
    p_collect.add_argument("run_dir", type=Path)
    p_collect.add_argument("--repeat", type=int, required=True)
    p_collect.add_argument("--wall-s", type=float, default=None, help="Wall-clock seconds of the eval process.")
    p_collect.add_argument("--out", type=Path, required=True)

    p_summarize = sub.add_parser("summarize", help="Render result JSONs as Markdown and gate on the baseline.")
    p_summarize.add_argument("results", type=Path, nargs="*")
    p_summarize.add_argument("--expected", type=int, default=None, help="Number of repeats launched.")
    p_summarize.add_argument("--baseline", type=Path, default=None, help="Baseline file to compare against.")

    p_baseline = sub.add_parser("baseline", help="Turn 2+ result JSONs into a baseline and its report.")
    p_baseline.add_argument("results", type=Path, nargs="+")
    p_baseline.add_argument("--out", type=Path, required=True, help="Baseline JSON to write.")
    p_baseline.add_argument("--report", type=Path, required=True, help="Markdown report to write.")

    args = parser.parse_args()
    if args.cmd == "collect":
        result = collect(args.run_dir, args.repeat, args.wall_s)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    results = [json.loads(path.read_text()) for path in args.results]
    if args.cmd == "baseline":
        if len(results) < 2:
            raise SystemExit(f"a baseline needs 2+ results to measure the spread, got {len(results)}")
        baseline = make_baseline(results)
        args.out.write_text(json.dumps(baseline, indent=2) + "\n")
        args.report.write_text(report(baseline))
        return

    baseline = json.loads(args.baseline.read_text()) if args.baseline and args.baseline.exists() else None
    markdown, regressed = summarize(results, args.expected, baseline)
    sys.stdout.write(markdown)
    sys.exit(1 if regressed else 0)


if __name__ == "__main__":
    main()
