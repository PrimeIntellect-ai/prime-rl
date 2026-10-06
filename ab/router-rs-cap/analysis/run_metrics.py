# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""A/B figures from a per-run summary CSV and per-step metrics: grouped bars with a noise band, and step lines.

Bars: `--csv` has one row per run with columns `group`, `variant`, and one column per metric. Each `--bar
COLUMN:TITLE` draws one panel with a bar per variant, panels side by side per group. `--noise V1,V2,...` shades the
min-to-max band of those variants (same-code repeats) in every panel, so a bar outside the band is a real effect.
Missing values are skipped.

Lines: each `--lines METRIC` draws, per group, that metric over steps for every run named in the CSV's `run_dir`
column (`<run_dir>/monitors/file/metrics.jsonl`), with `--steady LO HI` shaded.

Usage:
  uv run --script run_metrics.py --csv summary.csv --out bars.png --noise fp32-cap2,fp32-cap2-r2,fp32-main \\
    --bar "tok_s_gpu:tokens/s/GPU (higher is better)" --bar "peak_gib:peak memory GiB (lower is better)"
  uv run --script run_metrics.py --csv summary.csv --out steps.png --lines time/step --steady 5 20
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00", "#f0e442", "#999999", "#000000"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bar", action="append", default=[], help="COLUMN:TITLE")
    parser.add_argument("--noise", default=None, help="comma-separated same-code variants for the noise band")
    parser.add_argument("--lines", action="append", default=[], help="per-step metric name in metrics.jsonl")
    parser.add_argument("--steady", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--title", default="")
    return parser.parse_args()


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def number(value):
    return float(value) if value not in ("", None) else None


def draw_bars(rows, args):
    groups = list(dict.fromkeys(row["group"] for row in rows))
    noise = set(args.noise.split(",")) if args.noise else set()
    fig, axes = plt.subplots(
        len(args.bar), len(groups), figsize=(6.5 * len(groups), 3.2 * len(args.bar)), squeeze=False
    )
    for row_index, spec in enumerate(args.bar):
        column, title = spec.split(":", 1)
        for col_index, group in enumerate(groups):
            ax = axes[row_index][col_index]
            items = [(r["variant"], number(r.get(column))) for r in rows if r["group"] == group]
            items = [(v, x) for v, x in items if x is not None]
            if not items:
                ax.set_visible(False)
                continue
            labels = [v for v, _ in items]
            values = [x for _, x in items]
            colors = ["#999999" if v in noise else PALETTE[i % len(PALETTE)] for i, v in enumerate(labels)]
            ax.barh(labels, values, color=colors)
            band = [x for v, x in items if v in noise]
            if len(band) > 1:
                ax.axvspan(min(band), max(band), color="#cc0000", alpha=0.12, linewidth=0)
            low, high = min(values), max(values)
            if low > 0 and (high - low) < 0.2 * low:
                spread = high - low + 1e-9
                ax.set_xlim(low - 0.6 * spread - 0.01 * low, high + 0.35 * spread + 0.01 * low)
                ax.text(0.99, 0.02, "axis does not start at 0", transform=ax.transAxes, ha="right", fontsize=7,
                        color="#555555")
            else:
                ax.set_xlim(min(low, 0), high * 1.15 if high > 0 else 1)
            x_lo, x_hi = ax.get_xlim()
            for y, x in enumerate(values):
                ax.text(x + 0.01 * (x_hi - x_lo), y, f"{x:,.3g}" if abs(x) < 100 else f"{x:,.0f}", va="center",
                        fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{group}: {title}", loc="left", fontsize=10)
            ax.grid(axis="x", alpha=0.3)
    if noise:
        fig.text(0.5, 0.005, f"Gray bars: same-code repeats ({', '.join(sorted(noise))}); red band: their min-to-max.",
                 ha="center", fontsize=9)
    return fig


def load_steps(run_dir, metric):
    by_step = defaultdict(dict)
    for line in (Path(run_dir) / "monitors/file/metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        by_step[record["step"]].update(record)
    return sorted((step, values[metric]) for step, values in by_step.items() if metric in values)


def draw_lines(rows, args):
    groups = list(dict.fromkeys(row["group"] for row in rows))
    fig, axes = plt.subplots(
        len(args.lines), len(groups), figsize=(6.5 * len(groups), 3.4 * len(args.lines)), squeeze=False
    )
    for row_index, metric in enumerate(args.lines):
        for col_index, group in enumerate(groups):
            ax = axes[row_index][col_index]
            for i, r in enumerate(r for r in rows if r["group"] == group and r.get("run_dir")):
                points = load_steps(r["run_dir"], metric)
                ax.plot([s for s, _ in points], [v for _, v in points], label=r["variant"], color=PALETTE[i % len(PALETTE)],
                        linewidth=1.2, marker=".", markersize=3)
            if args.steady:
                ax.axvspan(args.steady[0], args.steady[1], color="#999999", alpha=0.1, linewidth=0)
            ax.set_title(f"{group}: {metric} per step", loc="left", fontsize=10)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            values = [v for r in rows if r["group"] == group and r.get("run_dir")
                      for s, v in load_steps(r["run_dir"], metric) if not args.steady or s >= args.steady[0]]
            if values:
                pad = 0.1 * (max(values) - min(values) + 1e-9)
                ax.set_ylim(min(values) - pad, max(values) + pad)
            ax.legend(fontsize=7, ncol=2)
    return fig


def main():
    args = parse_args()
    rows = read_rows(args.csv)
    fig = draw_lines(rows, args) if args.lines else draw_bars(rows, args)
    if args.title:
        fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
