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
column, with `--steady LO HI` shaded. A run dir is read as prime-rl's `<run_dir>/monitors/file/metrics.jsonl`; a
path to a `.jsonl` file is read directly, so any repo can log one JSON object per line with a `step` key and
metric keys (records of the same step are merged, and the last record carrying a metric wins). A path to a `.json`
file is read as one record at step 0 (its own `step` key if it has one), with nested keys joined by `/`
(`{"forward_ms": {"median": 2.9}}` gives `forward_ms/median`), for benchmarks that write one result object per run. `--prefer KEY=VALUE`
makes a record with that key value win instead, for logs where two processes write the same metric at a step.

Summary: `--summarize COLUMN=METRIC` (repeatable) writes the per-run summary CSV that the bar mode reads: one row per
`--run GROUP:VARIANT=RUN_DIR` with the median (`--stat mean` for the mean) of METRIC over the `--steady` steps (all
steps without it) in COLUMN, plus COLUMN_min and COLUMN_max. The mean suits all-step views, where one-time stalls
count. It also prints the rows as a markdown table. Non-finite values (NaN, inf) are
dropped, and a metric with no values in a run is reported on stderr. `--run` can replace `--csv` in lines mode
too; bars need the summary CSV. METRIC may also be a per-step ratio 'NUM / DEN' or 'NUM / DEN * SCALE' (spaces
around `/` and `*` required, since metric names contain slashes), e.g.
"s_per_mtok=time/forward_backward / progress/num_tokens * 1e6" to normalize a time by a varying per-step workload.
Each term of METRIC may be `delta(NAME)`, the per-step increase of a cumulative counter, and may end in `@RUN_DIR` to
read it from another run. E.g. when arm B pads rows that arm A does not, on the same samples, B's non-padding
tokens/s/GPU on 64 GPUs is "tok_s_gpu=delta(progress/num_tokens)@runs/a / time/step * 0.015625".

Step table: `--step-table METRIC` (repeatable) writes one CSV row per step to `--out`, with a column per metric and
run, and prints each metric as a markdown table, runs side by side, ending in each run's max |diff| from the first
`--run`. Use it for parity checks (do two arms track step by step?) and to read off per-step stalls; `--steady`
limits the steps and `--digits` sets the printed precision. METRIC takes the same forms as in `--summarize`.
`--stall-above X` adds rows with each run's 20th percentile over the table's steps (its unstalled step time while
fewer than 80% of steps stall) and the steps more than X above it (count, share, range of the excess), for counting
stalls such as a node's power-brake episodes.

Usage:
  uv run --script run_metrics.py --out parity.csv --step-table optim/grad_norm --step-table time/step \\
    --run "dsv4:main=runs/main" --run "dsv4:fix=runs/fix"
  uv run --script run_metrics.py --csv summary.csv --out bars.png --noise fp32-cap2,fp32-cap2-r2,fp32-main \\
    --bar "tok_s_gpu:tokens/s/GPU (higher is better)" --bar "peak_gib:peak memory GiB (lower is better)"
  uv run --script run_metrics.py --csv summary.csv --out steps.png --lines time/step --steady 5 20
  uv run --script run_metrics.py --out summary.csv --steady 5 20 --summarize "step_s=time/step" \\
    --summarize "peak_gib=perf/peak_memory" --run "qwen:A0=runs/a0/bench" --run "qwen:A1=runs/a1/bench"
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00", "#f0e442", "#999999", "#000000"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--run", action="append", default=[], help="GROUP:VARIANT=RUN_DIR, instead of --csv")
    parser.add_argument("--summarize", action="append", default=[], help="COLUMN=METRIC, write a summary CSV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bar", action="append", default=[], help="COLUMN:TITLE")
    parser.add_argument("--noise", default=None, help="comma-separated same-code variants for the noise band")
    parser.add_argument("--lines", action="append", default=[], help="per-step metric name in metrics.jsonl")
    parser.add_argument("--steady", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--stat", choices=("median", "mean"), default="median", help="summary statistic per run")
    parser.add_argument("--title", default="")
    parser.add_argument(
        "--step-table", action="append", default=[], help="METRIC, write per-step values side by side per run"
    )
    parser.add_argument("--digits", type=int, default=6, help="significant digits in the --step-table markdown")
    parser.add_argument("--stall-above", type=float, default=None, metavar="X",
                        help="with --step-table, count steps more than X above the run's 20th percentile over the table's steps")
    parser.add_argument("--prefer", default=None, metavar="KEY=VALUE",
                        help="when several records of a step carry a metric, take the one whose KEY equals VALUE")
    args = parser.parse_args()
    args.prefer = tuple(args.prefer.split("=", 1)) if args.prefer else None
    return args


def read_rows(args):
    if args.csv:
        with open(args.csv) as f:
            return list(csv.DictReader(f))
    rows = []
    for spec in args.run:
        key, run_dir = spec.split("=", 1)
        group, variant = key.split(":", 1)
        rows.append({"group": group, "variant": variant, "run_dir": run_dir})
    return rows


def write_summary(rows, args):
    columns = [spec.split("=", 1) for spec in args.summarize]
    fields = ["group", "variant", "run_dir"]
    for column, _ in columns:
        fields += [column, f"{column}_min", f"{column}_max"]
    for row in rows:
        for column, metric in columns:
            values = [
                v
                for s, v in load_metric(row["run_dir"], metric, args.prefer)
                if (not args.steady or args.steady[0] <= s <= args.steady[1]) and math.isfinite(v)
            ]
            if not values:
                print(f"warning: no finite '{metric}' values in {row['run_dir']} for the steps asked", file=sys.stderr)
            row[column] = getattr(statistics, args.stat)(values) if values else ""
            row[f"{column}_min"] = min(values) if values else ""
            row[f"{column}_max"] = max(values) if values else ""
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("| group | variant | " + " | ".join(f"{c} ({args.stat}, min-max)" for c, _ in columns) + " |")
    print("|---|---|" + "---|" * len(columns))
    for row in rows:
        cells = [f"{row[c]:.4g} ({row[c + '_min']:.4g}-{row[c + '_max']:.4g})" if row[c] != "" else "" for c, _ in columns]
        print(f"| {row['group']} | {row['variant']} | " + " | ".join(cells) + " |")
    print(f"wrote {args.out}")


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
    if not any(ax.get_visible() for ax in axes.flat):
        sys.exit("no bar values: each --bar COLUMN must be a numeric column of --csv (written by --summarize)")
    if noise:
        fig.text(0.5, 0.005, f"Gray bars: same-code repeats ({', '.join(sorted(noise))}); red band: their min-to-max.",
                 ha="center", fontsize=9)
    return fig


def flatten(record, prefix=""):
    flat = {}
    for key, value in record.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}/"))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def read_records(path):
    if path.suffix == ".json":
        return [{"step": 0, **flatten(json.loads(path.read_text()))}]
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_steps(run_dir, metric, prefer=None):
    """Per-step values of METRIC. The last record of a step wins, unless a record matches PREFER (KEY, VALUE)."""
    values = {}
    path = Path(run_dir)
    if not path.is_file():
        path = path / "monitors/file/metrics.jsonl"
    for record in read_records(path):
        step = record.get("step")
        if step is None or metric not in record:
            continue
        preferred = prefer is not None and str(record.get(prefer[0])) == prefer[1]
        if preferred or step not in values or not values[step][0]:
            values[step] = (preferred, record[metric])
    return sorted((step, value) for step, (_, value) in values.items())


def load_term(run_dir, term, prefer=None):
    """Per-step values of one term: NAME or delta(NAME), optionally suffixed @RUN_DIR to read another run."""
    if "@" in term:
        term, run_dir = term.rsplit("@", 1)
    if term.startswith("delta(") and term.endswith(")"):
        previous = 0.0
        deltas = []
        for step, value in load_steps(run_dir, term[len("delta(") : -1], prefer):
            deltas.append((step, value - previous))
            previous = value
        return deltas
    return load_steps(run_dir, term, prefer)


def load_metric(run_dir, spec, prefer=None):
    """Per-step values of a term (see load_term), or of the per-step ratio for a spec 'NUM / DEN [* SCALE]'."""
    if " / " not in spec:
        return load_term(run_dir, spec, prefer)
    numerator, denominator = spec.split(" / ", 1)
    scale = 1.0
    if " * " in denominator:
        denominator, scale_text = denominator.split(" * ", 1)
        scale = float(scale_text)
    den_by_step = dict(load_term(run_dir, denominator.strip(), prefer))
    numerator_steps = load_term(run_dir, numerator.strip(), prefer)
    return [(s, v / den_by_step[s] * scale) for s, v in numerator_steps if den_by_step.get(s)]


def draw_lines(rows, args):
    groups = list(dict.fromkeys(row["group"] for row in rows))
    fig, axes = plt.subplots(
        len(args.lines), len(groups), figsize=(6.5 * len(groups), 3.4 * len(args.lines)), squeeze=False
    )
    for row_index, metric in enumerate(args.lines):
        for col_index, group in enumerate(groups):
            ax = axes[row_index][col_index]
            for i, r in enumerate(r for r in rows if r["group"] == group and r.get("run_dir")):
                points = load_steps(r["run_dir"], metric, args.prefer)
                ax.plot([s for s, _ in points], [v for _, v in points], label=r["variant"], color=PALETTE[i % len(PALETTE)],
                        linewidth=1.2, marker=".", markersize=3)
            if args.steady:
                ax.axvspan(args.steady[0], args.steady[1], color="#999999", alpha=0.1, linewidth=0)
            ax.set_title(f"{group}: {metric} per step", loc="left", fontsize=10)
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
            values = [v for r in rows if r["group"] == group and r.get("run_dir")
                      for s, v in load_steps(r["run_dir"], metric, args.prefer)
                      if (not args.steady or s >= args.steady[0]) and math.isfinite(v)]
            if not values:
                print(f"warning: no finite '{metric}' values for group {group}", file=sys.stderr)
            else:
                pad = 0.1 * (max(values) - min(values) + 1e-9)
                ax.set_ylim(min(values) - pad, max(values) + pad)
            ax.legend(fontsize=7, ncol=2)
    return fig


def write_step_table(rows, args):
    """One CSV row per step with a column per (metric, run); prints each metric as markdown with the max |diff| vs
    the first run, so a parity check reads off which runs track step by step."""
    labels = [f"{r['group']}:{r['variant']}" for r in rows]
    by_metric = {
        metric: [dict(load_metric(r["run_dir"], metric, args.prefer)) for r in rows] for metric in args.step_table
    }
    steps = sorted({s for per_run in by_metric.values() for values in per_run for s in values})
    if args.steady:
        steps = [s for s in steps if args.steady[0] <= s <= args.steady[1]]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step"] + [f"{metric}@{label}" for metric in args.step_table for label in labels])
        for s in steps:
            writer.writerow([s] + [per_run[i].get(s, "") for per_run in by_metric.values() for i in range(len(rows))])
    for metric, per_run in by_metric.items():
        print(f"\n{metric}\n")
        print("| step | " + " | ".join(labels) + " |")
        print("|---:|" + "---:|" * len(labels))
        for s in steps:
            cells = [f"{values[s]:.{args.digits}g}" if s in values else "" for values in per_run]
            print(f"| {s} | " + " | ".join(cells) + " |")
        diffs = []
        for values in per_run:
            shared = [s for s in steps if s in values and s in per_run[0]]
            diffs.append(f"{max(abs(values[s] - per_run[0][s]) for s in shared):.3g}" if shared else "")
        print(f"| max abs diff vs {labels[0]} | " + " | ".join(diffs) + " |")
        if args.stall_above is not None:
            bases, stalls = [], []
            for values in per_run:
                present = sorted(values[s] for s in steps if s in values)
                base = present[int(0.2 * (len(present) - 1))] if present else math.nan
                excess = [values[s] - base for s in steps if s in values and values[s] - base > args.stall_above]
                bases.append(f"{base:.{args.digits}g}")
                span = f", +{min(excess):.3g} to +{max(excess):.3g}" if excess else ""
                stalls.append(f"{len(excess)}/{len(present)} ({100 * len(excess) / max(len(present), 1):.0f}%{span})")
            print("| p20 | " + " | ".join(bases) + " |")
            print(f"| steps over p20 + {args.stall_above:g} | " + " | ".join(stalls) + " |")
    print(f"\nwrote {args.out}")


def main():
    args = parse_args()
    rows = read_rows(args)
    if args.summarize:
        write_summary(rows, args)
        return
    if args.step_table:
        write_step_table(rows, args)
        return
    fig = draw_lines(rows, args) if args.lines else draw_bars(rows, args)
    if args.title:
        fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
