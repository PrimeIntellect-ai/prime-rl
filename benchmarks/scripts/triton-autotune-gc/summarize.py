"""Print a markdown table of peak memory, step time, and loss and grad norm drift relative to the first run dir."""

import argparse
import json
import statistics
from pathlib import Path

GiB = 2**30


def peaks(run):
    by_step = {}
    for path in sorted((run / "peaks").glob("rank*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not any(row[key] for key in ("max_allocated", "max_reserved", "allocated", "reserved")):
                continue
            allocated, reserved = by_step.get(row["call"], (0, 0))
            by_step[row["call"]] = (max(allocated, row["max_allocated"]), max(reserved, row["max_reserved"]))
    return by_step


def metrics(run):
    rows = {}
    for path in (run / "out").rglob("metrics.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "step" in row:
                rows.setdefault(int(row["step"]), {}).update(row)
    return rows


def summary(run):
    by_step = peaks(run)
    rows = metrics(run)
    last = max(rows)
    later = range(2, last + 1)
    return {
        "alloc1": by_step[1][0] / GiB,
        "res1": by_step[1][1] / GiB,
        "alloc_later": max(by_step[step][0] for step in later) / GiB,
        "res_later": max(by_step[step][1] for step in later) / GiB,
        "steady": statistics.median(rows[step]["time/step"] for step in range(4, last + 1)),
        "loss": {step: rows[step]["loss/mean"] for step in rows},
        "grad_norm": {step: rows[step]["optim/grad_norm"] for step in rows},
        "rc": (run / "rc.txt").read_text().strip(),
    }


def max_rel_diff(values, reference):
    return max(abs(values[step] - reference[step]) / abs(reference[step]) for step in reference)


parser = argparse.ArgumentParser()
parser.add_argument("run_dirs", nargs="+", type=Path)
args = parser.parse_args()

reference = summary(args.run_dirs[0])
print(
    "| run | rc | alloc step 1 (GiB) | reserved step 1 (GiB) | alloc steps 2+ (GiB) | reserved steps 2+ (GiB) "
    "| median step 4+ (s) | loss rel diff | grad norm rel diff |"
)
print("|---|---|---|---|---|---|---|---|---|")
for run in args.run_dirs:
    s = summary(run)
    print(
        f"| {run.name} | {s['rc']} | {s['alloc1']:.2f} | {s['res1']:.2f} | {s['alloc_later']:.2f} | {s['res_later']:.2f} "
        f"| {s['steady']:.3f} | {max_rel_diff(s['loss'], reference['loss']):.1e} "
        f"| {max_rel_diff(s['grad_norm'], reference['grad_norm']):.1e} |"
    )
