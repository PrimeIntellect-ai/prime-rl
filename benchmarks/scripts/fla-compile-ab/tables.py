"""Summarize runs against a base run: per-rank compiles and recompiles with reasons, graph breaks, ops captured, step times, peaks and numerics. Usage: python tables.py BASE_DIR RUN_DIR..."""

import glob
import json
import os
import re
import statistics
import sys

from analyze import analyze, is_layer_frame

GIB = 2**30


def peaks(run):
    alloc = res = 0
    for path in glob.glob(f"{run}/peaks/rank*.jsonl"):
        for line in open(path):
            r = json.loads(line)
            if r["call"] >= 2:
                alloc, res = max(alloc, r["max_allocated"]), max(res, r["max_reserved"])
    return alloc / GIB, res / GIB


def custom_ops(run):
    ops = {}
    for path in glob.glob(f"{run}/torch_trace/*rank_0_*.log"):
        for match in re.finditer(r"torch\.ops\.(prime_rl|prime_kernels)\.(\w+)", open(path, errors="replace").read()):
            ops[f"{match.group(1)}::{match.group(2)}"] = ops.get(f"{match.group(1)}::{match.group(2)}", 0) + 1
    return ops


def summarize(run, verbose=True):
    r = analyze(run)
    times = {s: v["time_step"] for s, v in r["steps"].items() if v["time_step"] is not None}
    steady = [times[s] for s in range(5, 21) if s in times]
    pa, pr = peaks(run)
    comp = r["compile"]
    recompiles = {k: c["recompiles"] for k, c in sorted(comp.items())}
    layer_compiles = {
        k: sum(f["compiles"] for f in c["frames"].values() if is_layer_frame(f["label"]))
        for k, c in sorted(comp.items())
    }
    gbs = {k: sum(c["graph_breaks"].values()) for k, c in sorted(comp.items())}
    ops0 = []
    if 0 in comp:
        for f in comp[0]["frames"].values():
            if is_layer_frame(f["label"]):
                ops0.append(f"{f['label'].split(' @ ')[0]}:{f['graph_ops']}")
    summary = {
        "run": run,
        "rc": r["rc"],
        "sha": (r.get("sha") or "")[:9],
        "step1": times.get(1),
        "median": statistics.median(steady) if steady else None,
        "max": max(steady) if steady else None,
        "times": times,
        "peak_alloc": pa,
        "peak_res": pr,
        "layer_compiles": layer_compiles,
        "recompiles": recompiles,
        "graph_breaks": gbs,
        "ops": ops0,
        "steps": r["steps"],
    }
    if verbose:
        print(f"## {os.path.basename(run)} rc={r['rc']} sha={summary['sha']} wall={r['wall_s']:.0f}s")
        print(
            f"layer-frame compiles per rank {layer_compiles}; recompiles per rank {recompiles}; graph breaks per rank {gbs}"
        )
        reasons = {}
        for c in comp.values():
            for k, v in c["recompile_reasons"].items():
                k = re.sub(r"\s+#.*", "", k)[:260]
                reasons[k] = reasons.get(k, 0) + v
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  recompile {v}x {k}")
        breaks = {}
        for c in comp.values():
            for k, v in c["graph_breaks"].items():
                breaks[k[:200]] = breaks.get(k[:200], 0) + v
        for k, v in sorted(breaks.items(), key=lambda kv: -kv[1]):
            print(f"  graph break {v}x {k}")
        print(f"  rank0 layer graph ops: {ops0}; custom ops in rank0 trace: {custom_ops(run)}")
        print("  time/step: " + " ".join(f"{s}:{t:.2f}" for s, t in sorted(times.items())))
        if steady:
            print(
                f"  step1={times.get(1):.1f} s, median5-20={summary['median']:.3f} s, max5-20={summary['max']:.3f} s, peak alloc/res {pa:.2f}/{pr:.2f} GiB"
            )
    return summary


def rel_diffs(base, other):
    common = sorted(set(base["steps"]) & set(other["steps"]))

    def rel(a, b):
        return abs(a - b) / max(abs(a), 1e-30)

    loss = max(rel(base["steps"][s]["loss"], other["steps"][s]["loss"]) for s in common)
    gn = max(rel(base["steps"][s]["grad_norm"], other["steps"][s]["grad_norm"]) for s in common)
    return loss, gn, len(common)


if __name__ == "__main__":
    base_run, *others = sys.argv[1:]
    base = summarize(base_run)
    for run in others:
        s = summarize(run)
        loss, gn, n = rel_diffs(base, s)
        print(f"  vs {os.path.basename(base_run)}: loss max rel {loss:.2e}, grad norm max rel {gn:.2e} over {n} steps")
