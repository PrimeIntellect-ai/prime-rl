"""Compare a base run and a PR run: compiles and recompile reasons over all ranks, graph breaks, step time over all
steps and over the steps after the last compile on any rank, per-step loss and grad norm differences, and peak memory.
Usage: python ab_report.py BASE_RUN PR_RUN"""

import glob
import os
import re
import statistics
import sys

from analyze import analyze, is_layer_frame
from tables import peaks


def last_compile_end_s(run):
    start = float(open(os.path.join(run, "start_epoch.txt")).read())
    ends = []
    for path in os.listdir(os.path.join(run, "torch_trace")):
        for line in open(os.path.join(run, "torch_trace", path), errors="replace"):
            match = re.search(r'"end_time_us": (\d+)', line)
            if match and "compilation_metrics" in line:
                ends.append(int(match.group(1)) / 1e6 - start)
    return max(ends) if ends else None


def custom_ops(run):
    ops = {}
    for path in glob.glob(f"{run}/torch_trace/*rank_0_*.log"):
        for match in re.finditer(
            r"torch\.ops\.(prime_rl\w*|prime_kernels)\.(\w+)", open(path, errors="replace").read()
        ):
            name = f"{match.group(1)}::{match.group(2)}"
            ops[name] = ops.get(name, 0) + 1
    return ops


def summarize(run):
    result = analyze(run)
    steps = result["steps"]
    times = {s: row["time_step"] for s, row in steps.items() if row["time_step"] is not None}
    step_end = {s: row["end_s_from_launch"] for s, row in steps.items()}
    last_compile = last_compile_end_s(run)
    post_steps = [s for s in sorted(times) if s - 1 in step_end and step_end[s - 1] >= last_compile]
    comp = result["compile"]
    layer_compiles = {
        rank: sum(f["compiles"] for f in c["frames"].values() if is_layer_frame(f["label"])) for rank, c in comp.items()
    }
    reasons = {}
    for c in comp.values():
        for frame in c["frames"].values():
            for entry in frame["entries"]:
                if entry["reason"]:
                    reason = re.sub(r"\s+#.*", "", entry["reason"])[:200]
                    reasons[reason] = reasons.get(reason, 0) + 1
    ops0 = [f["graph_ops"] for f in comp.get(0, {}).get("frames", {}).values() if is_layer_frame(f["label"])]
    alloc, reserved = peaks(run)
    return {
        "name": os.path.basename(run),
        "sha": (result.get("sha") or "")[:9],
        "rc": result["rc"],
        "steps": steps,
        "times": times,
        "last_compile_s": last_compile,
        "post_steps": post_steps,
        "layer_compiles": layer_compiles,
        "graph_breaks": {rank: sum(c["graph_breaks"].values()) for rank, c in comp.items()},
        "reasons": reasons,
        "ops0": ops0,
        "custom_ops": custom_ops(run),
        "peak_alloc": alloc,
        "peak_reserved": reserved,
    }


def fmt(values):
    if not values:
        return "n/a"
    return (
        f"n={len(values)} mean {statistics.mean(values):.3f} s, median {statistics.median(values):.3f} s, "
        f"min {min(values):.3f} s, max {max(values):.3f} s"
    )


def report(arm):
    print(f"## {arm['name']} rc={arm['rc']} sha={arm['sha']}")
    print(f"layer-frame compiles per rank: {arm['layer_compiles']}")
    print(f"rank 0 layer-frame ops per compile: {arm['ops0']}")
    print(f"prime_rl custom ops in rank 0's compiled code: {arm['custom_ops']}")
    print(f"graph breaks per rank: {arm['graph_breaks']}")
    for reason, count in sorted(arm["reasons"].items(), key=lambda kv: -kv[1]):
        print(f"  recompile {count}x (all ranks): {reason}")
    print("time/step: " + " ".join(f"{s}:{t:.2f}" for s, t in sorted(arm["times"].items())))
    print(f"all steps: {fmt(list(arm['times'].values()))}")
    post = [arm["times"][s] for s in arm["post_steps"]]
    print(f"last compile ends {arm['last_compile_s']:.0f} s after launch; steps after it: {fmt(post)}")
    print(f"peak memory over ranks: allocated {arm['peak_alloc']:.2f} GiB, reserved {arm['peak_reserved']:.2f} GiB")


def numerics(base, pr):
    common = sorted(set(base["steps"]) & set(pr["steps"]))
    for metric in ("loss", "grad_norm"):
        abs_diff = [abs(base["steps"][s][metric] - pr["steps"][s][metric]) for s in common]
        rel_diff = [d / max(abs(base["steps"][s][metric]), 1e-30) for s, d in zip(common, abs_diff)]
        worst = common[rel_diff.index(max(rel_diff))]
        print(
            f"{metric}: max abs diff {max(abs_diff):.3e}, max rel diff {max(rel_diff):.2e} (step {worst}) over "
            f"{len(common)} steps"
        )


if __name__ == "__main__":
    base, pr = summarize(sys.argv[1]), summarize(sys.argv[2])
    report(base)
    report(pr)
    numerics(base, pr)
    common = sorted(set(base["post_steps"]) & set(pr["post_steps"]))
    print(f"steps after the last compile in both arms: {common}")
    for arm in (base, pr):
        print(f"  {arm['name']}: {fmt([arm['times'][s] for s in common])}")
