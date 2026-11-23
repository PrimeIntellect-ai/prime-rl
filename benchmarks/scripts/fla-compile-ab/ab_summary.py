"""Compare the pr1 and pr2 runs of each cell: step time over all steps and over the steps after the last Dynamo compile
on any rank, compiles, recompiles with the step and guard that triggered them, graph breaks, peak memory and numerics.
Usage: python ab_summary.py RUNS_DIR CELL... (CELL is a config name such as qwen3_5_cp2)"""

import bisect
import glob
import re
import statistics
import sys
from collections import Counter

from tables import rel_diffs, summarize

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"(\d\d:\d\d:\d\d)\s+\w+\s+Step (\d+) \|")
RECOMPILE = re.compile(r"(\d\d:\d\d:\d\d)\.\d+ .*\[(\d+/\d+)\] \[__recompiles\] Recompiling function")
GUARD = re.compile(r"\[__recompiles\]\s+- (\S+): (.*)")


def recompiles(run):
    """Each rank's recompiles as (compile id, step, newest cache entry's failed guard)."""
    steps = []
    for line in open(f"{run}/logs/trainer.log", errors="replace"):
        match = STEP.search(ANSI.sub("", line))
        if match:
            steps.append((match.group(1), int(match.group(2))))
    times = [t for t, _ in steps]
    per_rank = {}
    for path in sorted(glob.glob(f"{run}/torchrun/*/attempt_0/*/stderr.log")):
        lines = open(path, errors="replace").read().splitlines()
        found = []
        for i, line in enumerate(lines):
            match = RECOMPILE.search(line)
            if not match:
                continue
            guards = []
            for nxt in lines[i + 2 : i + 12]:
                guard_match = GUARD.search(nxt)
                if not guard_match:
                    break
                guards.append(guard_match)
            guard = re.sub(r", \d+\)", ")", re.sub(r"  # .*", "", guards[-1].group(2)))[:120] if guards else "?"
            index = bisect.bisect_right(times, match.group(1))
            found.append((match.group(2), steps[index][1] if index < len(steps) else None, guard))
        per_rank[int(path.split("/")[-2])] = found
    return per_rank


def fmt(values, digits=3):
    return f"{values:.{digits}f}" if values == values else "n/a"


runs_dir = sys.argv[1]
for cell in sys.argv[2:]:
    print(f"### {cell}")
    summaries = {}
    for arm in ("pr1", "pr2"):
        run = f"{runs_dir}/{cell}-{arm}"
        summary = summarize(run, verbose=False)
        summaries[arm] = summary
        per_rank = recompiles(run)
        last = max([1] + [step or 10**6 for found in per_rank.values() for _, step, _ in found])
        times = summary["times"]
        all_steps = [times[s] for s in sorted(times)]
        post = [times[s] for s in sorted(times) if s > last]
        nan = float("nan")
        print(
            f"{arm} {summary['sha']}: steps 1-{len(all_steps)} total {sum(all_steps):.1f} s, steps 2-{len(all_steps)} total "
            f"{sum(all_steps[1:]):.1f} s, step 1 {times.get(1):.1f} s; last compile at step {last}; "
            f"{len(post)} post-compile steps median {fmt(statistics.median(post) if post else nan)} s "
            f"mean {fmt(statistics.mean(post) if post else nan)} s; steps 3-{len(all_steps)} median "
            f"{fmt(statistics.median(all_steps[2:]))} s; peak alloc/reserved "
            f"{summary['peak_alloc']:.2f}/{summary['peak_res']:.2f} GiB"
        )
        print(
            f"    layer-frame compiles/rank {summary['layer_compiles']}; recompiles/rank {summary['recompiles']}; "
            f"graph breaks/rank {summary['graph_breaks']}; rank 0 ops {summary['ops']}"
        )
        guards = Counter((step, guard) for found in per_rank.values() for _, step, guard in found)
        for (step, guard), count in sorted(guards.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1])):
            print(f"    recompile on {count} rank(s) at step {step}: {guard}")
        print("    time/step " + " ".join(f"{s}:{times[s]:.2f}" for s in sorted(times)))
    loss, grad_norm, n = rel_diffs(summaries["pr1"], summaries["pr2"])
    print(f"pr2 vs pr1: loss max rel {loss:.2e}, grad norm max rel {grad_norm:.2e} over {n} steps")
