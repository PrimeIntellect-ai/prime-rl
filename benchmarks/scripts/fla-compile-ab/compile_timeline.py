"""Per step, the seconds each rank spent in Dynamo compiles. Usage: python compile_timeline.py RUN_DIR [details]."""

import glob
import os
import sys
from collections import defaultdict

import analyze as A

run = os.path.abspath(sys.argv[1])
start = float(open(os.path.join(run, "start_epoch.txt")).read())
ends = A.step_end_times(A.find_trainer_log(run), start)
steps = sorted(ends)


def step_of(t):
    for s in steps:
        if t <= ends[s] + 1:
            return s
    return steps[-1] + 1


per_step = defaultdict(lambda: defaultdict(float))
details = defaultdict(list)
for path in sorted(glob.glob(os.path.join(run, "torch_trace", "*.log"))):
    rank = int(A.RANK_RE.search(os.path.basename(path)).group(1))
    for r in A.read_compile_records(path):
        t0 = r.get("start_time")
        if t0 is None:
            continue
        s = step_of(t0)
        dur = A.us(r.get("duration_us"))
        per_step[s][rank] += dur
        details[(s, rank)].append(
            (
                r.get("compile_id"),
                round(dur, 1),
                r.get("is_forward"),
                (A.short_reason(r.get("recompile_reason")) or "")[:160],
            )
        )
prev = start
for s in steps:
    print(
        f"step {s}: wall {ends[s] - prev:.0f} s, compile s per rank {dict(sorted((k, round(v)) for k, v in per_step[s].items()))}"
    )
    prev = ends[s]
if len(sys.argv) > 2:
    for (s, rank), items in sorted(details.items()):
        if s >= 2:
            for item in items:
                print(f"  step {s} rank {rank}: {item}")
