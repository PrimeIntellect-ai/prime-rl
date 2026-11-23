"""Per step, the Triton kernel binaries a run wrote to its cache, by kernel name. Usage: python triton_timeline.py RUN_DIR TRITON_CACHE_DIR."""

import glob
import json
import os
import sys
from collections import Counter, defaultdict

import analyze as A

run, cache = os.path.abspath(sys.argv[1]), sys.argv[2]
start = float(open(os.path.join(run, "start_epoch.txt")).read())
ends = A.step_end_times(A.find_trainer_log(run), start)
steps = sorted(ends)


def step_of(t):
    for s in steps:
        if t <= ends[s]:
            return s
    return steps[-1] + 1


by_step = defaultdict(Counter)
for meta in glob.glob(os.path.join(cache, "**", "*.json"), recursive=True):
    if os.path.basename(meta).startswith("__grp__"):
        continue
    try:
        name = json.load(open(meta)).get("name")
    except Exception:
        continue
    if not name:
        continue
    by_step[step_of(os.path.getmtime(meta))][name] += 1
for s in sorted(by_step):
    total = sum(by_step[s].values())
    print(f"step {s}: {total} kernel binaries written")
    if s >= 2 or len(sys.argv) > 3:
        for name, n in by_step[s].most_common(40):
            print(f"   {n:3d} {name}")
