"""Per-step metrics, means and recompile counts of one RL run: python analyze_rl_run.py <run dir>."""

import collections
import datetime
import glob
import json
import re
import statistics
import sys

run_dir = sys.argv[1]
rows = collections.defaultdict(dict)
for line in open(f"{run_dir}/monitors/file/metrics.jsonl"):
    d = json.loads(line)
    step = d.get("step")
    if step is None:
        continue
    r = rows[step]
    if "time/forward_backward" in d:
        r["trainer_step_s"] = d["time/step"]
        r["fwd_bwd_s"] = d["time/forward_backward"]
        r["end"] = d["time"]
    for key, name in [
        ("logprob_collapse_regime/all/mean", "collapse"),
        ("mismatch_kl/all/mean", "mismatch_kl"),
        ("entropy/all/mean", "entropy"),
        ("loss/mean", "loss"),
        ("train/agg/all/agent/reward/mean", "reward"),
    ]:
        if key in d:
            r[name] = d[key]
    for k, v in d.items():
        if isinstance(v, float) and v != v:
            print(f"NaN at step {step}: {k}")

recompile_re = re.compile(r"\[rank(\d+)\]:V(\d\d)(\d\d) (\d\d:\d\d:\d\d\.\d+) .*Recompiling function (\S+) in (\S+)")
per_rank = collections.Counter()
per_func = collections.Counter()
lines_per_rank = collections.Counter()
last_recompile = None
year = datetime.date.today().year
for path in glob.glob(f"{run_dir}/logs/latest/trainer/torchrun/*/*/*/stderr.log"):
    for line in open(path, errors="replace"):
        if "__recompiles" in line:
            m0 = re.match(r"\[rank(\d+)\]", line)
            lines_per_rank[m0.group(1) if m0 else "?"] += 1
        m = recompile_re.search(line)
        if not m:
            continue
        rank, mon, day, hms, func, loc = m.groups()
        per_rank[rank] += 1
        per_func[f"{func} @ {loc.split('/site-packages/')[-1].split('/src/')[-1]}"] += 1
        t = datetime.datetime.strptime(f"{year}-{mon}-{day} {hms}", "%Y-%m-%d %H:%M:%S.%f").timestamp()
        last_recompile = t if last_recompile is None else max(last_recompile, t)

cols = ["collapse", "mismatch_kl", "entropy", "loss", "reward", "trainer_step_s", "fwd_bwd_s"]
print("step | " + " | ".join(cols))
for s in sorted(rows):
    r = rows[s]
    print(f"{s} | " + " | ".join(f"{r[c]:.4g}" if c in r else "-" for c in cols))

def means(steps):
    return {c: statistics.mean(rows[s][c] for s in steps if c in rows[s]) for c in cols}

all_steps = sorted(rows)
print("mean all steps:", {k: round(v, 6) for k, v in means(all_steps).items()})
if last_recompile is not None:
    post = [s for s in all_steps if "end" in rows[s] and rows[s]["end"] - rows[s]["trainer_step_s"] > last_recompile]
    print(f"last recompile at {datetime.datetime.fromtimestamp(last_recompile)}; post-recompile steps: {post[0] if post else None}..{post[-1] if post else None} (n={len(post)})")
    if post:
        print("mean post-recompile:", {k: round(v, 6) for k, v in means(post).items()})
print("Recompiling-function events per rank:", dict(sorted(per_rank.items())))
print("__recompiles log lines per rank:", dict(sorted(lines_per_rank.items())))
for f, n in per_func.most_common():
    print(f"  {n:4d}  {f}")
