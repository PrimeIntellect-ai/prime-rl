import json, re, statistics, sys
from pathlib import Path

OUT = Path("/home/garrett/prl_output_dir/mock2")
FIRST_STEADY = 3

def load(run):
    rows = {}
    for line in open(run / "monitors/file/metrics.jsonl"):
        d = json.loads(line)
        rows.setdefault(d["step"], {}).update(d)
    return rows

def summary(rows, key):
    vals = [r[key] for s, r in sorted(rows.items()) if s >= FIRST_STEADY and key in r]
    return (statistics.median(vals), min(vals), max(vals), len(vals)) if vals else None

for px in (0,):
    runs = sorted(p for p in OUT.glob("mock2-*") if p.is_dir() and (p / "monitors/file/metrics.jsonl").exists())
    if not runs:
        continue
    print(f"\n1024 px: median over steps >= {FIRST_STEADY} (min-max), lower is better")
    print(f"{'run':28s} {'forward_backward s':>22s} {'step s':>22s} {'MFU %':>6s} {'peak GiB':>8s} {'n':>3s}")
    for run in runs:
        rows = load(run)
        fb, st, mfu = summary(rows, "time/forward_backward"), summary(rows, "time/step"), summary(rows, "perf/mfu")
        mem = max((r.get("perf/peak_memory", 0) for r in rows.values()), default=0)
        if fb is None:
            print(f"{run.name:28s} incomplete"); continue
        print(f"{run.name:28s} {fb[0]:7.2f} ({fb[1]:5.2f}-{fb[2]:5.2f}) {st[0]:7.2f} ({st[1]:5.2f}-{st[2]:5.2f}) {mfu[0]:6.1f} {mem:8.1f} {fb[3]:3d}")

print("\nParity vs base-a at the same image size: max over steps of |delta| (base-b is the same-code noise floor)")
for px in (0,):
    ref_dir = OUT / "mock2-base-a"
    if not (ref_dir / "monitors/file/metrics.jsonl").exists():
        continue
    ref = load(ref_dir)
    for run in sorted(p for p in OUT.glob("mock2-*") if p.is_dir() and p != ref_dir):
        try:
            rows = load(run)
        except FileNotFoundError:
            continue
        common = [s for s in ref if s in rows and "loss/mean" in ref[s] and "loss/mean" in rows[s]]
        if not common:
            continue
        dl = max(abs(rows[s]["loss/mean"] - ref[s]["loss/mean"]) for s in common)
        dg = max(abs(rows[s]["optim/grad_norm"] - ref[s]["optim/grad_norm"]) for s in common)
        print(f"  {run.name:28s} steps={len(common):2d} max|d loss/mean|={dl:.2e} max|d grad_norm|={dg:.2e}")
