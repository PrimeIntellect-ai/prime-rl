"""List every host-blocking CUDA call in the last full step(s) of a torch.profiler chrome trace, with cost and call site.

A step window runs from one CPU-side `user_annotation` named ANCHOR to the next, so it covers forward, backward,
optimizer, and logging. By default it is the last complete step (between the last two anchors).

Blocking calls: cudaStreamSynchronize, cudaDeviceSynchronize, cudaEventSynchronize, synchronous cudaMemcpy*, their
driver-API equivalents, and any cudaMemcpyAsync whose host call returned only after its GPU copy finished (D2H into
pageable memory, as in `.item()`, `.tolist()`, `.cpu()`). Calls inside the same innermost CPU op instance form one
sync point.

Per sync point:
- host: blocked time in the calls.
- idle@ret: the gap in the union of kernel, memcpy, and memset intervals (all streams) that contains the call's
  return, 0 if the GPU was busy then; each gap is charged once, to the first sync point returning inside it.
- exposed: all GPU idle from when the GPU drained before the return (or the return, if it was busy) until the host
  is ahead again (the first kernel launched after the return that starts more than 50 us after its launch call
  ends), capped at the next sync point and never counted twice. After a sync the queue is empty, so every
  host-bound gap until then is the sync's cost.
- the enclosing CPU op stack with the nearest user annotation (FSDP annotations name the module) and the innermost
  op's Input Dims.
Sites are deduplicated and sorted by exposed idle.

`--stacks syncs_rank<R>.jsonl` (from sync_stacks/sitecustomize.py, same run and rank as the trace) matches each
sync point to a Python stack recorded on the same thread, in order: each record goes to the earliest unmatched point
that started at most 1 ms after it and returned at most 20 ms before it (callbacks record just before a sync,
warnings just after the Python call that did one or more syncs), and groups by Python site.
`--dump out.json` writes the per-point rows for join_sync_dumps.py. `--list` prints every sync point in time order.

Usage: python3 find_syncs.py <trace.json.gz> [--steps N] [--anchor forward] [--stacks syncs.jsonl] [--list] [--dump f]
"""

import argparse
import bisect
import gzip
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sync_stacks_report import site_key, site_label

BLOCKING = {
    "cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaThreadSynchronize",
    "cudaMemcpy", "cudaMemcpy2D", "cudaMemcpy3D", "cudaMemcpyFromSymbol", "cudaMemcpyToSymbol", "cudaMemset",
    "cuStreamSynchronize", "cuCtxSynchronize", "cuEventSynchronize", "cuMemcpy", "cuMemcpyDtoH_v2", "cuMemcpyHtoD_v2",
    "cuMemcpyDtoD_v2", "cuMemcpyAtoH_v2",
}
ASYNC_COPIES = {"cudaMemcpyAsync", "cudaMemcpy2DAsync", "cuMemcpyAsync", "cuMemcpyDtoHAsync_v2"}
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
MARGIN = 5e4
LEAD = 50
SKIP_FOR_DIMS = ("aten::copy_", "aten::_to_copy", "aten::to", "aten::_local")

p = argparse.ArgumentParser()
p.add_argument("trace")
p.add_argument("--steps", type=int, default=1)
p.add_argument("--anchor", default="forward")
p.add_argument("--stacks")
p.add_argument("--list", action="store_true")
p.add_argument("--dump")
p.add_argument("--depth", type=int, default=4)
args = p.parse_args()

trace = json.load(gzip.open(args.trace))
xs = [e for e in trace["traceEvents"] if e.get("ph") == "X"]
anchors = sorted(e["ts"] for e in xs if e.get("cat") == "user_annotation" and e["name"] == args.anchor)
if len(anchors) < args.steps + 1:
    raise SystemExit(f"need {args.steps + 1} '{args.anchor}' annotations, found {len(anchors)}")
start, end = anchors[-1 - args.steps], anchors[-1]
n = args.steps

gpu = sorted((e["ts"], e["ts"] + e["dur"]) for e in xs if e.get("cat") in GPU_CATS and e["ts"] + e["dur"] >= start
             and e["ts"] <= end + 1e6)
busy = []
for a, b in gpu:
    if busy and a <= busy[-1][1]:
        busy[-1][1] = max(busy[-1][1], b)
    else:
        busy.append([a, b])
busy_starts = [iv[0] for iv in busy]
busy_in_window = sum(max(0.0, min(b, end) - max(a, start)) for a, b in busy)
claimed_gaps = set()


def idle_at(t):
    i = bisect.bisect_right(busy_starts, t) - 1
    if (i >= 0 and busy[i][1] >= t) or i in claimed_gaps:
        return 0.0
    claimed_gaps.add(i)
    prev_end = busy[i][1] if i >= 0 else t
    next_start = busy[i + 1][0] if i + 1 < len(busy) else t
    return max(0.0, next_start - prev_end)


def idle_between(a, b):
    total, t = 0.0, a
    for s, e in busy[max(0, bisect.bisect_right(busy_starts, a) - 1):]:
        if t >= b:
            break
        if s > t:
            total += min(s, b) - t
        t = max(t, e)
    return total + max(0.0, b - t)


launch_by_corr = {e["args"]["correlation"]: e for e in xs if e.get("cat") in ("cuda_runtime", "cuda_driver")
                  and "correlation" in e.get("args", {}) and start - MARGIN <= e["ts"] <= end + MARGIN}
launches = sorted((launch_by_corr[k["args"]["correlation"]]["ts"] + launch_by_corr[k["args"]["correlation"]]["dur"],
                   k["ts"]) for k in xs if k.get("cat") == "kernel" and k.get("args", {}).get("correlation") in launch_by_corr)
launch_ends = [le for le, _ in launches]


def host_ahead_at(t):
    for launch_end, kernel_start in launches[bisect.bisect_left(launch_ends, t):]:
        if kernel_start - launch_end > LEAD:
            return launch_end
    return end


copy_by_corr = {e["args"]["correlation"]: e for e in xs if e.get("cat") == "gpu_memcpy" and "correlation" in e.get("args", {})}
ops_by_tid = defaultdict(list)
for e in xs:
    if e.get("cat") in ("cpu_op", "user_annotation") and e["ts"] + e["dur"] >= start - MARGIN and e["ts"] <= end + MARGIN:
        ops_by_tid[e["tid"]].append(e)
starts_by_tid, parent_by_tid = {}, {}
for tid, ops in ops_by_tid.items():
    ops.sort(key=lambda e: e["ts"])
    starts_by_tid[tid] = [e["ts"] for e in ops]
    parents, open_ = [], []
    for i, o in enumerate(ops):
        while open_ and ops[open_[-1]]["ts"] + ops[open_[-1]]["dur"] < o["ts"] + o["dur"]:
            open_.pop()
        parents.append(open_[-1] if open_ else -1)
        open_.append(i)
    parent_by_tid[tid] = parents
phases = [e for e in xs if e.get("cat") == "user_annotation" and e["name"] in ("forward", "backward")
          and e["ts"] + e["dur"] >= start and e["ts"] <= end]


def enclosing(call):
    ops, parents = ops_by_tid[call["tid"]], parent_by_tid.get(call["tid"], [])
    i = bisect.bisect_right(starts_by_tid.get(call["tid"], []), call["ts"]) - 1
    chain = []
    while i >= 0:
        if ops[i]["ts"] + ops[i]["dur"] >= call["ts"] + call["dur"]:
            chain.append(ops[i])
        i = parents[i]
    return chain[::-1]


def phase(t):
    return next((e["name"] for e in phases if e["ts"] <= t <= e["ts"] + e["dur"]), "other")


def blocking_kind(e):
    if e["name"] in BLOCKING:
        return e["name"]
    if e["name"] in ASYNC_COPIES:
        copy = copy_by_corr.get(e.get("args", {}).get("correlation"))
        if copy is not None and e["ts"] + e["dur"] >= copy["ts"] + copy["dur"]:
            return f"{e['name']} [{copy['name'].split('(')[-1].rstrip(')')}]"
    return None


calls = sorted((e for e in xs if e.get("cat") in ("cuda_runtime", "cuda_driver") and start - MARGIN <= e["ts"] < end + MARGIN
                and blocking_kind(e)), key=lambda e: e["ts"])
points = {}
for c in calls:
    stack = enclosing(c)
    cpu = [o for o in stack if o.get("cat") == "cpu_op"]
    key = (c["tid"], id(cpu[-1])) if cpu else (c["tid"], id(c))
    pt = points.setdefault(key, {"tid": c["tid"], "ts": c["ts"], "stack": stack, "cpu": cpu, "kinds": [], "host": 0.0,
                                 "ret": 0.0, "rec": None})
    pt["kinds"].append(blocking_kind(c))
    pt["host"] += c["dur"]
    pt["ret"] = max(pt["ret"], c["ts"] + c["dur"])
with_margin = sorted(points.values(), key=lambda pt: pt["ts"])
ordered = [pt for pt in with_margin if start <= pt["ts"] < end]

if args.stacks:
    base_us = trace["baseTimeNanoseconds"] / 1e3
    recs_by_tid = defaultdict(list)
    for line in open(args.stacks):
        r = json.loads(line)
        r["t"] = r["t_ns"] / 1e3 - base_us
        replayed = r["stack"] and r["stack"][-1][2] == "_engine_run_backward"
        if start - MARGIN <= r["t"] <= end + MARGIN and not replayed:
            recs_by_tid[r["tid"]].append(r)
    lags = []
    ordered_ids = {id(pt) for pt in ordered}
    for tid, recs in recs_by_tid.items():
        recs.sort(key=lambda r: r["t"])
        pts = [pt for pt in with_margin if pt["tid"] == tid]
        k = 0
        for r in recs:
            while k < len(pts) and pts[k]["ret"] + 2e4 < r["t"]:
                k += 1
            if k < len(pts) and pts[k]["ts"] <= r["t"] + 1e3:
                r["matched"] = True
                pts[k]["rec"] = r
                if id(pts[k]) in ordered_ids:
                    lags.append(r["t"] - pts[k]["ts"])
                k += 1
    unmatched = [r for recs in recs_by_tid.values() for r in recs if not r.get("matched") and start <= r["t"] < end]
    print(f"stacks: matched {len(lags)} of {len(ordered)} sync points (median record minus call start "
          f"{statistics.median(lags) if lags else float('nan'):.0f} us); {len(unmatched)} recorded syncs in the window "
          f"had no blocking call in the trace")
    for r in unmatched:
        print(f"  unmatched {r['source']:6} {(r['t'] - start) / 1e3:9.2f} ms  {site_label(r)}")

covered = start
for k, pt in enumerate(ordered):
    next_ts = ordered[k + 1]["ts"] if k + 1 < len(ordered) else end
    until = min(host_ahead_at(pt["ret"]), next_ts)
    i = bisect.bisect_right(busy_starts, pt["ret"]) - 1
    drained = busy[i][1] if i >= 0 and busy[i][1] < pt["ret"] else pt["ret"]
    a = max(drained, covered)
    pt["exposed"] = idle_between(a, until) if until > a else 0.0
    covered = max(covered, until)

sites = {}
for pt in ordered:
    ann = next((o["name"] for o in reversed(pt["stack"]) if o.get("cat") == "user_annotation"), "-")
    names = [o["name"] for o in pt["cpu"]][-args.depth:]
    pt["ops"] = f"[{ann}] " + " > ".join(names) + f"  ({'+'.join(sorted(set(pt['kinds'])))})"
    pt["idle"] = idle_at(pt["ret"])
    dims = next((o.get("args", {}).get("Input Dims") for o in reversed(pt["cpu"]) if not o["name"].startswith(SKIP_FOR_DIMS)), None)
    key = site_key(pt["rec"]) if pt["rec"] else pt["ops"]
    pt["label"] = site_label(pt["rec"]) if pt["rec"] else pt["ops"]
    s = sites.setdefault(key, {"n": 0, "host": 0.0, "idle": 0.0, "exposed": 0.0, "max_host": 0.0, "dims": dims,
                               "phases": set(), "pt": pt})
    s["n"] += 1
    s["host"] += pt["host"]
    s["idle"] += pt["idle"]
    s["exposed"] += pt["exposed"]
    s["max_host"] = max(s["max_host"], pt["host"])
    s["phases"].add(phase(pt["ts"]))

total = end - start
print(f"{args.trace}")
print(f"  window {total / n / 1e3:.1f} ms/step over {n} step(s); GPU busy {busy_in_window / n / 1e3:.1f} ms/step "
      f"({100 * busy_in_window / total:.1f}%)")
print(f"  sync points {len(ordered) / n:.1f}/step, host blocked {sum(pt['host'] for pt in ordered) / n / 1e3:.2f} ms/step, "
      f"GPU idle at their return {sum(pt['idle'] for pt in ordered) / n / 1e3:.2f} ms/step, until the host is ahead "
      f"{sum(pt['exposed'] for pt in ordered) / n / 1e3:.2f} ms/step, all GPU idle {(total - busy_in_window) / n / 1e3:.2f} ms/step")
print(f"  {'per step':>8} {'exposed':>8} {'idle@ret':>8} {'host ms':>8} {'max host':>9}  phase    site")
for s in sorted(sites.values(), key=lambda s: (-s["exposed"], -s["idle"], -s["host"])):
    print(f"  x{s['n'] / n:<7g} {s['exposed'] / n / 1e3:8.2f} {s['idle'] / n / 1e3:8.2f} {s['host'] / n / 1e3:8.2f} "
          f"{s['max_host'] / 1e3:9.2f}  {','.join(sorted(s['phases'])):8} {s['pt']['label']}")
    if s["pt"]["rec"]:
        print(f"  {'':55}ops: {s['pt']['ops']}")
    print(f"  {'':55}dims={str(s['dims'])[:150]}")
if args.list:
    print("  time order (ms from window start):")
    for pt in ordered:
        print(f"    {(pt['ts'] - start) / 1e3:9.2f}  host {pt['host'] / 1e3:8.2f}  idle@ret {pt['idle'] / 1e3:6.2f}  "
              f"exposed {pt['exposed'] / 1e3:6.2f}  "
              f"tid {pt['tid']}  {pt['label']}")
if args.dump:
    rows = [{"t_ms": (pt["ts"] - start) / 1e3, "main_thread": pt["tid"] == pt["stack"][0]["pid"] if pt["stack"] else None,
             "ops": pt["ops"], "label": pt["label"], "host_us": pt["host"], "idle_us": pt["idle"],
             "exposed_us": pt["exposed"]} for pt in ordered]
    json.dump(rows, open(args.dump, "w"), indent=1)
