# /// script
# requires-python = ">=3.10"
# ///
"""What is allocated at the peak of a recorded window, by allocation site, from a CUDA allocator history dump.

Input: the JSON `memory_history/sitecustomize.py` writes (`torch.cuda.memory._snapshot()` taken with
`_record_memory_history(stacks="python")`). Allocated bytes follow `torch.cuda.memory_allocated()`: an `alloc` event
adds its size, a `free_requested` event subtracts it (the allocator lowers its stat there, before any deferred free
for blocks used on other streams completes), so the peak here is the window's `max_memory_allocated()`. The
allocated bytes at the window's start come from the dump's final segments (active blocks) minus the trace's net
change, or, when the dump holds it, the `memory_allocated()` the hook read as recording started (needed under
`expandable_segments`, whose segments do not list blocks the same way); blocks freed in the window without an `alloc`
in it, and the rest of the start bytes, are reported as "allocated before the window".

Each block live at the peak is grouped by its site: the innermost Python frame whose file matches `--site` (default
`prime_rl|tools/`), as `file:line function`, with the next `--context` matching frames outward appended. Prints the
peak (GiB, event index, time), then the groups by bytes; `--diff B.json` prints both dumps' groups side by side with
B minus A, largest change first. `--blocks REGEX` lists every block live at the peak whose site matches, with its
size, stream and the event index of its alloc (to order them against the window's phases). When the dump holds the hook's `max_memory_allocated()` readings, they
are printed as a check on the replayed peak. `--peaks K` also lists the K highest local peaks (event index, GiB) to show whether
the peak repeats every step. `--device` picks the device trace (default: the one with most events).

Usage: uv run --script memory_peak.py A.json [--diff B.json] [--site REGEX] [--context 2] [--top 30] [--peaks 0]
         [--device N] [--blocks REGEX]
"""

import argparse
import json
import re
from collections import defaultdict

GIB = 1024**3


def site_key(frames, site: re.Pattern, context: int) -> str:
    matched = [f for f in frames if site.search(f.get("filename", ""))]
    if not matched:
        inner = frames[0] if frames else {}
        return f"(no matching frame) {inner.get('filename', '?')}:{inner.get('line', '?')} {inner.get('name', '?')}"
    parts = [f"{f['filename'].split('/')[-1]}:{f['line']} {f['name']}" for f in matched[: 1 + context]]
    return " < ".join(parts)


def analyze(path: str, site: re.Pattern, context: int, device: int | None, peaks: int) -> dict:
    with open(path) as f:
        snap = json.load(f)
    traces = snap["device_traces"]
    if device is None:
        device = max(range(len(traces)), key=lambda d: len(traces[d]))
    events = traces[device]
    final = sum(
        b["size"]
        for seg in snap["segments"]
        if seg.get("device", device) == device
        for b in seg["blocks"]
        if b["state"] == "active_allocated"
    )
    net = sum(e["size"] for e in events if e["action"] == "alloc") - sum(
        e["size"] for e in events if e["action"] == "free_requested"
    )
    start = from_segments = final - net
    recorded = snap.get("memory_history_hook", {}).get("start_allocated")
    if recorded is not None:
        start = recorded
    current, best, best_index, series = start, start, -1, []
    for i, e in enumerate(events):
        if e["action"] == "alloc":
            current += e["size"]
        elif e["action"] == "free_requested":
            current -= e["size"]
        else:
            continue
        series.append((i, current))
        if current > best:
            best, best_index = current, i
    live, freed_before = {}, 0
    for i, e in enumerate(events[: best_index + 1]):
        if e["action"] == "alloc":
            live[e["addr"]] = {**e, "index": i}
        elif e["action"] == "free_requested":
            if live.pop(e["addr"], None) is None:
                freed_before += e["size"]
    groups = defaultdict(lambda: [0, 0])
    for e in live.values():
        g = groups[site_key(e.get("frames", []), site, context)]
        g[0] += e["size"]
        g[1] += 1
    groups["(allocated before the window)"] = [start - freed_before, 0]
    local = []
    for k in range(1, len(series) - 1):
        if series[k][1] >= series[k - 1][1] and series[k][1] > series[k + 1][1]:
            local.append(series[k])
    local.sort(key=lambda x: -x[1])
    return {
        "path": path,
        "device": device,
        "events": len(events),
        "start": start,
        "start_from_segments": from_segments,
        "peak": best,
        "peak_index": best_index,
        "peak_time_us": events[best_index].get("time_us") if best_index >= 0 else None,
        "groups": dict(groups),
        "live": [(site_key(e.get("frames", []), site, context), e["size"], e.get("stream"), e["index"]) for e in live.values()],
        "local_peaks": local[:peaks],
        "recorded_peaks": snap.get("memory_history_hook", {}).get("peaks_before_call", [])
        + [snap.get("memory_history_hook", {}).get("peak_at_dump")],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dump")
    p.add_argument("--diff", default=None)
    p.add_argument("--site", default=r"prime_rl|tools/")
    p.add_argument("--context", type=int, default=2)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--peaks", type=int, default=0)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--blocks", default=None)
    args = p.parse_args()
    site = re.compile(args.site)
    runs = [analyze(path, site, args.context, args.device, args.peaks) for path in [args.dump, args.diff] if path]
    for r in runs:
        print(
            f"{r['path']}: device {r['device']}, {r['events']} events, window start {r['start'] / GIB:.3f} GiB "
            f"(from segments {r['start_from_segments'] / GIB:.3f}), "
            f"peak {r['peak'] / GIB:.3f} GiB at event {r['peak_index']} (t {r['peak_time_us']} us)"
        )
        recorded = [p for p in r["recorded_peaks"] if p is not None]
        if recorded:
            print("  max_memory_allocated per hook interval (GiB): " + ", ".join(f"{p / GIB:.3f}" for p in recorded))
        if r["local_peaks"]:
            print("  local peaks: " + ", ".join(f"{i}: {v / GIB:.3f}" for i, v in r["local_peaks"]))
        if args.blocks:
            pattern = re.compile(args.blocks)
            for key, size, stream, index in sorted(r["live"], key=lambda x: x[3]):
                if pattern.search(key):
                    print(f"  block alloc@{index} stream {stream} {size / GIB:.3f} GiB  {key}")
    if len(runs) == 1:
        rows = sorted(runs[0]["groups"].items(), key=lambda kv: -kv[1][0])[: args.top]
        print(f"{'GiB':>8} {'blocks':>7}  site")
        for key, (size, count) in rows:
            print(f"{size / GIB:8.3f} {count:7d}  {key}")
        return
    a, b = runs[0]["groups"], runs[1]["groups"]
    keys = sorted(set(a) | set(b), key=lambda k: -abs(b.get(k, [0, 0])[0] - a.get(k, [0, 0])[0]))[: args.top]
    print(f"{'A GiB':>8} {'B GiB':>8} {'B-A':>8}  site")
    for key in keys:
        sa, sb = a.get(key, [0, 0])[0], b.get(key, [0, 0])[0]
        print(f"{sa / GIB:8.3f} {sb / GIB:8.3f} {(sb - sa) / GIB:+8.3f}  {key}")
    print(f"{runs[0]['peak'] / GIB:8.3f} {runs[1]['peak'] / GIB:8.3f} {(runs[1]['peak'] - runs[0]['peak']) / GIB:+8.3f}  peak")


if __name__ == "__main__":
    main()
