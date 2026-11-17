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

Reserved memory (`torch.cuda.memory_reserved()`, what `max_memory_reserved()` and `nvidia-smi` see) is replayed too:
`segment_alloc` and `segment_map` (expandable segments) events add their size, `segment_free` and `segment_unmap`
subtract it, starting from the final segments' total size minus the window's net change, or the hook's
`memory_reserved()` at the start when the dump holds it. The header prints the reserved peak, the allocated bytes at
that event, and the reserved-minus-allocated gap (cached or fragmented bytes). `--at reserved` groups the blocks live
at the reserved peak instead of the allocated peak (default `--at allocated`). `--before-sizes` splits "allocated
before the window" into blocks by size and by the site that frees them in the window (the hook records no allocation
stacks for them, so size and free site are their fingerprint), `--blocks` also lists them with the event index of
their free, `--size BYTES` lists every `alloc` and `free_requested` of blocks of exactly that size in the window with
its event index, stream and site (to follow a persistent block's size to where it is created or freed elsewhere), and
`--segments N` lists the N largest mapped address ranges at the chosen peak with their allocated bytes, bytes freed
but awaiting a cross-stream `free_completed`, free bytes, and largest free gap.

Usage: uv run --script memory_peak.py A.json [--diff B.json] [--site REGEX] [--context 2] [--top 30] [--peaks 0]
         [--device N] [--blocks REGEX] [--at allocated|reserved] [--before-sizes] [--segments 0] [--size BYTES]
"""

import argparse
import json
import re
from collections import Counter, defaultdict

GIB = 1024**3
RESERVE_ACTIONS = {"segment_alloc": 1, "segment_map": 1, "segment_free": -1, "segment_unmap": -1}


def site_key(frames, site: re.Pattern, context: int) -> str:
    matched = [f for f in frames if site.search(f.get("filename", ""))]
    if not matched:
        inner = frames[0] if frames else {}
        return f"(no matching frame) {inner.get('filename', '?')}:{inner.get('line', '?')} {inner.get('name', '?')}"
    parts = [f"{f['filename'].split('/')[-1]}:{f['line']} {f['name']}" for f in matched[: 1 + context]]
    return " < ".join(parts)


def add_range(ranges: list, start: int, end: int) -> list:
    merged = []
    for s, e in sorted(ranges + [(start, end)]):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def remove_range(ranges: list, start: int, end: int) -> list:
    kept = []
    for s, e in ranges:
        if e <= start or s >= end:
            kept.append((s, e))
            continue
        if s < start:
            kept.append((s, start))
        if e > end:
            kept.append((end, e))
    return kept


def segment_layout(final_ranges: list, events: list, target: int, occupied: dict, pending: dict) -> list:
    ranges = list(final_ranges)
    for e in reversed(events[target + 1 :]):
        sign = RESERVE_ACTIONS.get(e["action"])
        if sign == 1:
            ranges = remove_range(ranges, e["addr"], e["addr"] + e["size"])
        elif sign == -1:
            ranges = add_range(ranges, e["addr"], e["addr"] + e["size"])
    blocks = sorted([(a, s, False) for a, s in occupied.items()] + [(a, s, True) for a, s in pending.items()])
    layout, k = [], 0
    for start, end in ranges:
        allocated = waiting = gap = 0
        cursor = start
        while k < len(blocks) and blocks[k][0] < start:
            k += 1
        while k < len(blocks) and blocks[k][0] < end:
            addr, size, is_pending = blocks[k]
            gap = max(gap, addr - cursor)
            cursor = max(cursor, addr + size)
            if is_pending:
                waiting += size
            else:
                allocated += size
            k += 1
        gap = max(gap, end - cursor)
        layout.append((start, end - start, allocated, waiting, end - start - allocated - waiting, gap))
    return layout


def analyze(
    path: str,
    site: re.Pattern,
    context: int,
    device: int | None,
    peaks: int,
    at: str = "allocated",
    before_sizes: bool = False,
    segments: int = 0,
    size: int | None = None,
) -> dict:
    with open(path) as f:
        snap = json.load(f)
    traces = snap["device_traces"]
    if device is None:
        device = max(range(len(traces)), key=lambda d: len(traces[d]))
    events = traces[device]
    hook = snap.get("memory_history_hook", {})
    device_segments = [seg for seg in snap["segments"] if seg.get("device", device) == device]
    final_blocks = {
        b["address"]: b["size"] for seg in device_segments for b in seg["blocks"] if b["state"] == "active_allocated"
    }
    final = sum(final_blocks.values())
    net = sum(e["size"] for e in events if e["action"] == "alloc") - sum(
        e["size"] for e in events if e["action"] == "free_requested"
    )
    start = from_segments = final - net
    if hook.get("start_allocated") is not None:
        start = hook["start_allocated"]
    final_reserved = sum(seg["total_size"] for seg in device_segments)
    reserved_net = sum(RESERVE_ACTIONS.get(e["action"], 0) * e.get("size", 0) for e in events)
    reserved_start = reserved_from_segments = final_reserved - reserved_net
    if hook.get("start_reserved") is not None:
        reserved_start = hook["start_reserved"]
    current, best, best_index, series = start, start, -1, []
    reserved, reserved_best, reserved_best_index, allocated_at_reserved_best = reserved_start, reserved_start, -1, start
    for i, e in enumerate(events):
        sign = RESERVE_ACTIONS.get(e["action"])
        if sign is not None:
            reserved += sign * e["size"]
            if reserved > reserved_best:
                reserved_best, reserved_best_index, allocated_at_reserved_best = reserved, i, current
            continue
        if e["action"] == "alloc":
            current += e["size"]
        elif e["action"] == "free_requested":
            current -= e["size"]
        else:
            continue
        series.append((i, current))
        if current > best:
            best, best_index = current, i
    target = best_index if at == "allocated" else reserved_best_index
    live, end_live, before_freed, freed_before = {}, {}, [], 0
    for i, e in enumerate(events):
        if e["action"] == "alloc":
            end_live[e["addr"]] = e
        elif e["action"] == "free_requested" and end_live.pop(e["addr"], None) is None:
            before_freed.append((i, e["addr"], e["size"], site_key(e.get("frames", []), site, context)))
    occupied, pending = {}, {}
    for i, e in enumerate(events[: target + 1]):
        if e["action"] == "alloc":
            live[e["addr"]] = {**e, "index": i}
            occupied[e["addr"]] = e["size"]
        elif e["action"] == "free_requested":
            if live.pop(e["addr"], None) is None:
                freed_before += e["size"]
            pending[e["addr"]] = occupied.pop(e["addr"], e["size"])
        elif e["action"] == "free_completed":
            pending.pop(e["addr"], None)
    before_live = [(a, s, -1, "not freed in the window") for a, s in final_blocks.items() if a not in end_live]
    before_live += [(a, s, i, f"freed at {key}") for i, a, s, key in before_freed if i > target]
    for a, s, _, _ in before_live:
        occupied.setdefault(a, s)
    groups = defaultdict(lambda: [0, 0])
    for e in live.values():
        g = groups[site_key(e.get("frames", []), site, context)]
        g[0] += e["size"]
        g[1] += 1
    before_bytes = start - freed_before
    if before_sizes:
        for _, size, _, freed in before_live:
            g = groups[f"(allocated before the window) {size} B blocks, {freed}"]
            g[0] += size
            g[1] += 1
        before_bytes -= sum(size for _, size, _, _ in before_live)
        if before_bytes:
            groups["(allocated before the window, not in the final segments)"] = [before_bytes, 0]
    else:
        groups["(allocated before the window)"] = [before_bytes, 0]
    local = []
    for k in range(1, len(series) - 1):
        if series[k][1] >= series[k - 1][1] and series[k][1] > series[k + 1][1]:
            local.append(series[k])
    local.sort(key=lambda x: -x[1])
    layout = []
    if segments:
        final_ranges = []
        for seg in device_segments:
            final_ranges = add_range(final_ranges, seg["address"], seg["address"] + seg["total_size"])
        layout = segment_layout(final_ranges, events, target, occupied, pending)
    return {
        "path": path,
        "device": device,
        "events": len(events),
        "start": start,
        "start_from_segments": from_segments,
        "peak": best,
        "peak_index": best_index,
        "peak_time_us": events[best_index].get("time_us") if best_index >= 0 else None,
        "reserved_start": reserved_start,
        "reserved_start_from_segments": reserved_from_segments,
        "reserved_peak": reserved_best,
        "reserved_peak_index": reserved_best_index,
        "allocated_at_reserved_peak": allocated_at_reserved_best,
        "reserve_actions": Counter(e["action"] for e in events if e["action"] in RESERVE_ACTIONS),
        "target": target,
        "groups": dict(groups),
        "live": [(site_key(e.get("frames", []), site, context), e["size"], e.get("stream"), e["index"]) for e in live.values()],
        "before_live": [(f"(allocated before the window) {freed}", size, index) for _, size, index, freed in before_live],
        "local_peaks": local[:peaks],
        "recorded_peaks": hook.get("peaks_before_call", []) + [hook.get("peak_at_dump")],
        "recorded_reserved_peaks": hook.get("reserved_peaks_before_call", []) + [hook.get("reserved_peak_at_dump")],
        "layout": sorted(layout, key=lambda x: -x[1])[:segments],
        "sized": [
            (i, e["action"], e.get("stream"), site_key(e.get("frames", []), site, context))
            for i, e in enumerate(events)
            if size is not None and e["action"] in ("alloc", "free_requested") and e.get("size") == size
        ],
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
    p.add_argument("--at", choices=["allocated", "reserved"], default="allocated")
    p.add_argument("--before-sizes", action="store_true")
    p.add_argument("--segments", type=int, default=0)
    p.add_argument("--size", type=int, default=None)
    args = p.parse_args()
    site = re.compile(args.site)
    runs = [
        analyze(path, site, args.context, args.device, args.peaks, args.at, args.before_sizes, args.segments, args.size)
        for path in [args.dump, args.diff]
        if path
    ]
    for r in runs:
        print(
            f"{r['path']}: device {r['device']}, {r['events']} events, window start {r['start'] / GIB:.3f} GiB "
            f"(from segments {r['start_from_segments'] / GIB:.3f}), "
            f"peak {r['peak'] / GIB:.3f} GiB at event {r['peak_index']} (t {r['peak_time_us']} us)"
        )
        recorded = [p for p in r["recorded_peaks"] if p is not None]
        if recorded:
            print("  max_memory_allocated per hook interval (GiB): " + ", ".join(f"{p / GIB:.3f}" for p in recorded))
        actions = ", ".join(f"{k} {v}" for k, v in sorted(r["reserve_actions"].items())) or "none"
        print(
            f"  reserved: window start {r['reserved_start'] / GIB:.3f} GiB "
            f"(from segments {r['reserved_start_from_segments'] / GIB:.3f}), "
            f"peak {r['reserved_peak'] / GIB:.3f} GiB at event {r['reserved_peak_index']}, "
            f"allocated there {r['allocated_at_reserved_peak'] / GIB:.3f} GiB, "
            f"gap {(r['reserved_peak'] - r['allocated_at_reserved_peak']) / GIB:.3f} GiB; segment events: {actions}"
        )
        recorded = [p for p in r["recorded_reserved_peaks"] if p is not None]
        if recorded:
            print("  max_memory_reserved per hook interval (GiB): " + ", ".join(f"{p / GIB:.3f}" for p in recorded))
        if r["local_peaks"]:
            print("  local peaks: " + ", ".join(f"{i}: {v / GIB:.3f}" for i, v in r["local_peaks"]))
        if r["layout"]:
            print(f"  mapped ranges at event {r['target']} (GiB): size, allocated, awaiting free_completed, free, largest gap")
            for start, size, allocated, waiting, free, gap in r["layout"]:
                print(
                    f"    {start:#x} {size / GIB:8.3f} {allocated / GIB:8.3f} {waiting / GIB:8.3f} "
                    f"{free / GIB:8.3f} {gap / GIB:8.3f}"
                )
        for index, action, stream, key in r["sized"]:
            print(f"  {action}@{index} stream {stream} {args.size} B  {key}")
        if args.blocks:
            pattern = re.compile(args.blocks)
            for key, size, stream, index in sorted(r["live"], key=lambda x: x[3]):
                if pattern.search(key):
                    print(f"  block alloc@{index} stream {stream} {size / GIB:.3f} GiB  {key}")
            for key, size, index in sorted(r["before_live"], key=lambda x: x[2]):
                if pattern.search(key):
                    print(f"  block free@{index} {size / GIB:.3f} GiB  {key}")
    if len(runs) == 1:
        rows = sorted(runs[0]["groups"].items(), key=lambda kv: -kv[1][0])[: args.top]
        print(f"blocks live at the {args.at} peak")
        print(f"{'GiB':>8} {'blocks':>7}  site")
        for key, (size, count) in rows:
            print(f"{size / GIB:8.3f} {count:7d}  {key}")
        return
    a, b = runs[0]["groups"], runs[1]["groups"]
    keys = sorted(set(a) | set(b), key=lambda k: -abs(b.get(k, [0, 0])[0] - a.get(k, [0, 0])[0]))[: args.top]
    print(f"blocks live at the {args.at} peak")
    print(f"{'A GiB':>8} {'B GiB':>8} {'B-A':>8}  site")
    for key in keys:
        sa, sb = a.get(key, [0, 0])[0], b.get(key, [0, 0])[0]
        print(f"{sa / GIB:8.3f} {sb / GIB:8.3f} {(sb - sa) / GIB:+8.3f}  {key}")
    print(f"{runs[0]['peak'] / GIB:8.3f} {runs[1]['peak'] / GIB:8.3f} {(runs[1]['peak'] - runs[0]['peak']) / GIB:+8.3f}  peak")
    ra, rb = runs[0]["reserved_peak"], runs[1]["reserved_peak"]
    print(f"{ra / GIB:8.3f} {rb / GIB:8.3f} {(rb - ra) / GIB:+8.3f}  reserved peak")


if __name__ == "__main__":
    main()
