# /// script
# requires-python = ">=3.10"
# ///
"""GPU stalls at host-side stream waits, and overlap of labeled communication kernels with compute.

Built for FSDP2's reduce-scatter buffer cap: `FSDP::post_backward_rs_wait (...)` is a CPU range that only
enqueues `wait_event` on the compute stream, so its host duration is meaningless. The stall it causes shows on
the GPU: for each wait range, take the last main-stream kernel launched before it and the first launched after
it; the stall is the next kernel's start minus the previous kernel's end. The stall is attributed to the wait
only if a communication kernel ends within `--match-us` of the next kernel's start (the popped one). Otherwise it
is reported as unmatched (host latency, not the wait). `host lag` is how much of the stall the next kernel's own
launch was late, i.e. not caused by the GPU waiting.

Communication kernels (`--comm-kernel` regex) are labeled by the innermost host range enclosing their launch
that matches `--launch-annotation`, mapped through ordered `--comm-label LABEL:REGEX` rules (first match wins;
unmatched ones are labeled by the range name). Each one's overlap is the fraction of its GPU interval covered by
the union of main-stream kernels that do not match `--exclude-compute` (e.g. NCCL kernels on the compute stream).

`comm tail` is how long the last communication kernel runs past the last main-stream compute kernel: a wait
that a larger cap avoids at the last hook can reappear there (e.g. FSDP's final callback waits for every
reduce-scatter), so compare stall plus tail across arms.

The analyzed window is the `--step-index`th CPU-side `user_annotation` named `--step-marker` (default: the last
`backward`). Kernels count if their launch falls inside it. The main stream is the stream with the most kernel time outside `--exclude-compute`.

Usage:
  uv run --script comm_waits.py --trace "ep8 fp32 cap1=traces/ep8-fp32-cap1/trace_0.json.gz" [--trace ...] \\
    --comm-label "E:\\) \\[pg=2\\]$" --comm-label "D:\\) \\[pg=16\\]$" --comm-label "R:mlp\\.router\\)$" \\
    --comm-label "D+E:layers\\.\\d+\\)$" [--json out.json] [--per-wait]
"""

import argparse
import json
import re
from collections import defaultdict

from trace_events import load_trace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--step-marker", default="backward")
    parser.add_argument("--step-index", type=int, default=-1)
    parser.add_argument("--wait-annotation", default=r"^FSDP::post_backward_rs_wait")
    parser.add_argument("--comm-kernel", default=r"ReduceScatter")
    parser.add_argument("--launch-annotation", default=r"^FSDP::post_backward_reduce")
    parser.add_argument("--comm-label", action="append", default=[], help="LABEL:REGEX on the launching range")
    parser.add_argument("--exclude-compute", default=r"^nccl", help="main-stream kernels not counted as compute")
    parser.add_argument("--match-us", type=float, default=200.0, help="tolerance for matching the popped kernel (cross-stream event wake-up plus launch is ~0.1 ms)")
    parser.add_argument("--per-wait", action="store_true", help="print one line per wait range")
    parser.add_argument("--per-comm", default=None, metavar="REGEX",
                        help="print each comm kernel whose launching range matches, in ms from --list-from")
    parser.add_argument("--list-from", default=None, metavar="NAME",
                        help="time origin for --per-comm: GPU start of the comm kernel launched in this range")
    parser.add_argument("--json", default=None, help="write all results here")
    return parser.parse_args()


def union_overlap(lo, hi, intervals):
    """Length of [lo, hi) covered by sorted, merged intervals."""
    covered = 0.0
    for start, end in intervals:
        if end <= lo:
            continue
        if start >= hi:
            break
        covered += min(hi, end) - max(lo, start)
    return covered


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def parse_rules(specs):
    rules = []
    for spec in specs:
        label, pattern = spec.split(":", 1)
        rules.append((label, re.compile(pattern)))
    return rules


def label_for(name, rules):
    return next((label for label, pattern in rules if pattern.search(name)), name)


def analyze_trace(trace, args, rules):
    """Waits and labeled comm kernels in one trace; `args` carries the CLI options (see parse_args)."""
    lo, hi = trace.window(args.step_marker, args.step_index, span="event")
    launched = [k for k in trace.kernels if k["launch"] is not None and lo <= k["launch"]["ts"] < hi]
    main = trace.main_stream(launched, args.exclude_compute)
    stream = [k for k in launched if (k["pid"], k["tid"]) == main]
    stream.sort(key=lambda k: k["launch"]["ts"])
    exclude = re.compile(args.exclude_compute)
    compute = merge([(k["ts"], k["ts"] + k["dur"]) for k in stream if not exclude.search(k["name"])])

    comm_pattern = re.compile(args.comm_kernel)
    launch_pattern = re.compile(args.launch_annotation)
    comms = []
    for k in launched:
        if not comm_pattern.search(k["name"]):
            continue
        owner = trace.enclosing(k["launch"], launch_pattern)
        owner_name = owner["name"] if owner else "?"
        comms.append(
            {
                "label": label_for(owner_name, rules),
                "owner": owner_name,
                "stream": k["tid"],
                "start": k["ts"],
                "end": k["ts"] + k["dur"],
                "dur_ms": k["dur"] / 1e3,
                "overlap": union_overlap(k["ts"], k["ts"] + k["dur"], compute) / k["dur"] if k["dur"] else 1.0,
            }
        )
    comms.sort(key=lambda c: c["start"])

    launch_times = [k["launch"]["ts"] for k in stream]
    waits = []
    for wait in trace.annotations(args.wait_annotation):
        if not lo <= wait["ts"] < hi:
            continue
        before = [k for k, t in zip(stream, launch_times) if t < wait["ts"]]
        after = next((k for k, t in zip(stream, launch_times) if t > wait["ts"] + wait["dur"]), None)
        if not before or after is None:
            continue
        prev_end = max(k["ts"] + k["dur"] for k in before[-50:])
        stall = max(0.0, after["ts"] - prev_end)
        host_lag = max(0.0, min(after["launch"]["ts"], after["ts"]) - prev_end)
        popped = min(comms, key=lambda c: abs(c["end"] - after["ts"]), default=None)
        matched = popped is not None and abs(popped["end"] - after["ts"]) <= args.match_us and stall > 0
        waits.append(
            {
                "wait": wait["name"],
                "gpu_start": prev_end,
                "gpu_end": after["ts"],
                "stall_ms": stall / 1e3,
                "host_lag_ms": host_lag / 1e3,
                "matched": matched,
                "popped": popped["owner"] if matched else None,
                "popped_label": popped["label"] if matched else None,
                "nearest": popped["owner"] if popped else None,
                "nearest_end_to_next_ms": (after["ts"] - popped["end"]) / 1e3 if popped else None,
            }
        )
    last_compute_end = compute[-1][1] if compute else lo
    tail = max((c["end"] for c in comms), default=last_compute_end) - last_compute_end
    return {
        "window_ms": (hi - lo) / 1e3,
        "main_stream": main[1],
        "waits": waits,
        "comms": comms,
        "lo": lo,
        "hi": hi,
        "comm_tail_ms": max(0.0, tail) / 1e3,
    }


def summarize(label, result):
    waits = result["waits"]
    matched = [w for w in waits if w["matched"]]
    stall = sum(w["stall_ms"] for w in matched)
    by_label = defaultdict(list)
    for w in matched:
        by_label[w["popped_label"]].append(w["stall_ms"])
    per_comm = defaultdict(list)
    for c in result["comms"]:
        per_comm[c["label"]].append(c)
    print(f"\n{label}: backward window {result['window_ms']:.1f} ms, main stream {result['main_stream']}")
    print(
        f"  waits {len(waits)}, matched {len(matched)}, stall on popped comm {stall:.2f} ms "
        f"(+{sum(w['stall_ms'] for w in waits if not w['matched']):.2f} ms unmatched gaps), "
        f"comm tail after last compute kernel {result['comm_tail_ms']:.2f} ms"
    )
    for popped_label, stalls in sorted(by_label.items()):
        print(f"    stalled on {popped_label:<6} {len(stalls):>3} times, {sum(stalls):8.2f} ms, max {max(stalls):6.2f} ms")
    print(f"  {'comm':<8} {'count':>5} {'total ms':>9} {'mean ms':>8} {'overlap % (time-weighted)':>26}")
    for comm_label, items in sorted(per_comm.items()):
        total = sum(c["dur_ms"] for c in items)
        covered = sum(c["dur_ms"] * c["overlap"] for c in items)
        print(f"  {comm_label:<8} {len(items):>5} {total:>9.1f} {total / len(items):>8.2f} {100 * covered / total:>26.1f}")
    return {
        "stall_ms": stall,
        "comm_tail_ms": result["comm_tail_ms"],
        "stall_by_label": {k: sum(v) for k, v in by_label.items()},
        "overlap_by_label": {
            k: sum(c["dur_ms"] * c["overlap"] for c in v) / sum(c["dur_ms"] for c in v) for k, v in per_comm.items()
        },
        "comm_ms_by_label": {k: sum(c["dur_ms"] for c in v) for k, v in per_comm.items()},
    }


def main():
    args = parse_args()
    rules = parse_rules(args.comm_label)
    output = {}
    for spec in args.trace:
        label, path = spec.split("=", 1)
        result = analyze_trace(load_trace(path), args, rules)
        summary = summarize(label, result)
        if args.per_wait:
            for w in result["waits"]:
                popped = (
                    f"popped {w['popped_label']} ({w['popped']})"
                    if w["matched"]
                    else f"unmatched; nearest comm end {w['nearest_end_to_next_ms']:+.3f} ms before next ({w['nearest']})"
                )
                print(f"    {w['wait'][:70]:<70} stall {w['stall_ms']:7.3f} ms host lag {w['host_lag_ms']:7.3f} {popped}")
        if args.per_comm:
            pattern = re.compile(args.per_comm)
            origin = next((c["start"] for c in result["comms"] if c["owner"] == args.list_from), result["lo"])
            for c in result["comms"]:
                if pattern.search(c["owner"]):
                    print(f"    {(c['start'] - origin) / 1e3:9.3f} to {(c['end'] - origin) / 1e3:9.3f} ms "
                          f"stream {c['stream']:>3} {c['label']:<10} {c['owner']}")
        output[label] = {"path": path, "summary": summary, **result}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(output, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
