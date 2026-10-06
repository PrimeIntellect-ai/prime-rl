# /// script
# requires-python = ">=3.10"
# ///
"""Shared loading helpers for torch.profiler chrome traces, plus a CLI that lists host ranges in a window.

Library (imported by comm_waits.py and timeline_figure.py):
- `load_trace(path)` returns a `Trace` with complete events split by category, kernels joined to their launch
  (`cuda_runtime`/`cuda_driver` event with the same `args["correlation"]`), and per-thread host stacks.
- `Trace.enclosing(launch, pattern)` returns the innermost host event (cpu_op or user_annotation) on the launch's
  thread that spans the launch and whose name matches `pattern`.
- `Trace.window(marker, index)` returns the [start, end) of the `index`th CPU-side `user_annotation` named
  `marker`, from its start to its end (`span="event"`) or to the next marker of that name (`span="next"`).
- `Trace.main_stream(kernels, exclude)` is the (pid, tid) with the most kernel time among `kernels`, ignoring
  kernels matching `exclude` (pass the NCCL pattern when communication can outweigh compute).

CLI: print every host range on the busiest host thread inside a window, indented by nesting, to learn a
trace's annotation structure before writing category or label regexes.

Usage:
  uv run --script trace_events.py <trace.json.gz> --from-annotation "FSDP::pre_backward (model.layers.12)" \\
    [--from-index -1] [--to-annotation "FSDP::post_backward_reduce (model.layers.12) [pg=16]"] \\
    [--name-filter REGEX] [--min-us 0] [--max-depth 6]
"""

import argparse
import bisect
import gzip
import json
import re
from collections import defaultdict

HOST_CATEGORIES = ("cpu_op", "user_annotation")
LAUNCH_CATEGORIES = ("cuda_runtime", "cuda_driver")


class Trace:
    def __init__(self, events):
        complete = [e for e in events if e.get("ph") == "X"]
        self.kernels = sorted((e for e in complete if e.get("cat") == "kernel"), key=lambda e: e["ts"])
        self.host = [e for e in complete if e.get("cat") in HOST_CATEGORIES]
        self.gpu_annotations = sorted(
            (e for e in complete if e.get("cat") == "gpu_user_annotation"), key=lambda e: e["ts"]
        )
        launches = {}
        for e in complete:
            if e.get("cat") in LAUNCH_CATEGORIES and "correlation" in e.get("args", {}):
                launches[e["args"]["correlation"]] = e
        for kernel in self.kernels:
            kernel["launch"] = launches.get(kernel.get("args", {}).get("correlation"))
        self._by_thread = defaultdict(list)
        for e in self.host:
            self._by_thread[(e["pid"], e["tid"])].append(e)
        self._starts = {}
        for key, thread_events in self._by_thread.items():
            thread_events.sort(key=lambda e: (e["ts"], -e.get("dur", 0)))
            self._starts[key] = [e["ts"] for e in thread_events]

    def annotations(self, name_pattern):
        """CPU-side user annotations whose name matches the pattern, by start time."""
        pattern = re.compile(name_pattern)
        return sorted(
            (e for e in self.host if e.get("cat") == "user_annotation" and pattern.search(e["name"])),
            key=lambda e: e["ts"],
        )

    def window(self, marker, index=-1, span="event"):
        starts = [e for e in self.host if e.get("cat") == "user_annotation" and e["name"] == marker]
        starts.sort(key=lambda e: e["ts"])
        event = starts[index]
        if span == "event":
            return event["ts"], event["ts"] + event["dur"]
        return event["ts"], next((e["ts"] for e in starts if e["ts"] > event["ts"]), float("inf"))

    def enclosing(self, launch, pattern):
        """Innermost host event on the launch's thread that spans it and matches the compiled pattern."""
        if launch is None:
            return None
        key = (launch["pid"], launch["tid"])
        thread_events = self._by_thread.get(key, [])
        upper = bisect.bisect_right(self._starts.get(key, []), launch["ts"])
        for e in reversed(thread_events[:upper]):
            if e["ts"] + e.get("dur", 0) >= launch["ts"] and pattern.search(e["name"]):
                return e
        return None

    def busiest_thread(self, lo, hi):
        busy = defaultdict(float)
        for key, thread_events in self._by_thread.items():
            for e in thread_events:
                if lo <= e["ts"] < hi:
                    busy[key] += e.get("dur", 0)
        return max(busy, key=busy.get)

    def thread_events(self, key):
        return self._by_thread[key]

    @staticmethod
    def main_stream(kernels, exclude=None):
        """Stream with the most kernel time, not counting kernels whose name matches `exclude` (e.g. NCCL)."""
        pattern = re.compile(exclude) if exclude else None
        per_stream = defaultdict(float)
        for e in kernels:
            if pattern is None or not pattern.search(e["name"]):
                per_stream[(e["pid"], e["tid"])] += e["dur"]
        return max(per_stream, key=per_stream.get)


def load_trace(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return Trace(json.load(f)["traceEvents"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace")
    parser.add_argument("--from-annotation", required=True, help="exact CPU user_annotation name opening the window")
    parser.add_argument("--from-index", type=int, default=-1, help="which occurrence opens the window")
    parser.add_argument("--to-annotation", default=None, help="first later annotation with this name closes it")
    parser.add_argument("--name-filter", default=None, help="only print ranges whose name matches this regex")
    parser.add_argument("--min-us", type=float, default=0.0, help="only print ranges at least this long")
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    trace = load_trace(args.trace)
    opener = [e for e in trace.annotations("") if e["name"] == args.from_annotation][args.from_index]
    lo = opener["ts"]
    hi = opener["ts"] + opener["dur"]
    if args.to_annotation:
        hi = next(e["ts"] + e["dur"] for e in trace.annotations("") if e["name"] == args.to_annotation and e["ts"] > lo)
    key = trace.busiest_thread(lo, hi)
    name_filter = re.compile(args.name_filter) if args.name_filter else None
    stack = []
    for e in trace.thread_events(key):
        if not lo <= e["ts"] < hi:
            continue
        while stack and stack[-1] <= e["ts"]:
            stack.pop()
        depth = len(stack)
        stack.append(e["ts"] + e.get("dur", 0))
        if depth > args.max_depth or e.get("dur", 0) < args.min_us:
            continue
        if name_filter and not name_filter.search(e["name"]):
            continue
        print(f"{(e['ts'] - lo) / 1e3:10.3f} ms {e.get('dur', 0) / 1e3:9.3f} ms {'  ' * depth}{e['name'][:120]}")


if __name__ == "__main__":
    main()
