#!/usr/bin/env python3
"""Post-process a multi-step PyTorch profiler chrome trace (as written by the SFT
trainer's ``trace_path`` support with per-step ``ProfilerStep#`` frames).

Produces, per input trace:
  - ``steps/step_<n>.json.gz``   one perfetto-loadable trace per training step
  - ``average.json.gz``          synthetic "average step" trace: the CPU/Python call
                                 tree with durations averaged across steps (flamegraph
                                 view in perfetto) plus a flat track of GPU kernels
                                 averaged per step
  - ``summary.json``             per-step wall time, GPU busy time, top ops/kernels

Usage:
    uv run python benchmarks/profiling/trace_tools.py --trace <trace.json.gz> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
TREE_CATS = {"cpu_op", "user_annotation", "python_function", "cuda_runtime", "cuda_driver"}


def load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def dump_trace(trace: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(trace, f)


def find_step_windows(events: list[dict]) -> list[tuple[int, float, float]]:
    """Return (step_number, start_ts, end_ts) for each ProfilerStep# frame."""
    windows = []
    for e in events:
        # The CPU-side annotation defines the step window; ProfilerStep# also shows up as
        # gpu_user_annotation on GPU tracks and as a python_function frame.
        if e.get("ph") == "X" and e.get("cat") == "user_annotation" and e.get("name", "").startswith("ProfilerStep#"):
            step = int(e["name"].removeprefix("ProfilerStep#"))
            windows.append((step, e["ts"], e["ts"] + e["dur"]))
    windows.sort(key=lambda w: w[1])
    # The profiler opens a final stub ProfilerStep# frame right before it stops recording;
    # drop degenerate windows so they don't drag the per-step average down.
    if windows:
        median_dur = sorted(end - start for _, start, end in windows)[len(windows) // 2]
        windows = [w for w in windows if (w[2] - w[1]) > 0.05 * median_dur]
    return windows


def assign_step(windows: list[tuple[int, float, float]], ts: float) -> int | None:
    """Assign an event to a step. GPU work launched in step N can execute past the CPU
    window end, so each step owns [its start, next step's start) and the last step owns
    everything after its start."""
    for i, (step, start, _end) in enumerate(windows):
        next_start = windows[i + 1][1] if i + 1 < len(windows) else float("inf")
        if start <= ts < next_start:
            return step
    return None


def split_steps(trace: dict, out_dir: Path) -> list[Path]:
    events = trace["traceEvents"]
    windows = find_step_windows(events)
    if not windows:
        raise ValueError("No ProfilerStep# frames found in trace")

    meta_events = [e for e in events if e.get("ph") == "M"]
    per_step: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("ph") == "M" or "ts" not in e:
            continue
        step = assign_step(windows, e["ts"])
        if step is not None:
            per_step[step].append(e)

    header = {k: v for k, v in trace.items() if k != "traceEvents"}
    paths = []
    for step, step_events in sorted(per_step.items()):
        out = dict(header)
        out["traceEvents"] = meta_events + step_events
        path = out_dir / "steps" / f"step_{step}.json.gz"
        dump_trace(out, path)
        paths.append(path)
    return paths


class Node:
    __slots__ = ("total_us", "count", "children")

    def __init__(self):
        self.total_us = 0.0
        self.count = 0
        self.children: dict[str, Node] = {}


def build_step_tree(events: list[dict], root: Node) -> None:
    """Accumulate one step's nested CPU/Python events (per thread) into the aggregate tree.

    Standard interval-nesting: events sorted by (ts, -dur); an event is a child of the
    innermost enclosing event on the same thread. Sibling events with the same name merge.
    """
    by_tid: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in TREE_CATS and e.get("dur", 0) > 0:
            if e.get("name", "").startswith("ProfilerStep#"):
                continue
            by_tid[(e["pid"], e["tid"])].append(e)

    for (pid, tid), tid_events in by_tid.items():
        tid_events.sort(key=lambda e: (e["ts"], -e["dur"]))
        thread_name = f"thread {pid}:{tid}"
        thread_node = root.children.setdefault(thread_name, Node())
        thread_node.count = max(thread_node.count, 1)
        stack: list[tuple[float, Node]] = []
        for e in tid_events:
            ts, dur = e["ts"], e["dur"]
            while stack and ts >= stack[-1][0] - 1e-6:
                stack.pop()
            parent = stack[-1][1] if stack else thread_node
            if not stack:
                thread_node.total_us += dur
            node = parent.children.setdefault(e["name"], Node())
            node.total_us += dur
            node.count += 1
            stack.append((ts + dur, node))


def emit_average_tree(root: Node, n_steps: int, pid: int) -> list[dict]:
    """Emit the aggregate tree as synthetic nested X events, durations divided by n_steps."""
    events: list[dict] = []
    tid = 0
    for thread_name, thread_node in sorted(root.children.items(), key=lambda kv: -kv[1].total_us):
        tid += 1
        events.append(
            {"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": f"avg/step {thread_name}"}}
        )

        # Iterative DFS — python stacks nest thousands of frames deep, far past the recursion limit.
        cursor = 0.0
        work: list[tuple[Node, str, float]] = []
        for child_name, child in sorted(thread_node.children.items(), key=lambda kv: kv[1].total_us):
            work.append((child, child_name, cursor))
            cursor += child.total_us / n_steps
        # children are pushed in ascending order so the largest subtree is processed first when popping
        while work:
            node, name, ts = work.pop()
            dur = node.total_us / n_steps
            if dur < 1.0:  # drop sub-microsecond averages to keep the trace small
                continue
            events.append(
                {
                    "ph": "X",
                    "cat": "avg",
                    "name": name,
                    "pid": pid,
                    "tid": tid,
                    "ts": round(ts, 3),
                    "dur": round(dur, 3),
                    "args": {"avg_count_per_step": round(node.count / n_steps, 2), "steps_averaged": n_steps},
                }
            )
            child_ts = ts
            for child_name, child in sorted(node.children.items(), key=lambda kv: -kv[1].total_us):
                work.append((child, child_name, child_ts))
                child_ts += child.total_us / n_steps
    return events


def average_gpu_kernels(events: list[dict], windows) -> dict[tuple, dict[str, dict]]:
    """Aggregate GPU-side events by (device pid, category, kernel name) across steps."""
    agg: dict[tuple, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"total_us": 0.0, "count": 0}))
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in GPU_CATS and "ts" in e:
            if assign_step(windows, e["ts"]) is None:
                continue
            entry = agg[(e["pid"], e["cat"])][e["name"]]
            entry["total_us"] += e.get("dur", 0)
            entry["count"] += 1
    return agg


def emit_average_gpu(agg, n_steps: int, base_pid: int) -> list[dict]:
    events: list[dict] = []
    pid = base_pid
    for (dev_pid, cat), kernels in sorted(agg.items()):
        pid += 1
        events.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "tid": 0,
                "args": {"name": f"avg/step GPU {dev_pid} {cat}"},
            }
        )
        cursor = 0.0
        for name, entry in sorted(kernels.items(), key=lambda kv: -kv[1]["total_us"]):
            dur = entry["total_us"] / n_steps
            if dur < 1.0:
                continue
            events.append(
                {
                    "ph": "X",
                    "cat": "avg_gpu",
                    "name": name,
                    "pid": pid,
                    "tid": 0,
                    "ts": round(cursor, 3),
                    "dur": round(dur, 3),
                    "args": {"avg_count_per_step": round(entry["count"] / n_steps, 2), "steps_averaged": n_steps},
                }
            )
            cursor += dur
    return events


def make_average_trace(trace: dict, out_path: Path) -> dict:
    events = trace["traceEvents"]
    windows = find_step_windows(events)
    if not windows:
        raise ValueError("No ProfilerStep# frames found in trace")
    n_steps = len(windows)

    per_step_events: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("ph") != "X" or "ts" not in e:
            continue
        step = assign_step(windows, e["ts"])
        if step is not None:
            per_step_events[step].append(e)

    root = Node()
    for step in sorted(per_step_events):
        build_step_tree(per_step_events[step], root)

    avg_pid = 1
    out_events: list[dict] = [
        {"ph": "M", "name": "process_name", "pid": avg_pid, "tid": 0, "args": {"name": "avg/step CPU+Python"}}
    ]
    out_events += emit_average_tree(root, n_steps, avg_pid)
    gpu_agg = average_gpu_kernels(events, windows)
    out_events += emit_average_gpu(gpu_agg, n_steps, base_pid=1000)

    out = {"traceEvents": out_events, "displayTimeUnit": "ms", "steps_averaged": n_steps}
    dump_trace(out, out_path)
    return out


def summarize(trace: dict, out_path: Path) -> dict:
    events = trace["traceEvents"]
    windows = find_step_windows(events)
    n_steps = len(windows)

    gpu_busy_per_step: dict[int, float] = defaultdict(float)
    kernel_totals: dict[str, dict] = defaultdict(lambda: {"total_us": 0.0, "count": 0})
    annotation_totals: dict[str, float] = defaultdict(float)
    for e in events:
        if e.get("ph") != "X" or "ts" not in e:
            continue
        step = assign_step(windows, e["ts"])
        if step is None:
            continue
        if e.get("cat") in GPU_CATS:
            gpu_busy_per_step[step] += e.get("dur", 0)
            entry = kernel_totals[e["name"]]
            entry["total_us"] += e.get("dur", 0)
            entry["count"] += 1
        elif e.get("cat") == "user_annotation" and not e["name"].startswith("ProfilerStep#"):
            annotation_totals[e["name"]] += e.get("dur", 0)

    top_kernels = sorted(kernel_totals.items(), key=lambda kv: -kv[1]["total_us"])[:40]
    summary = {
        "num_steps": n_steps,
        "steps": [
            {
                "step": step,
                "wall_ms": round((end - start) / 1e3, 3),
                "gpu_busy_ms": round(gpu_busy_per_step.get(step, 0.0) / 1e3, 3),
            }
            for step, start, end in windows
        ],
        "avg_wall_ms": round(sum(end - start for _, start, end in windows) / n_steps / 1e3, 3),
        "avg_gpu_busy_ms": round(sum(gpu_busy_per_step.values()) / n_steps / 1e3, 3),
        "avg_annotation_ms": {
            name: round(total / n_steps / 1e3, 3) for name, total in sorted(annotation_totals.items())
        },
        "top_gpu_kernels_avg_per_step": [
            {
                "name": name,
                "avg_ms": round(entry["total_us"] / n_steps / 1e3, 3),
                "avg_count": round(entry["count"] / n_steps, 1),
            }
            for name, entry in top_kernels
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def process_trace(trace_path: Path, out_dir: Path) -> dict:
    trace = load_trace(trace_path)
    step_paths = split_steps(trace, out_dir)
    make_average_trace(trace, out_dir / "average.json.gz")
    summary = summarize(trace, out_dir / "summary.json")
    print(f"Wrote {len(step_paths)} per-step traces, average.json.gz and summary.json to {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = process_trace(args.trace, args.out_dir)
    print(json.dumps({k: summary[k] for k in ("num_steps", "avg_wall_ms", "avg_gpu_busy_ms")}, indent=2))


if __name__ == "__main__":
    main()
