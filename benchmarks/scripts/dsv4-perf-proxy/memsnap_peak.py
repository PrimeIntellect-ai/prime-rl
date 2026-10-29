"""Replay a CUDA memory snapshot's allocation trace, find the allocated-bytes peak, and group the blocks live at it."""

import argparse
import pickle
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

GIB = 1024**3
SEGMENT_DELTA = {"segment_alloc": 1, "segment_map": 1, "segment_free": -1, "segment_unmap": -1}
SITE_FILE_MARKERS = (
    "/prime_rl/",
    "/prime_kernels/",
    "/deep_gemm/",
    "/tilelang/",
    "/torch/distributed/",
    "/torch/optim/",
)
INDUCTOR_ALLOC = re.compile(r"^\s*(buf\w+)\s*=\s*empty_strided_cuda\((\([^)]*\))")


def device_trace(snapshot: dict) -> tuple[int, list[dict]]:
    traces = snapshot["device_traces"]
    device = max(range(len(traces)), key=lambda d: len(traces[d]))
    return device, traces[device]


def blocks_live_at_end(snapshot: dict, device: int) -> dict[int, tuple[int, list[dict]]]:
    live = {}
    for segment in snapshot["segments"]:
        if segment["device"] != device:
            continue
        for block in segment["blocks"]:
            if block["state"].startswith("active"):
                live[block["address"]] = (block["size"], block.get("frames", []))
    return live


def blocks_live_at_start(events: list[dict], end_live: dict) -> dict[int, tuple[int, list[dict]]]:
    """Blocks allocated before the (possibly truncated) trace window, with their free frames if freed inside it."""
    allocated_in_window = set()
    start_live = {}
    for event in events:
        if event["action"] == "alloc":
            allocated_in_window.add(event["addr"])
        elif event["action"] == "free_completed":
            if event["addr"] in allocated_in_window:
                allocated_in_window.remove(event["addr"])
            else:
                start_live[event["addr"]] = (event["size"], event.get("frames", []))
    for address, block in end_live.items():
        if address not in allocated_in_window:
            start_live[address] = block
    return start_live


def replay(events: list[dict], start_live: dict, stop_index: int | None = None, curve: list | None = None):
    live = {address: (size, frames, -1) for address, (size, frames) in start_live.items()}
    allocated = sum(size for size, _, _ in live.values())
    peak, peak_index = allocated, -1
    for index, event in enumerate(events):
        if event["action"] == "alloc":
            live[event["addr"]] = (event["size"], event.get("frames", []), index)
            allocated += event["size"]
        elif event["action"] == "free_completed":
            size, _, _ = live.pop(event["addr"])
            allocated -= size
        if curve is not None:
            curve.append(allocated)
        if allocated > peak:
            peak, peak_index = allocated, index
        if index == stop_index:
            break
    return peak, peak_index, live


def reserved_at(snapshot: dict, device: int, events: list[dict], index: int) -> int:
    reserved = sum(segment["total_size"] for segment in snapshot["segments"] if segment["device"] == device)
    for event in reversed(events[index + 1 :]):
        reserved -= SEGMENT_DELTA.get(event["action"], 0) * event["size"]
    return reserved


def stream_peaks(snapshot: dict, device: int, events: list[dict]) -> dict[int, tuple[int, int]]:
    """Per stream pool: the peak allocated bytes inside the window and the reserved bytes at the end."""
    peaks = {}
    for stream in {segment["stream"] for segment in snapshot["segments"] if segment["device"] == device}:
        segments = [s for s in snapshot["segments"] if s["device"] == device and s["stream"] == stream]
        allocated = peak = sum(segment["allocated_size"] for segment in segments)
        for event in reversed(events):
            if event.get("stream") != stream:
                continue
            if event["action"] == "alloc":
                allocated -= event["size"]
            elif event["action"] == "free_completed":
                allocated += event["size"]
            peak = max(peak, allocated)
        peaks[stream] = (peak, sum(segment["total_size"] for segment in segments))
    return peaks


def python_frames(frames: list[dict]) -> list[dict]:
    return [frame for frame in frames if frame["filename"].endswith(".py")]


@cache
def source_lines(path: str) -> list[str]:
    try:
        return Path(path).read_text().splitlines()
    except OSError:
        return []


def inductor_buffer(path: str, line: int) -> str:
    lines = source_lines(path)
    if not 0 < line <= len(lines):
        return ""
    match = INDUCTOR_ALLOC.match(lines[line - 1])
    if match is None:
        return lines[line - 1].strip()[:120]
    name, shape = match.groups()
    for text in lines[line : line + 12]:
        if "Source Nodes:" in text:
            nodes = text.split("Source Nodes:", 1)[1].split("], Original ATen")[0].strip(" [")
            return f"{name}{shape} <- {nodes[:100]}"
    return f"{name}{shape}"


def allocation_site(frames: list[dict], path_map: list[tuple[str, str]]) -> str:
    for frame in python_frames(frames):
        filename = frame["filename"]
        if "/inductor/" in filename and "/torch/_inductor/" not in filename:
            local = filename
            for old, new in path_map:
                local = local.replace(old, new)
            return f"inductor {Path(filename).name[:10]}:{frame['line']} {inductor_buffer(local, frame['line'])}"
        if any(marker in filename for marker in SITE_FILE_MARKERS):
            short = filename.split("/site-packages/")[-1].split("/src/")[-1]
            return f"{short}:{frame['line']} {frame['name']}"
    top = python_frames(frames)[:1]
    return f"{top[0]['filename'].split('/')[-1]}:{top[0]['line']} {top[0]['name']}" if top else "<no python frames>"


CATEGORY_RULES = (
    ("optimizer state", ("torch/optim/optimizer.py:530",)),
    ("gradients", ("zero_grad", "foreach_reduce", "reduce_scatter")),
    ("FSDP all-gather", ("_fsdp_collectives.py", "_fsdp_param.py")),
    (
        "MoE dispatch/combine",
        (
            "_all_to_all_single",
            "routed_input",
            "top_scores",
            "grouped_fp8_gemm",
            "_grouped_fp8",
            "grouped_per_",
            "_compute_grad_weight",
            "build_grouped_layout",
            "outs",
        ),
    ),
)


def category(phase_name: str, site: str) -> str:
    """Coarse owner of a live block, from where it was allocated or, for blocks older than the window, freed."""
    if "<no python frames>" in site:
        return "parameters and other pre-recording state"
    for name, markers in CATEGORY_RULES:
        if any(marker in site for marker in markers):
            return name
    freed = re.search(r"\bdel (\w+)", site)
    if freed is not None:
        return "saved for backward: " + re.sub(r"_\d+$", "", freed.group(1))
    if phase_name.startswith("train.py") and "forward" in phase_name:
        return "saved for backward: forward-allocated"
    if phase_name in ("backward", "recompute"):
        return "backward/recompute temporaries"
    return "other"


def phase(frames: list[dict]) -> str:
    names = [frame["name"] for frame in frames]
    files = [frame["filename"] for frame in frames]
    if any("/torch/optim/" in f or "optim" in f.split("/")[-1] for f in files) or any(
        "optimizer" in n.lower() for n in names
    ):
        return "optimizer"
    if any(
        f.endswith("torch/utils/checkpoint.py") and ("recompute" in n or "unpack" in n) for f, n in zip(files, names)
    ):
        return "recompute"
    if any("backward" in n.lower() or "Engine::" in n for n in names):
        return "backward"
    for frame in frames:
        if frame["filename"].endswith("trainer/rl/train.py"):
            text = source_lines(frame["filename"])[frame["line"] - 1].strip() if frame["line"] else ""
            return f"train.py:{frame['line']} {text[:60]}"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshots", nargs="+", type=Path, help="rank_<r>.pickle files from torch.cuda.memory._snapshot"
    )
    parser.add_argument("--peaks-only", action="store_true", help="print each snapshot's peak and stop")
    parser.add_argument("--top", type=int, default=40, help="allocation sites to print")
    parser.add_argument("--context", type=int, default=12, help="alloc events to print before the peak")
    parser.add_argument(
        "--timeline-bins", type=int, default=0, help="print the max allocated GiB in this many time bins"
    )
    parser.add_argument(
        "--path-map", action="append", default=[], help="OLD=NEW prefix rewrite to read copied inductor code"
    )
    args = parser.parse_args()
    path_map = [tuple(item.split("=", 1)) for item in args.path_map]

    for path in args.snapshots:
        with open(path, "rb") as handle:
            snapshot = pickle.load(handle)
        device, events = device_trace(snapshot)
        end_live = blocks_live_at_end(snapshot, device)
        start_live = blocks_live_at_start(events, end_live)
        curve = []
        peak, peak_index, _ = replay(events, start_live, curve=curve)
        reserved = reserved_at(snapshot, device, events, peak_index)
        mapped_in_window = sum(SEGMENT_DELTA.get(event["action"], 0) * event["size"] for event in events)
        span_s = (events[-1]["time_us"] - events[0]["time_us"]) / 1e6 if events else 0.0
        peak_s = (events[peak_index]["time_us"] - events[0]["time_us"]) / 1e6 if peak_index >= 0 else 0.0
        print(
            f"{path}: device {device}, {len(events)} events over {span_s:.1f} s, "
            f"{sum(s for s, _ in start_live.values()) / GIB:.2f} GiB live at window start, "
            f"peak allocated {peak / GIB:.2f} GiB at event {peak_index} (t={peak_s:.2f} s), reserved there {reserved / GIB:.2f} GiB, "
            f"end reserved {sum(seg['total_size'] for seg in snapshot['segments'] if seg['device'] == device) / GIB:.2f} GiB "
            f"({mapped_in_window / GIB:+.2f} GiB mapped inside the window)"
        )
        if args.timeline_bins:
            start, span = events[0]["time_us"], max(events[-1]["time_us"] - events[0]["time_us"], 1)
            bins = [0] * args.timeline_bins
            for event, allocated in zip(events, curve):
                position = min((event["time_us"] - start) * args.timeline_bins // span, args.timeline_bins - 1)
                bins[position] = max(bins[position], allocated)
            print("  max allocated GiB per time bin: " + " ".join(f"{b / GIB:.1f}" for b in bins))
        if args.peaks_only:
            continue

        print("per stream pool (stream, peak allocated GiB in window, reserved GiB at end):")
        for stream, (stream_peak, stream_reserved) in sorted(stream_peaks(snapshot, device, events).items()):
            print(f"  {stream:#x} {stream_peak / GIB:8.2f} {stream_reserved / GIB:8.2f}")

        _, _, live = replay(events, start_live, stop_index=peak_index)
        print(f"\nphase of the peak alloc: {phase(events[peak_index].get('frames', []))}")
        print("alloc events before the peak (index, GiB, phase, site):")
        allocs = [i for i in range(max(0, peak_index - 4000), peak_index + 1) if events[i]["action"] == "alloc"]
        for i in allocs[-args.context :]:
            frames = events[i].get("frames", [])
            print(f"  {i} {events[i]['size'] / GIB:7.3f} {phase(frames)[:40]:40s} {allocation_site(frames, path_map)}")

        by_site = defaultdict(lambda: [0, 0])
        by_phase = defaultdict(int)
        by_category = defaultdict(int)
        for size, frames, index in live.values():
            key = (phase(frames) if index >= 0 else "<before window>", allocation_site(frames, path_map))
            by_site[key][0] += size
            by_site[key][1] += 1
            by_phase[key[0]] += size
            by_category[category(*key)] += size
        print(f"\nlive GiB at the peak by category (total {peak / GIB:.2f}):")
        for name, size in sorted(by_category.items(), key=lambda item: -item[1]):
            print(f"  {size / GIB:8.3f}  {name}")
        print(f"\nlive GiB at the peak by allocation phase (total {peak / GIB:.2f}):")
        for name, size in sorted(by_phase.items(), key=lambda item: -item[1]):
            print(f"  {size / GIB:8.3f}  {name}")
        print("\nlive GiB at the peak by allocation site (GiB, blocks, phase, site):")
        for (phase_name, site), (size, count) in sorted(by_site.items(), key=lambda item: -item[1][0])[: args.top]:
            print(f"  {size / GIB:8.3f} {count:5d}  {phase_name[:40]:40s} {site}")
        print()


if __name__ == "__main__":
    main()
