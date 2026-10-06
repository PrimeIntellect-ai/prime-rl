# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Before/after timeline figure from torch.profiler chrome traces, in the style of prime-rl PR #3638.

One panel per arm shows the main GPU stream's kernels in a window around an anchor kernel (the Nth launch
of that kernel inside the selected step), colored by category, with every arm aligned at the anchor. Without
`--anchor`, arms align at the step start. With two arms, a bottom panel shows the per-category change in
main-stream kernel time over the whole selected step (`--no-delta` omits it, e.g. when one rank's
kernel savings do not reach step time and the panel would mislead).

Steps start at the CPU-side `user_annotation` events named `--step-marker`; `--step-index` picks one (default
-1, the last) and the step runs until the next marker or the end of the trace. The main stream is the stream
with the most kernel time. Categories are `label:regex:color`, matched in order against kernel names;
unmatched kernels fall into "other".

`--anchor-annotation NAME` aligns at the Nth (`--anchor-index`) GPU-side `record_function` annotation with
exactly that name instead of a kernel, e.g. `FSDP::all_gather_copy_out (model.layers.2)`, which starts a
layer's compute; a window starting at 0 then left-aligns every arm. `--end-annotation NAME` marks the first
such annotation after the anchor as the end of a region: each panel gets a dashed line and the region's
length, and with two arms an arrow on the second panel labels the difference.

`--annotate-gaps MIN_MS` shades every interval of at least MIN_MS in the window where no stream of that GPU
(or, with `--gap-scope main`, not the main stream) runs a kernel, labeled with its length and, if one exists, the shortest CPU op or annotation spanning the
whole gap (what the host was inside). Gap details and all spanning CPU events are printed. A dashed line
marks the end of the step when it falls inside the window.

Communication row (`--comm-kernel REGEX`): each panel gets a second row with that GPU's kernels matching the
regex from other streams, labeled by the host range enclosing their launch (`--launch-annotation`) through
ordered `--comm-label LABEL:REGEX:COLOR` rules. `--wait-annotation REGEX` shades the GPU stalls those host waits
cause on the main stream (computed as in comm_waits.py, over the last `--comm-step-marker` range), labeled
with what was popped. `--anchor-comm-owner NAME` aligns arms at the GPU start of the first communication
kernel launched inside a host range named exactly NAME (e.g. a layer's expert reduce-scatter).

Phase coloring (`--segment LABEL:REGEX:COLOR[:end]`, repeatable): main-stream kernels not matched by a
`--category` take the color of the phase their launch falls in. Phases form a cycle in the given order: phase
i+1 starts at the start (or, with `:end`, the end) of the next host range matching its regex once phase i has
started; kernels before the first phase are "other". E.g. recompute / MLP backward / attention backward:
  --segment "recompute:^FSDP::pre_backward [(]model[.]layers[.][0-9]+[)]( [[]pg=2[]])?$:#cc79a7" \\
  --segment "MLP backward:^autograd::engine::evaluate_function: MmBackward0:#0072b2:end" \\
  --segment "attention backward:evaluate_function: RMSNormFunctionBackward:#009e73"

Usage:
  Preferred layout, left-aligned at the start of a region with its end marked (see reporting/figures.md):
  uv run --script timeline_figure.py --out fig.png --window -2 100 \\
    --anchor-annotation "FSDP::all_gather_copy_out (model.layers.2)" --end-annotation "<annotation ending it>" \\
    --trace "Before: ...=before/trace_0.json.gz" --trace "After: ...=after/trace_0.json.gz" --category ...
  Aligned at a kernel inside the region (reads worse: the axis runs negative):
  uv run --script timeline_figure.py --out fig.png --anchor main_kernel --anchor-index 2 \\
    --trace "Before: main (abc1234)=before/trace_0.json.gz" --trace "After: PR (def5678)=after/trace_0.json.gz" \\
    --category "torch.cat:CatArrayBatchedCopy:#d55e00" --category "GEMM:^nvjet|cutlass(?!.*rmsnorm):#0072b2" \\
    --window -65 25 --title "Model, scale, shared settings"
  Whole-step stall view, aligned at the start of the third traced step:
  uv run --script timeline_figure.py --out stall.png --step-index 2 --window -200 30000 --annotate-gaps 1000 \\
    --trace "Before: ...=before/trace_0.json.gz" --trace "After: ...=after/trace_0.json.gz" --title "..."
"""

import argparse
import re
from collections import defaultdict
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from comm_waits import analyze_trace
from trace_events import load_trace

OTHER = ("other", None, "#999999")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace", action="append", required=True, help="LABEL=PATH, in panel order")
    parser.add_argument("--anchor", default=None, help="kernel name to align arms on (default: step start)")
    parser.add_argument("--anchor-index", type=int, default=0, help="which occurrence of the anchor in the step")
    parser.add_argument("--anchor-annotation", default=None, help="GPU annotation name to align arms on")
    parser.add_argument("--end-annotation", default=None, help="GPU annotation name that ends the marked region")
    parser.add_argument("--region-label", default="region", help="wording for the marked region")
    parser.add_argument("--step-index", type=int, default=-1, help="which step-marker event starts the step")
    parser.add_argument("--annotate-gaps", type=float, default=None, metavar="MIN_MS", help="mark GPU idle gaps")
    parser.add_argument("--gap-scope", choices=["device", "main"], default="device",
                        help="idle means no kernel on any stream of the GPU, or none on the main stream")
    parser.add_argument("--window", type=float, nargs=2, default=(-50.0, 50.0), metavar=("LO_MS", "HI_MS"))
    parser.add_argument("--category", action="append", default=[], help="label:regex:color, matched in order")
    parser.add_argument("--step-marker", default="forward")
    parser.add_argument("--title", default="")
    parser.add_argument("--anchor-label", default=None, help="x-axis wording for the anchor")
    parser.add_argument("--no-delta", action="store_true", help="omit the per-category delta panel")
    parser.add_argument("--segment", action="append", default=[], help="LABEL:REGEX:COLOR[:end], a phase cycle")
    parser.add_argument("--comm-kernel", default=None, help="draw a communication row for kernels matching this")
    parser.add_argument("--comm-label", action="append", default=[], help="LABEL:REGEX:COLOR on the launching range")
    parser.add_argument("--launch-annotation", default=r"^FSDP::post_backward_reduce")
    parser.add_argument("--wait-annotation", default=None, help="shade GPU stalls caused by these host waits")
    parser.add_argument("--comm-step-marker", default="backward", help="host range the comm analysis covers")
    parser.add_argument("--match-us", type=float, default=200.0)
    parser.add_argument("--exclude-compute", default=r"^nccl")
    parser.add_argument("--main-exclude", default=None,
                        help="ignore kernels matching this when picking the main stream (e.g. ^nccl)")
    parser.add_argument("--anchor-comm-owner", default=None, help="align at the comm kernel launched in this range")
    parser.add_argument("--comm-row-label", default="reduce-scatter", help="y-axis label of the communication row")
    parser.add_argument("--comm-legend-suffix", default=" reduce-scatter", help="appended to comm labels in the legend")
    return parser.parse_args()


def parse_categories(specs):
    categories = []
    for spec in specs:
        label, rest = spec.split(":", 1)
        pattern, color = rest.rsplit(":", 1)
        categories.append((label, re.compile(pattern), color))
    return categories + [OTHER]


def categorize(name, categories):
    return next((label, color) for label, pattern, color in categories if pattern is None or pattern.search(name))


def load_step(path, step_marker, step_index, main_exclude=None):
    trace = load_trace(path)
    start, end = trace.window(step_marker, step_index, span="next")
    kernels = [e for e in trace.kernels if start <= e["ts"] < end]
    main = trace.main_stream(kernels, main_exclude)
    stream = [e for e in kernels if (e["pid"], e["tid"]) == main]
    device = [e for e in kernels if e["pid"] == main[0]]
    annotations = [e for e in trace.gpu_annotations if start <= e["ts"] < end]
    return {
        "start": start,
        "end": end,
        "stream": stream,
        "device": device,
        "host": trace.host,
        "annotations": annotations,
        "trace": trace,
    }


def parse_segments(specs):
    segments = []
    for spec in specs:
        parts = spec.split(":")
        use_end = parts[-1] == "end"
        if use_end:
            parts = parts[:-1]
        label, color = parts[0], parts[-1]
        segments.append((label, re.compile(":".join(parts[1:-1])), color, use_end))
    return segments


def assign_segments(stream, host, segments):
    """Map id(kernel) -> (label, color) by the phase cycle its launch falls in."""
    boundaries = []
    for index, (_, pattern, _, use_end) in enumerate(segments):
        for e in host:
            if pattern.search(e["name"]):
                boundaries.append((e["ts"] + e.get("dur", 0) if use_end else e["ts"], index, e["ts"]))
    boundaries.sort()
    phases, current, current_start, cursor = {}, -1, float("-inf"), 0
    for kernel in sorted((k for k in stream if k.get("launch")), key=lambda k: k["launch"]["ts"]):
        while cursor < len(boundaries) and boundaries[cursor][0] <= kernel["launch"]["ts"]:
            at, index, range_start = boundaries[cursor]
            # An `:end` boundary only counts for a range opened after the current phase began.
            if index == (current + 1) % len(segments) and range_start >= current_start:
                current, current_start = index, at
            cursor += 1
        if current >= 0:
            phases[id(kernel)] = segments[current][0], segments[current][2]
    return phases


def comm_analysis(step, args):
    options = SimpleNamespace(
        step_marker=args.comm_step_marker,
        step_index=args.step_index,
        wait_annotation=args.wait_annotation or r"(?!)",
        comm_kernel=args.comm_kernel,
        launch_annotation=args.launch_annotation,
        exclude_compute=args.exclude_compute,
        match_us=args.match_us,
    )
    rules = [(label, pattern) for label, pattern, _ in parse_comm_labels(args.comm_label)]
    return analyze_trace(step["trace"], options, rules)


def parse_comm_labels(specs):
    labels = []
    for spec in specs:
        label, rest = spec.split(":", 1)
        pattern, color = rest.rsplit(":", 1)
        labels.append((label, re.compile(pattern), color))
    return labels


def legend_count(args):
    return len(args.category) + len(args.segment) + len(args.comm_label) + 1


def kernel_label(kernel, categories, phases):
    label, color = categorize(kernel["name"], categories)
    if label == OTHER[0] and id(kernel) in phases:
        return phases[id(kernel)]
    return label, color


def draw_comm_row(ax, comm, anchor_ts, lo, hi, comm_labels, row_label="reduce-scatter"):
    """Second row (y 1.25 to 2.25) with labeled communication kernels, plus shaded stalls on the main row."""
    colors = {label: color for label, _, color in comm_labels}
    for c in comm["comms"]:
        t0, t1 = (c["start"] - anchor_ts) / 1e3, (c["end"] - anchor_ts) / 1e3
        if t1 < lo or t0 > hi:
            continue
        color = colors.get(c["label"], OTHER[2])
        ax.broken_barh([(t0, t1 - t0)], (1.25, 1), facecolors=color, edgecolor="white", linewidth=0.5)
        if t1 - t0 > 0.03 * (hi - lo):
            layer = re.search(r"layers\.(\d+)", c["owner"])
            text = c["label"] + (f" L{layer.group(1)}" if layer else "")
            ax.text((max(t0, lo) + min(t1, hi)) / 2, 1.75, text, ha="center", va="center", fontsize=8, clip_on=True)
    for w in comm["waits"]:
        if not w["matched"]:
            continue
        x0, x1 = (w["gpu_start"] - anchor_ts) / 1e3, (w["gpu_end"] - anchor_ts) / 1e3
        if x1 < lo or x0 > hi:
            continue
        ax.axvspan(x0, x1, ymin=0, ymax=1 / 2.35, color="#cc0000", alpha=0.25, linewidth=0)
        ax.text((x0 + x1) / 2, -0.25, f"stall {x1 - x0:.1f} ms\nwaits on {w['popped_label']}", ha="center",
                va="top", fontsize=8, color="#cc0000", clip_on=False)
    ax.set_ylim(-0.1, 2.35)
    ax.set_yticks([0.5, 1.75])
    ax.set_yticklabels(["compute stream", row_label], fontsize=9)


def idle_gaps(kernels, lo, hi, min_us):
    gaps, busy_until = [], None
    for e in kernels:
        if e["ts"] + e["dur"] < lo or e["ts"] > hi:
            continue
        if busy_until is not None and e["ts"] - busy_until >= min_us:
            gaps.append((busy_until, e["ts"]))
        busy_until = max(busy_until or 0.0, e["ts"] + e["dur"])
    return gaps


def format_us(us):
    return f"{us / 1e6:.2f} s" if us >= 1e6 else f"{us / 1e3:.1f} ms"


def annotate_gaps(ax, step, anchor_ts, lo, hi, min_ms, label, scope="device"):
    kernels = step["device"] if scope == "device" else step["stream"]
    gaps = idle_gaps(kernels, anchor_ts + lo * 1e3, anchor_ts + hi * 1e3, min_ms * 1e3)
    print(f"{label}: {len(gaps)} {'GPU' if scope == 'device' else 'main-stream'} idle gaps >= {min_ms:g} ms")
    for gap_start, gap_end in gaps:
        spanning = sorted(
            (e for e in step["host"] if e["ts"] <= gap_start and e["ts"] + e["dur"] >= gap_end), key=lambda e: e["dur"]
        )
        print(f"  {format_us(gap_end - gap_start)} at {(gap_start - anchor_ts) / 1e3:.1f} ms")
        for e in spanning:
            print(f"    spanned by {e['cat']} {e['name']} ({format_us(e['dur'])})")
        x0, x1 = (gap_start - anchor_ts) / 1e3, (gap_end - anchor_ts) / 1e3
        ax.axvspan(x0, x1, color="#cc0000", alpha=0.12, linewidth=0)
        ax.annotate("", xy=(x0, 0.85), xytext=(x1, 0.85), arrowprops=dict(arrowstyle="<->", color="#cc0000"))
        text = f"{'GPU' if scope == 'device' else 'stream'} idle {format_us(gap_end - gap_start)}"
        if spanning:
            text += f"\nhost inside {spanning[0]['name']}"
        ax.text((x0 + x1) / 2, 0.45, text, ha="center", va="center", fontsize=10, color="#cc0000", clip_on=True)


def main():
    args = parse_args()
    categories = parse_categories(args.category)
    segments = parse_segments(args.segment)
    comm_labels = parse_comm_labels(args.comm_label)
    arms = []
    for spec in args.trace:
        label, path = spec.split("=", 1)
        arms.append((label, load_step(path, args.step_marker, args.step_index, args.main_exclude)))

    show_delta = len(arms) == 2 and not args.no_delta
    heights = [1] * len(arms) + ([1.4] if show_delta else [])
    fig = plt.figure(figsize=(14, 2.8 * len(heights) + 0.6))
    legend_rows = -(-legend_count(args) // 6)
    fig_height = 2.8 * len(heights) + 0.6
    grid = fig.add_gridspec(
        len(heights), 1, height_ratios=heights, hspace=0.6, top=1 - (0.45 + 0.3 * legend_rows) / fig_height
    )
    lo, hi = args.window
    totals, region_ends, axes = [], [], []
    for row, (label, step) in enumerate(arms):
        ax = fig.add_subplot(grid[row])
        stream = step["stream"]
        comm = comm_analysis(step, args) if args.comm_kernel else None
        phases = assign_segments(stream, step["host"], segments) if segments else {}
        if args.anchor_comm_owner:
            anchor_ts = next(c["start"] for c in comm["comms"] if c["owner"] == args.anchor_comm_owner)
        elif args.anchor_annotation:
            anchor_ts = [e for e in step["annotations"] if e["name"] == args.anchor_annotation][args.anchor_index]["ts"]
        elif args.anchor:
            anchor_ts = [e for e in stream if e["name"] == args.anchor][args.anchor_index]["ts"]
        else:
            anchor_ts = step["start"]
        for e in stream:
            t0 = (e["ts"] - anchor_ts) / 1e3
            if lo <= t0 + e["dur"] / 1e3 and t0 <= hi:
                color = kernel_label(e, categories, phases)[1]
                ax.broken_barh([(t0, e["dur"] / 1e3)], (0, 1), facecolors=color, edgecolor="white", linewidth=0.3)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        if comm is not None:
            draw_comm_row(ax, comm, anchor_ts, lo, hi, comm_labels, args.comm_row_label)
            print(f"{label}: " + ", ".join(
                f"stall {w['stall_ms']:.2f} ms on {w['popped_label']}" for w in comm["waits"] if w["matched"]
                and lo <= (w["gpu_start"] - anchor_ts) / 1e3 <= hi
            ))
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(label, loc="left", fontsize=11)
        if args.annotate_gaps is not None:
            annotate_gaps(ax, step, anchor_ts, lo, hi, args.annotate_gaps, label, args.gap_scope)
        if args.end_annotation:
            end_ts = next(e["ts"] for e in step["annotations"] if e["name"] == args.end_annotation and e["ts"] > anchor_ts)
            region_end = (end_ts - anchor_ts) / 1e3
            region_ends.append(region_end)
            ax.axvline(region_end, color="black", linestyle="--", linewidth=1)
            ax.text(region_end, 1.04, f"{args.region_label}: {region_end:.1f} ms", ha="center", va="bottom", fontsize=10,
                    transform=ax.get_xaxis_transform())
            print(f"{label}: {args.region_label} {region_end:.1f} ms")
        axes.append(ax)
        step_end = (step["end"] - anchor_ts) / 1e3
        if lo < step_end < hi:
            ax.axvline(step_end, color="black", linestyle="--", linewidth=1)
            ax.text(step_end, 0.5, " step end", ha="left", va="center", fontsize=10)
        per_category = defaultdict(float)
        for e in stream:
            per_category[kernel_label(e, categories, phases)[0]] += e["dur"] / 1e3
        totals.append(per_category)
        if row == len(arms) - 1:
            if args.anchor_comm_owner:
                default_anchor = f"the GPU start of the comm kernel launched in {args.anchor_comm_owner}"
            elif args.anchor_annotation:
                default_anchor = args.anchor_annotation
            elif args.anchor:
                default_anchor = f"launch {args.anchor_index} of {args.anchor}"
            else:
                default_anchor = "step start"
            anchor_label = args.anchor_label or default_anchor
            step_label = "last step" if args.step_index == -1 else f"step index {args.step_index}"
            ax.set_xlabel(f"ms relative to {anchor_label} ({step_label}, main GPU stream)")

    if len(region_ends) == 2:
        (before_end, after_end), ax = region_ends, axes[1]
        ax.axvline(before_end, color="#cc0000", linestyle=":", linewidth=1)
        ax.annotate("", xy=(after_end, 0.5), xytext=(before_end, 0.5), arrowprops=dict(arrowstyle="<->", color="#cc0000"))
        ax.text((before_end + after_end) / 2, 0.62, f"{after_end - before_end:+.1f} ms", ha="center", va="bottom",
                fontsize=11, color="#cc0000", fontweight="bold")

    legend_entries = [(label, color) for label, _, color in categories[:-1]]
    legend_entries += [(label, color) for label, _, color, _ in segments]
    legend_entries += [(f"{label}{args.comm_legend_suffix}", color) for label, _, color in comm_labels]
    legend_entries.append(OTHER[::2])
    comm_legend = {f"{label}{args.comm_legend_suffix}" for label, _, _ in comm_labels}
    labels = [label for label, _ in legend_entries if label not in comm_legend]
    if show_delta:
        ax = fig.add_subplot(grid[len(arms)])
        before, after = totals
        deltas = [after.get(k, 0.0) - before.get(k, 0.0) for k in labels]
        ax.barh(labels, deltas, color=[dict(legend_entries)[label] for label in labels])
        span = max(abs(d) for d in deltas) or 1.0
        for y, d in enumerate(deltas):
            offset = 0.02 * span if d >= 0 else -0.02 * span
            ax.text(d + offset, y, f"{d:+.1f} ms", va="center", ha="left" if d >= 0 else "right", fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlim(-1.35 * span, 1.35 * span)
        total_before, total_after = sum(before.values()), sum(after.values())
        ax.set_title(
            f"Main-stream kernel time per step, {arms[1][0].split(':')[0]} minus {arms[0][0].split(':')[0]}: "
            f"{total_after - total_before:+.1f} ms ({total_before:.0f} -> {total_after:.0f} ms)",
            loc="left",
            fontsize=11,
        )
        for label, d in zip(labels, deltas):
            print(f"{label:24s} {before.get(label, 0.0):9.1f} -> {after.get(label, 0.0):9.1f} ms  ({d:+.1f})")

    handles = [Patch(color=color, label=label) for label, color in legend_entries]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6), frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.35 / fig_height))
    if args.title:
        fig.suptitle(args.title, fontsize=13, y=1 - 0.05 / fig_height)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
