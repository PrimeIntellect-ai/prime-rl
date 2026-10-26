"""Count n-gram table lookups, all-to-alls and host syncs per training step in a torch.profiler trace of an A/B run."""

import argparse
import gzip
import json
import re
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

from graph_breaks import table

STEP_END = re.compile(r"^Optimizer\.step#\w+\.step$")
FSDP_SPAN = re.compile(r"^FSDP::(pre|post)_(forward|backward)\w* \((?P<module>[^)]+)\)")
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)$")
NGRAM = re.compile(r"(?:layers\.(\d+)\.\S*ngram\S*|ngram\.tables\.(\d+))$")
SYNC_CALLS = {
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cuStreamSynchronize",
    "cuCtxSynchronize",
}
COPY_CALLS = {"cudaMemcpyAsync", "cudaMemcpy", "cuMemcpyAsync", "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoH_v2"}
SCALAR_OPS = {"aten::item", "aten::_local_scalar_dense"}
HOST_CATS = {"cpu_op", "user_annotation", "cuda_runtime", "cuda_driver"}
DEFAULT_VOCAB = 248320


def load(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]


def resolve_trace(path: Path, rank: int) -> Path:
    if path.is_file():
        return path
    meta_path = path / "run.json"
    trace_dir = Path(json.loads(meta_path.read_text())["trace_dir"]) if meta_path.exists() else path
    return trace_dir / f"trace_{rank}.json.gz"


def link_parents(events: list[dict]) -> None:
    by_thread = defaultdict(list)
    for e in events:
        if e.get("cat") in HOST_CATS:
            by_thread[e["tid"]].append(e)
    for thread_events in by_thread.values():
        stack: list[dict] = []
        for e in sorted(thread_events, key=lambda e: (e["ts"], -e["dur"])):
            while stack and stack[-1]["ts"] + stack[-1]["dur"] < e["ts"] + e["dur"]:
                stack.pop()
            e["parent"] = stack[-1] if stack else None
            stack.append(e)


def ancestors(event: dict):
    parent = event.get("parent")
    while parent is not None:
        yield parent
        parent = parent.get("parent")


def step_windows(events: list[dict]) -> list[tuple[int, float, float]]:
    ends = sorted(e["ts"] + e["dur"] for e in events if e.get("cat") == "user_annotation" and STEP_END.match(e["name"]))
    start = min(e["ts"] for e in events)
    windows = []
    for step, end in enumerate(ends, start=1):
        windows.append((step, start, end))
        start = end
    return windows


def arg(event: dict, key: str, index: int):
    values = event.get("args", {}).get(key) or []
    return values[index] if index < len(values) else None


def classify(events: list[dict], vocab: int) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    copies = {
        e["args"]["correlation"]: e["name"]
        for e in events
        if e.get("cat") == "gpu_memcpy" and "correlation" in e.get("args", {})
    }
    marks = []
    sync_points = {}
    for e in events:
        cat, name = e.get("cat"), e["name"]
        if cat == "cpu_op" and any(a["name"] == name for a in ancestors(e)):
            continue
        if cat == "cpu_op" and name == "aten::embedding":
            weight = arg(e, "Input Dims", 0) or [-1, -1]
            table_name = "n-gram" if weight[0] != vocab else "embed_tokens"
            marks.append((e["ts"], f"lookup fwd {table_name} {weight[0]}x{weight[-1]}"))
        elif cat == "cpu_op" and name == "aten::embedding_dense_backward":
            rows = int(arg(e, "Concrete Inputs", 2) or -1)
            table_name = "n-gram" if rows != vocab else "embed_tokens"
            marks.append((e["ts"], f"lookup bwd {table_name} {rows}x{(arg(e, 'Input Dims', 0) or [-1])[-1]}"))
        elif cat == "cpu_op" and name == "record_param_comms" and "to_all" in str(e["args"].get("Collective name", "")):
            caller = next((a["name"] for a in ancestors(e) if "all_to_all" in a["name"]), "?")
            marks.append((e["ts"], f"a2a {e['args'].get('dtype', '?')} {caller.split('(')[0][:48]}"))
        elif cat == "cpu_op" and name in SCALAR_OPS | {"aten::bincount"}:
            marks.append((e["ts"], name))
        elif cat in ("cuda_runtime", "cuda_driver") and (name in SYNC_CALLS or name in COPY_CALLS):
            copy = copies.get(e.get("args", {}).get("correlation"), "")
            if name in COPY_CALLS and "DtoH" not in copy:
                continue
            label = name if name in SYNC_CALLS else f"{name} {copy.removeprefix('Memcpy ')}"
            marks.append((e["ts"], label))
            op = e.get("parent")
            if op is not None and id(op) not in sync_points:
                outer = [a["name"] for a in ancestors(e) if a.get("cat") == "cpu_op" and a["name"].startswith("aten::")]
                sync_points[id(op)] = (op["ts"], f"sync point in {outer[-1] if outer else op['name'][:40]}")
    return marks, list(sync_points.values())


def module_spans(events: list[dict]) -> list[tuple[str, str, float, float]]:
    opened: dict[tuple[str, str, int], float] = {}
    spans = []
    annotations = sorted(
        (e for e in events if e.get("cat") == "user_annotation" and FSDP_SPAN.match(e["name"])), key=lambda e: e["ts"]
    )
    for e in annotations:
        match = FSDP_SPAN.match(e["name"])
        phase, direction, module = match.group(1), match.group(2), match["module"]
        key = (module, direction, e["tid"])
        if phase == "pre":
            opened.setdefault(key, e["ts"])
        elif key in opened:
            spans.append((module, direction, opened.pop(key), e["ts"] + e["dur"]))
    return spans


def span_gpu_ms(events: list[dict], spans, windows) -> dict[int, dict[tuple[str, str], list[float]]]:
    launches = sorted(
        (e["ts"], e["args"]["correlation"])
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    )
    launch_ts = [ts for ts, _ in launches]
    kernel_ms = defaultdict(float)
    for e in events:
        if e.get("cat") == "kernel":
            kernel_ms[e["args"].get("correlation")] += e["dur"] / 1000
    ends = [end for _, _, end in windows]
    per_step: dict[int, dict[tuple[str, str], list[float]]] = defaultdict(lambda: defaultdict(list))
    for module, direction, start, end in spans:
        index = bisect_left(ends, start)
        if index == len(windows):
            continue
        ms = sum(kernel_ms[corr] for _, corr in launches[bisect_left(launch_ts, start) : bisect_right(launch_ts, end)])
        per_step[windows[index][0]][(module, direction)].append(ms)
    return per_step


def module_label(module: str) -> str | None:
    if match := NGRAM.search(module):
        return f"n-gram table {match.group(1) or match.group(2)}"
    if match := LAYER.search(module):
        return f"layer {match.group(1)}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="trace_<rank>.json.gz, a trace dir, or a run_ab.py output dir")
    parser.add_argument("--rank", type=int, default=0, help="rank to read when given a directory")
    parser.add_argument(
        "--vocab-size", type=int, default=DEFAULT_VOCAB, help="embed_tokens rows; other embeddings are n-gram"
    )
    parser.add_argument("--skip", type=int, default=1, help="leading compile steps left out of the steady columns")
    args = parser.parse_args()

    path = resolve_trace(args.trace, args.rank)
    events = load(path)
    link_parents(events)
    windows = step_windows(events)
    ends = [end for _, _, end in windows]
    spans = module_spans(events)
    lookup_spans = sorted(
        (start, end) for module, direction, start, end in spans if NGRAM.search(module) and direction == "forward"
    )
    marks, sync_points = classify(events, args.vocab_size)
    in_lookup = [
        (ts, "sync point inside an n-gram table forward")
        for ts, _ in sync_points
        if (i := bisect_right(lookup_spans, (ts, float("inf"))) - 1) >= 0
        and lookup_spans[i][0] <= ts <= lookup_spans[i][1]
    ]
    counts: dict[int, Counter] = defaultdict(Counter)
    for ts, name in marks + sync_points + in_lookup:
        index = bisect_left(ends, ts)
        if index < len(windows):
            counts[windows[index][0]][name] += 1

    steps = [step for step, _, _ in windows]
    steady = [step for step in steps if step > args.skip]
    print(f"{path}\n{len(steps)} steps, each ending at an Optimizer.step annotation. Steady steps: {steady}.\n")
    print(
        "Host-side calls per step (fewer is better for syncs). Lookups: aten::embedding (forward and AC recompute) and "
        "aten::embedding_dense_backward, by weight rows x width. a2a: record_param_comms all-to-alls by dtype and the "
        "all-to-all op that issued them. Sync point: one CPU op holding a blocking CUDA call or a D2H copy, named by its "
        "outermost aten op. An op nested in an op of the same name (e.g. under AC's dispatch mode) counts once. The "
        "n-gram table forward runs from its FSDP pre-forward hook to its post-forward hook.\n"
    )
    names = sorted(set().union(*counts.values()))
    rows = [[name, *(counts[step][name] for step in steps)] for name in names]
    print(table(["event", *(f"step {s}" for s in steps)], rows))

    per_step = span_gpu_ms(events, spans, windows)
    keys = sorted(
        {key for step in steady for key in per_step[step] if module_label(key[0])},
        key=lambda k: (module_label(k[0]), k[1]),
    )
    if keys:
        print(
            "\nGPU kernel time launched between each module's FSDP pre and post hooks per step, median over steady "
            "steps (ms, lower is better). Spans: hook pairs per step (micro-batches, plus AC recomputes in forward). A "
            "layer's backward includes its AC recompute.\n"
        )
        rows = []
        for module, direction in keys:
            spans = [per_step[step].get((module, direction), []) for step in steady]
            rows.append(
                [
                    module_label(module),
                    direction,
                    statistics.median(len(s) for s in spans),
                    f"{statistics.median(sum(s) for s in spans):.2f}",
                ]
            )
        print(table(["module", "pass", "spans", "GPU ms"], rows, right={2, 3}))


if __name__ == "__main__":
    main()
