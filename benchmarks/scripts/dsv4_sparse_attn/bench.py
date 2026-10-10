"""Steady-state benchmark of the DSv4 sparse attention backends on the corpus, gated on correctness.

usage (repo root, on an otherwise idle GPU):
  uv run --no-sync python benchmarks/scripts/dsv4_sparse_attn/bench.py --out RESULTS.json [--label NAME]
      [--corpus DIR] [--items GLOB ...] [--backends NAME ...] [--rounds K] [--iters N] [--warmup N]
      [--profile-iters N]
  uv run --no-sync python benchmarks/scripts/dsv4_sparse_attn/bench.py --compare A.json [B.json ...]

Per item, every backend's outputs are first compared against `tilelang` (out, lse, dq, dkv, dsink), and on
items of total length at most 4096 also against the float32 dense reference; a backend that fails is
reported and not timed. Each passing backend is then timed on the forward and on forward+backward
(`torch.autograd.grad` with a fixed `dO`):

- op-boundary time: CUDA events around one call that starts with the GPU idle and the L2 flushed, so host
  launch overhead is included. K rounds alternate the backend order (ABBA), each round takes the median
  of N calls, and the median, p20 and p80 across rounds are reported in microseconds.
- GPU kernel time: the summed duration of the kernels, memsets and memcpys each call launches, from a
  torch.profiler trace, so op-boundary minus kernel time is host overhead and launch gaps.
- peak memory above the pre-call allocation.

Results go to a JSON with provenance (GPU, driver, clocks, power limit, host, git SHA, corpus hash), and the
markdown tables are printed. `--compare` prints the tables for several result files and refuses files built
on different corpora. Items default to the grid, excluding the dynamic stream's items.
"""

import argparse
import bisect
import json
import statistics
import tempfile
from pathlib import Path

import torch
from common import (
    BASELINE_BACKEND,
    DEFAULT_CORPUS_DIR,
    DENSE_REFERENCE_MAX_LEN,
    PEAK_SOURCE,
    SM_SCALE,
    backend_names,
    executed_slots,
    forward_backward,
    load_backend,
    load_indices,
    load_manifest,
    make_inputs,
    peak_dense_tflops,
    provenance,
    select_items,
    slot_coverage,
    useful_flops,
)
from torch.profiler import ProfilerActivity, profile, record_function

from tests.unit.train.models.test_deepseek_v4_kernels import (
    DKV_RTOL,
    DQ_RTOL,
    DSINK_RTOL,
    LSE_RTOL,
    LSE_RTOL_BY_BACKEND,
    OUT_RTOL,
    _dense_reference,
    _reference_lse,
)

L2_FLUSH_BYTES = 256 * 2**20
PROFILE_RANGE = "dsv4_sparse_attn_bench_call"
GPU_ACTIVITY_CATEGORIES = ("kernel", "gpu_memset", "gpu_memcpy")
MODES = ("fwd", "fwd_bwd")


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Largest absolute deviation over the reference's largest magnitude, the tests' `_assert_relative` measure."""
    actual, reference = actual.float(), reference.float()
    return float((actual - reference).abs().max() / reference.abs().max())


def tolerances(name: str) -> dict[str, float]:
    lse_rtol = LSE_RTOL_BY_BACKEND.get(name, LSE_RTOL)
    return {"out": OUT_RTOL, "lse": lse_rtol, "dq": DQ_RTOL, "dkv": DKV_RTOL, "dsink": DSINK_RTOL}


def backend_outputs(backend, inputs: dict, indices: torch.Tensor) -> dict[str, torch.Tensor | None]:
    q, kv, sinks = inputs["q"], inputs["kv"], inputs["sinks"]
    with torch.no_grad():
        out, lse = backend.fwd(q, kv, indices, sinks, SM_SCALE)
    outputs = {"out": out, "lse": lse}
    if not backend.FORWARD_ONLY:
        dq, dkv, dsink = forward_backward(backend, q, kv, indices, sinks, SM_SCALE, inputs["grad_out"])
        outputs.update(dq=dq, dkv=dkv, dsink=dsink)
    return outputs


def dense_outputs(inputs: dict, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    """The float32 dense reference: output, log2 sink-inclusive LSE and the three gradients."""
    q, kv, sinks = (inputs[name].detach().float().requires_grad_(True) for name in ("q", "kv", "sinks"))
    out = _dense_reference(q, kv, indices, sinks, SM_SCALE)
    dq, dkv, dsink = torch.autograd.grad(out, (q, kv, sinks), inputs["grad_out"].float())
    lse = _reference_lse(inputs["q"].detach(), inputs["kv"].detach(), indices, inputs["sinks"].detach())
    return {"out": out.detach(), "lse": lse, "dq": dq, "dkv": dkv, "dsink": dsink}


def gate(name: str, outputs: dict, reference: dict) -> dict[str, dict]:
    checks = {}
    for tensor, rtol in tolerances(name).items():
        if outputs.get(tensor) is None or reference.get(tensor) is None:
            continue
        error = relative_error(outputs[tensor], reference[tensor])
        checks[tensor] = {"relative_error": error, "rtol": rtol, "ok": error <= rtol}
    return checks


def summarize(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"median": values[0], "p20": values[0], "p80": values[0]}
    p20, _, _, p80 = statistics.quantiles(values, n=5, method="inclusive")
    return {"median": statistics.median(values), "p20": p20, "p80": p80}


def time_op_boundary(fns: dict, rounds: int, iters: int, flush: torch.Tensor) -> dict[str, list[float]]:
    """Per backend, the median microseconds of each round; rounds alternate the backend order."""
    names = list(fns)
    round_medians = {name: [] for name in names}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for round_index in range(rounds):
        for name in names if round_index % 2 == 0 else reversed(names):
            samples = []
            for _ in range(iters):
                flush.zero_()
                torch.cuda.synchronize()
                start.record()
                fns[name]()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1e3)
            round_medians[name].append(statistics.median(samples))
    return round_medians


def time_gpu_activity(fn, iters: int, flush: torch.Tensor) -> dict:
    """GPU busy time per call, from the kernels, memsets and memcpys launched inside each profiled call."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            flush.zero_()
            torch.cuda.synchronize()
            with record_function(PROFILE_RANGE):
                fn()
            torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as trace_file:
        prof.export_chrome_trace(trace_file.name)
        events = json.loads(Path(trace_file.name).read_text())["traceEvents"]

    calls = sorted(
        (event["ts"], event["ts"] + event["dur"])
        for event in events
        if event.get("name") == PROFILE_RANGE and event.get("cat") == "user_annotation"
    )
    assert len(calls) == iters, f"found {len(calls)} profiled calls, expected {iters}"
    call_starts = [call_start for call_start, _ in calls]
    launch_ts = {
        event["args"]["correlation"]: event["ts"]
        for event in events
        if event.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in event.get("args", {})
    }
    busy_us, launches = [0.0] * iters, [0] * iters
    for event in events:
        if event.get("cat") not in GPU_ACTIVITY_CATEGORIES:
            continue
        launched = launch_ts.get(event.get("args", {}).get("correlation"))
        if launched is None:
            continue
        call = bisect.bisect_right(call_starts, launched) - 1
        if call >= 0 and launched <= calls[call][1]:
            busy_us[call] += event["dur"]
            launches[call] += 1
    return {"us": summarize(busy_us), "launches_per_call": statistics.median(launches)}


def peak_mib(fn) -> float:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - baseline) / 2**20


def callables(backend, inputs: dict, indices: torch.Tensor) -> dict:
    q, kv, sinks, grad_out = inputs["q"], inputs["kv"], inputs["sinks"], inputs["grad_out"]

    def fwd():
        with torch.no_grad():
            backend.fwd(q, kv, indices, sinks, SM_SCALE)

    def fwd_bwd():
        forward_backward(backend, q, kv, indices, sinks, SM_SCALE, grad_out)

    return {"fwd": fwd} if backend.FORWARD_ONLY else {"fwd": fwd, "fwd_bwd": fwd_bwd}


def flops_record(backend, coverage: dict, n_queries: int, n_slots: int, sum_valid: int) -> dict:
    useful = useful_flops(sum_valid)
    tiles = backend.EXECUTED_SLOT_TILE
    if tiles is None:
        multiple = getattr(backend, "SLOT_MULTIPLE", 1)
        padded = n_queries * (-(-n_slots // multiple) * multiple)
        slots = {"fwd": padded, "bwd": padded}
    else:
        slots = {phase: executed_slots(coverage, tile) for phase, tile in tiles.items()}
    executed = {"fwd": useful["fwd"] * slots["fwd"] // sum_valid, "bwd": useful["bwd"] * slots["bwd"] // sum_valid}
    return {
        "executed_basis": "padded slots (proxy)" if tiles is None else f"tiles {tiles}",
        "executed": executed,
        "executed_over_useful": {phase: slots[phase] / sum_valid for phase in ("fwd", "bwd")},
    }


def benchmark_item(item, corpus_dir: Path, backends: dict, args, flush: torch.Tensor) -> dict:
    indices = load_indices(corpus_dir, item)
    coverage = slot_coverage(indices)
    sum_valid = int(coverage["n_valid"].sum())
    inputs = make_inputs(item, requires_grad=True)
    record = {
        "id": item.id,
        "total_len": item.total_len,
        "layer_type": item.layer_type,
        "cp": item.cp,
        "cp_rank": item.cp_rank,
        "n_queries": item.n_queries,
        "n_positions": item.n_positions,
        "n_slots": item.n_slots,
        "max_doc_len": max(item.doc_lens),
        "sum_valid": sum_valid,
        "useful_flops": useful_flops(sum_valid),
        "backends": {},
    }

    reference = backend_outputs(backends[BASELINE_BACKEND], inputs, indices)
    dense = dense_outputs(inputs, indices) if item.total_len <= DENSE_REFERENCE_MAX_LEN else None
    passing = {}
    for name, backend in backends.items():
        entry = {"label": backend.LABEL, "forward_only": backend.FORWARD_ONLY}
        try:
            outputs = reference if name == BASELINE_BACKEND else backend_outputs(backend, inputs, indices)
        except Exception as error:
            entry.update(passed=False, error=repr(error))
            record["backends"][name] = entry
            print(f"{item.id} {name}: raised {error!r}, not timed", flush=True)
            continue
        checks = {"vs_tilelang": gate(name, outputs, reference)}
        if dense is not None:
            checks["vs_dense_fp32"] = gate(name, outputs, dense)
        failed = [
            f"{basis}.{tensor}"
            for basis, by_tensor in checks.items()
            for tensor, check in by_tensor.items()
            if not check["ok"]
        ]
        entry.update(correctness=checks, passed=not failed, failed=failed)
        if failed:
            print(f"{item.id} {name}: failed {failed}, not timed", flush=True)
        else:
            passing[name] = callables(backend, inputs, indices)
            entry.update(flops_record(backend, coverage, item.n_queries, item.n_slots, sum_valid))
        record["backends"][name] = entry
        del outputs
    del reference, dense
    torch.cuda.empty_cache()

    for mode in MODES:
        fns = {name: fns_by_mode[mode] for name, fns_by_mode in passing.items() if mode in fns_by_mode}
        for fn in fns.values():
            for _ in range(args.warmup):
                fn()
        rounds = time_op_boundary(fns, args.rounds, args.iters, flush)
        for name, fn in fns.items():
            record["backends"][name].setdefault("timing", {})[mode] = {
                "op_us": summarize(rounds[name]),
                "op_us_rounds": rounds[name],
                "gpu": time_gpu_activity(fn, args.profile_iters, flush),
                "peak_mib": peak_mib(fn),
            }
    return record


def fmt_us(value: float) -> str:
    return f"{value:.1f}" if value < 1000 else f"{value:.0f}"


def baseline_time(run: dict, item_id: str, mode: str, key: str) -> float | None:
    entry = run["items"].get(item_id, {}).get("backends", {}).get(BASELINE_BACKEND, {})
    timing = entry.get("timing", {}).get(mode)
    return None if timing is None else (timing["op_us"]["median"] if key == "op" else timing["gpu"]["us"]["median"])


def print_tables(runs: list[dict]) -> None:
    hashes = {run["provenance"]["corpus_hash"] for run in runs}
    if len(hashes) > 1:
        raise SystemExit(f"refusing to compare results built on different corpora: {sorted(hashes)}")
    for run in runs:
        prov = run["provenance"]
        query = prov["gpu"]["query"]
        print(
            f"- `{run['label']}`: {query['name']}, driver {query['driver_version']}, SM clock {query['clocks.sm']} "
            f"(max {query['clocks.max.sm']}), power limit {query['power.limit']}, host {prov['hostname']}, "
            f"git {prov['git']['sha'][:9]}{' (dirty)' if prov['git']['dirty'] else ''}"
        )
    print(f"- corpus hash {hashes.pop()[:16]}; synthetic corpus: random-weight CSA picks are near-uniform, while a")
    print("  trained indexer favors recent and neighboring entries, so CSA gather locality here is pessimistic.\n")

    item_ids = list(dict.fromkeys(item_id for run in runs for item_id in run["items"]))
    multiple = len(runs) > 1

    def rows():
        for item_id in item_ids:
            for run in runs:
                item = run["items"].get(item_id)
                if item is None:
                    continue
                for name, entry in item["backends"].items():
                    yield item_id, run, item, (f"{name}@{run['label']}" if multiple else name), entry

    print("Op-boundary time per call in µs (lower is better): median over rounds, p20-p80 across rounds.")
    print("`/TL` is this time divided by tilelang's in the same run.\n")
    print("| item | backend | fwd µs | fwd p20-p80 | fwd /TL | f+b µs | f+b p20-p80 | f+b /TL |")
    print("|---|---|---|---|---|---|---|---|")
    for item_id, run, _item, label, entry in rows():
        cells = []
        for mode in MODES:
            timing = entry.get("timing", {}).get(mode)
            if timing is None:
                cells += ["-", "-", "-"]
                continue
            op = timing["op_us"]
            base = baseline_time(run, item_id, mode, "op")
            ratio = f"{op['median'] / base:.2f}" if base else "-"
            cells += [fmt_us(op["median"]), f"{fmt_us(op['p20'])}-{fmt_us(op['p80'])}", ratio]
        print(f"| {item_id} | {label} | " + " | ".join(cells) + " |")

    print("\nGPU busy time per call in µs from profiler traces (lower is better); `host` is op-boundary minus GPU")
    print("busy time (launch overhead and gaps); `/TL` divides GPU busy time by tilelang's; `peak MiB` is the")
    print("allocation above the inputs during one call, forward+backward where the backend has it, else forward.\n")
    print("| item | backend | fwd gpu µs | fwd host | fwd /TL | f+b gpu µs | f+b host | f+b /TL | peak MiB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for item_id, run, _item, label, entry in rows():
        cells, peak = [], "-"
        for mode in MODES:
            timing = entry.get("timing", {}).get(mode)
            if timing is None:
                cells += ["-", "-", "-"]
                continue
            gpu = timing["gpu"]["us"]["median"]
            base = baseline_time(run, item_id, mode, "gpu")
            cells += [fmt_us(gpu), fmt_us(timing["op_us"]["median"] - gpu), f"{gpu / base:.2f}" if base else "-"]
            peak = f"{timing['peak_mib']:.0f}"
        print(f"| {item_id} | {label} | " + " | ".join(cells) + f" | {peak} |")

    peak_tflops = runs[0]["peak_dense_bf16_tflops"]
    print("\nUseful FLOPs count valid slots only (fwd 4HD, bwd 10HD per slot); `exec/useful` counts the slots each")
    print("backend's tiles touch, or every padded slot for an arm without tile information. TFLOP/s divide")
    print("useful FLOPs by op-boundary time (higher is better);")
    print(f"`% peak` is f+b against {peak_tflops} dense BF16 TFLOP/s ({runs[0]['peak_source']}).\n")
    print("| item | backend | f+b GFLOP | exec/useful fwd | exec/useful bwd | fwd TFLOP/s | f+b TFLOP/s | % peak |")
    print("|---|---|---|---|---|---|---|---|")
    for item_id, _run, item, label, entry in rows():
        if "executed_over_useful" not in entry:
            continue
        ratios = entry["executed_over_useful"]
        rates = []
        for mode in MODES:
            timing = entry.get("timing", {}).get(mode)
            rates.append(None if timing is None else item["useful_flops"][mode] / timing["op_us"]["median"] / 1e6)
        percent = f"{100 * rates[1] / peak_tflops:.1f}" if rates[1] is not None and peak_tflops else "-"
        rate_cells = ["-" if rate is None else f"{rate:.1f}" for rate in rates]
        print(
            f"| {item_id} | {label} | {item['useful_flops']['fwd_bwd'] / 1e9:.1f} | {ratios['fwd']:.2f} | "
            f"{ratios['bwd']:.2f} | {rate_cells[0]} | {rate_cells[1]} | {percent} |"
        )

    failures = [(item_id, label, entry) for item_id, _run, _item, label, entry in rows() if not entry["passed"]]
    print(f"\nCorrectness failures (excluded from timing): {len(failures)}")
    for item_id, label, entry in failures:
        print(f"- {item_id} {label}: {entry.get('error') or ', '.join(entry['failed'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs="+", type=Path, help="print tables for these result files and exit")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--label", default=None, help="run name in tables; defaults to the short git SHA")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--items", nargs="+", default=None, help="item id globs; default: every non-stream item")
    parser.add_argument("--backends", nargs="+", default=None, help="default: every available backend")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=5)
    args = parser.parse_args()

    if args.compare:
        print_tables([json.loads(path.read_text()) for path in args.compare])
        return
    if args.out is None:
        parser.error("--out is required unless --compare is given")

    manifest = load_manifest(args.corpus)
    items = select_items(manifest, args.items)
    if args.items is None:
        items = [item for item in items if not item.row.startswith("stream")]
    names = args.backends or backend_names()
    if BASELINE_BACKEND not in names:
        names = [BASELINE_BACKEND, *names]
    backends, skipped = {}, {}
    for name in names:
        backend = load_backend(name)
        reason = backend.unavailable_reason()
        if reason is None:
            backends[name] = backend
        else:
            skipped[name] = reason
            print(f"skipping {name}: {reason}")

    prov = provenance(manifest["corpus_hash"])
    result = {
        "label": args.label or prov["git"]["sha"][:9],
        "provenance": prov,
        "settings": {key: value for key, value in vars(args).items() if key not in ("compare", "out", "corpus")},
        "corpus": str(args.corpus),
        "backends": {name: backend.LABEL for name, backend in backends.items()},
        "skipped_backends": skipped,
        "peak_dense_bf16_tflops": peak_dense_tflops(),
        "peak_source": PEAK_SOURCE,
        "items": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.int8, device="cuda")
    for position, item in enumerate(items):
        print(f"[{position + 1}/{len(items)}] {item.id}", flush=True)
        result["items"][item.id] = benchmark_item(item, args.corpus, backends, args, flush)
        args.out.write_text(json.dumps(result, indent=1))
    print_tables([result])


if __name__ == "__main__":
    main()
