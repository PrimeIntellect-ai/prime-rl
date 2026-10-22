"""Run bench_dist_moe.main() with its bench() wrapped: per-rank event times printed on every rank, and an optional
torch.profiler capture of a few timed iterations.

Each bench() call is labeled by order: balanced/bf16_ep, balanced/dist_moe, random/bf16_ep, random/dist_moe.
Every iteration is wrapped in record_function("fwd") / ("bwd") and stepped as a ProfilerStep.

Env:
  PROFILE_DIR   if set, write traces to $PROFILE_DIR/<routing>_<path>/trace_rank<r>.json.gz
  PROFILE_STACK if "1", record Python stacks

Usage (from the prime-rl worktree):
  PROFILE_DIR=... torchrun --nproc_per_node=8 <this file> <bench args>
"""

import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.getcwd(), "benchmarks", "dist_moe"))
import bench_dist_moe  # noqa: E402

LABELS = ["balanced_bf16_ep", "balanced_dist_moe", "random_bf16_ep", "random_dist_moe"]
calls = []


def bench(fwd, bwd, warmup, iters):
    label = LABELS[len(calls)]
    calls.append(label)
    rank = dist.get_rank()
    for _ in range(warmup):
        bwd(fwd())
    torch.cuda.synchronize()
    dist.barrier()
    events = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(iters)]
    for start, mid, end in events:
        start.record()
        y = fwd()
        mid.record()
        bwd(y)
        end.record()
    torch.cuda.synchronize()
    fwd_ms = [s.elapsed_time(m) for s, m, _ in events]
    bwd_ms = [m.elapsed_time(e) for _, m, e in events]
    tot_ms = [s.elapsed_time(e) for s, _, e in events]
    print(
        f"EVENTS {label} rank={rank} fwd mean={statistics.fmean(fwd_ms):.3f} med={statistics.median(fwd_ms):.3f} "
        f"min={min(fwd_ms):.3f} max={max(fwd_ms):.3f} | bwd mean={statistics.fmean(bwd_ms):.3f} "
        f"med={statistics.median(bwd_ms):.3f} min={min(bwd_ms):.3f} max={max(bwd_ms):.3f} | "
        f"tot mean={statistics.fmean(tot_ms):.3f} med={statistics.median(tot_ms):.3f} "
        f"min={min(tot_ms):.3f} max={max(tot_ms):.3f} | first5 fwd={[round(t, 3) for t in fwd_ms[:5]]}",
        flush=True,
    )

    out = os.environ.get("PROFILE_DIR")
    if out:
        out = os.path.join(out, label)
        os.makedirs(out, exist_ok=True)
        dist.barrier()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            record_shapes=True,
            with_stack=os.environ.get("PROFILE_STACK") == "1",
            on_trace_ready=lambda p: p.export_chrome_trace(f"{out}/trace_rank{rank}.json.gz"),
        ) as prof:
            for _ in range(6):
                with torch.profiler.record_function("fwd"):
                    y = fwd()
                with torch.profiler.record_function("bwd"):
                    bwd(y)
                prof.step()
        torch.cuda.synchronize()
    return statistics.fmean(fwd_ms), statistics.fmean(bwd_ms)


bench_dist_moe.bench = bench

if __name__ == "__main__":
    bench_dist_moe.main()
