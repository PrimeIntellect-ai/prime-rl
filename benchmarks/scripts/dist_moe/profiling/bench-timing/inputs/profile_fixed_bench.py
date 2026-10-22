"""Run the fixed bench_dist_moe.main() with every bench() call under torch.profiler, fwd/bwd wrapped in
record_function("fwd") / ("bwd"), to check that the sleep-led GPU-timed iterations have no host-launch gaps.

Each bench() call is labeled by order: balanced/bf16_ep, balanced/dist_moe, random/bf16_ep, random/dist_moe. With
--warmup W --iters N the k-th fwd annotation is warmup for k < W, GPU-timed for W <= k < W+N, back-to-back after.

Usage (from the prime-rl worktree):
  PROFILE_DIR=... torchrun --nproc_per_node=8 <this file> <bench args> --warmup 3 --iters 3
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.getcwd(), "benchmarks", "dist_moe"))
import bench_dist_moe  # noqa: E402

LABELS = ["balanced_bf16_ep", "balanced_dist_moe", "random_bf16_ep", "random_dist_moe"]
calls = []
original_bench = bench_dist_moe.bench


def bench(fwd, bwd, warmup, iters, lead_cycles):
    label = LABELS[len(calls)]
    calls.append(label)
    out = os.path.join(os.environ["PROFILE_DIR"], label)
    os.makedirs(out, exist_ok=True)

    def annotated_fwd():
        with torch.profiler.record_function("fwd"):
            return fwd()

    def annotated_bwd(y):
        with torch.profiler.record_function("bwd"):
            bwd(y)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        ms = original_bench(annotated_fwd, annotated_bwd, warmup, iters, lead_cycles)
    prof.export_chrome_trace(f"{out}/trace_rank{dist.get_rank()}.json.gz")
    return ms


bench_dist_moe.bench = bench

if __name__ == "__main__":
    bench_dist_moe.main()
