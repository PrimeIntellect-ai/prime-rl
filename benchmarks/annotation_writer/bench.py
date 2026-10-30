"""Mocked trainer-bound step: GPU work, AnnotationWriter.flush(), then a grad-norm-style all_reduce.

Run from a prime-rl checkout (main for the baseline arm, fix/trainer-async-annotation-writer for the
async arm); the arm is detected from whether AnnotationWriter has close(). See HANDOFF.md next to this file:

    uv run torchrun --nproc-per-node 8 <path>/bench.py --out-dir /local/nvme/annot-bench/<arm>

Records are synthetic but built with the real make_update and written by the real FileMonitor, so the
on-disk bytes are what production writes. Results land in <out-dir>/results.json and stdout.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from prime_rl import monitors
from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.file.traces.update import make_update
from prime_rl.trainer.rl import annotations
from prime_rl.trainer.rl.annotations import AnnotationWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--tokens-per-step",
        type=float,
        default=5e8,
        help="Across all ranks; a trainer-bound production run trained 4e8 to 5e8.",
    )
    parser.add_argument("--steps", type=int, default=6, help="Step 0 is warmup and excluded from the summary.")
    parser.add_argument(
        "--gpu-seconds",
        type=float,
        default=90.0,
        help="Per-step GPU work. Must exceed the background write time, as production's ~900 s step does.",
    )
    parser.add_argument("--min-seq-len", type=int, default=50_000)
    parser.add_argument("--max-seq-len", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_streams(rng: np.random.Generator, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Logprob and entropy streams as _masked_floats leaves them: 4 decimals, NaN outside the loss mask.

    The mask is a prompt prefix followed by alternating tool-output and assistant spans."""
    mask = np.zeros(length, dtype=bool)
    position = int(length * rng.uniform(0.1, 0.3))
    trained = True
    while position < length:
        span = int(rng.integers(2_000, 8_000))
        mask[position : position + span] = trained
        position += span
        trained = not trained
    logprobs = -rng.exponential(0.5, length).astype(np.float32).round(4)
    entropies = rng.exponential(0.8, length).astype(np.float32).round(4)
    nan = np.float32(np.nan)
    return np.where(mask, logprobs, nan), np.where(mask, entropies, nan)


def make_rank_streams(args: argparse.Namespace, rank: int, world_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(args.seed * 1_000 + rank)
    budget = int(args.tokens_per_step) // world_size
    streams = []
    while budget > 0:
        length = min(budget, int(rng.integers(args.min_seq_len, args.max_seq_len + 1)))
        streams.append(make_streams(rng, length))
        budget -= length
    return streams


def calibrate_matmuls(gpu_seconds: float) -> tuple[torch.Tensor, int]:
    """Many small launches, so a GIL-holding background thread on rank 0 shows up as slower GPU work."""
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    iters = 200
    start = time.perf_counter()
    for _ in range(iters):
        a @ a
    torch.cuda.synchronize()
    per_iter = (time.perf_counter() - start) / iters
    count = torch.tensor([gpu_seconds / per_iter], device="cuda")
    dist.all_reduce(count, op=dist.ReduceOp.MIN)
    return a, int(count.item())


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    is_async = hasattr(AnnotationWriter, "close")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(monitors.setup(file=FileMonitorConfig(), output_dir=args.out_dir, producer="trainer"))
    write_seconds: list[float] = []
    if rank == 0 and is_async:
        log_annotations = annotations._log_annotations

        def timed_log_annotations(records):
            start = time.perf_counter()
            log_annotations(records)
            write_seconds.append(time.perf_counter() - start)

        annotations._log_annotations = timed_log_annotations

    writer = AnnotationWriter(SimpleNamespace(cp_enabled=False), SimpleNamespace(world_size=world_size, rank=rank), 4)
    streams = make_rank_streams(args, rank, world_size)
    matrix, matmul_count = calibrate_matmuls(args.gpu_seconds)
    scalar = torch.ones((), device="cuda")

    steps = []
    for step in range(args.steps):
        writer._pending = [
            make_update(
                f"r{rank}-s{step}-{index}", branches={0: {"trainer_logprobs": logprobs, "entropies": entropies}}
            )
            for index, (logprobs, entropies) in enumerate(streams)
        ]
        dist.barrier()
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        for _ in range(matmul_count):
            matrix @ matrix
        torch.cuda.synchronize()
        flush_start = time.perf_counter()
        writer.flush()
        flush_end = time.perf_counter()
        dist.all_reduce(scalar)
        torch.cuda.synchronize()
        step_end = time.perf_counter()
        steps.append(
            {
                "step": step,
                "gpu_work_s": flush_start - step_start,
                "flush_return_s": flush_end - flush_start,
                "flush_to_allreduce_done_s": step_end - flush_start,
                "step_s": step_end - step_start,
            }
        )

    if is_async:
        writer.close()
    asyncio.run(monitors.finalize())

    gathered: list | None = [None] * world_size if rank == 0 else None
    dist.gather_object(steps, gathered, dst=0)
    if rank == 0:
        report(args, is_async, gathered, write_seconds)
    dist.destroy_process_group()


def report(args: argparse.Namespace, is_async: bool, gathered: list, write_seconds: list[float]) -> None:
    rank0, others = gathered[0], gathered[1:]
    rows = []
    for step in range(1, args.steps):
        rows.append(
            {
                "step": step,
                "rank0_gpu_work_s": rank0[step]["gpu_work_s"],
                "others_gpu_work_s": float(np.mean([steps[step]["gpu_work_s"] for steps in others])),
                "rank0_flush_return_s": rank0[step]["flush_return_s"],
                "others_stall_s": float(np.mean([steps[step]["flush_to_allreduce_done_s"] for steps in others])),
                "step_s": rank0[step]["step_s"],
                "background_write_s": write_seconds[step] if is_async else None,
            }
        )
    result = {"arm": "async" if is_async else "sync", "args": {k: str(v) for k, v in vars(args).items()}, "steps": rows}
    (args.out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"arm={result['arm']} tokens_per_step={args.tokens_per_step:.3g} gpu_seconds={args.gpu_seconds}")
    header = ["step", "rank0_gpu_work_s", "others_gpu_work_s", "rank0_flush_return_s", "others_stall_s", "step_s"]
    header.append("background_write_s")
    print(" ".join(f"{name:>22}" for name in header))
    for row in rows:
        print(" ".join(f"{row[name]:>22.2f}" if row[name] is not None else f"{'-':>22}" for name in header))


if __name__ == "__main__":
    main()
