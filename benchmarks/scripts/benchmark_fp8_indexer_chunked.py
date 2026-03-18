#!/usr/bin/env python3
"""Benchmark full vs chunked FP8 indexer memory and speed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch

from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer, fp8_indexer_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=str, default="2048,4096,8192,16384,32768,65536")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/fp8_indexer_chunked.json"))
    return parser.parse_args()


def make_inputs(seq_len: int, heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
    q = torch.randn(seq_len, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(seq_len, head_dim, device=device, dtype=dtype)
    w = torch.randn(seq_len, heads, device=device, dtype=dtype)
    ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device)
    return q, k, w, ks, ke


def measure_impl(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    warmup: int,
    iters: int,
) -> dict:
    device = q.device
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    try:
        for _ in range(warmup):
            _ = fn(q, k, w, ks, ke, topk)
        torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        timings_ms: list[float] = []
        output = None
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn(q, k, w, ks, ke, topk)
            end.record()
            torch.cuda.synchronize(device)
            timings_ms.append(start.elapsed_time(end))

        assert output is not None
        peak_mem = torch.cuda.max_memory_allocated(device)
        return {
            "success": True,
            "latency_ms_mean": sum(timings_ms) / len(timings_ms),
            "latency_ms_p50": sorted(timings_ms)[len(timings_ms) // 2],
            "latency_ms_min": min(timings_ms),
            "latency_ms_max": max(timings_ms),
            "peak_memory_gib": peak_mem / (1024**3),
            "checksum": int(output[:, 0].sum().item()),
            "output": output,
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        return {
            "success": False,
            "error": str(exc).split("\n")[0],
        }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    device = torch.device("cuda")

    results: list[dict] = []

    for seq_len in seq_lens:
        q, k, w, ks, ke = make_inputs(seq_len, args.heads, args.head_dim, dtype, device)

        full_result = measure_impl(fp8_indexer_full, q, k, w, ks, ke, args.topk, args.warmup, args.iters)
        chunked_fn = lambda q_, k_, w_, ks_, ke_, topk_: fp8_indexer(  # noqa: E731
            q_, k_, w_, ks_, ke_, topk_, chunk_size=args.chunk_size
        )
        chunked_result = measure_impl(chunked_fn, q, k, w, ks, ke, args.topk, args.warmup, args.iters)

        full_output = full_result.pop("output", None)
        chunked_output = chunked_result.pop("output", None)

        same_output = None
        mismatch_count = None
        if full_output is not None and chunked_output is not None:
            same_output = torch.equal(full_output, chunked_output)
            mismatch_count = int((full_output != chunked_output).sum().item())

        record = {
            "seq_len": seq_len,
            "full": full_result,
            "chunked": chunked_result,
            "same_output": same_output,
            "mismatch_count": mismatch_count,
        }
        results.append(record)

        full_mem = full_result.get("peak_memory_gib")
        chunked_mem = chunked_result.get("peak_memory_gib")
        full_lat = full_result.get("latency_ms_mean")
        chunked_lat = chunked_result.get("latency_ms_mean")
        print(
            f"seq={seq_len:6d} | "
            f"full(mem={full_mem}, ms={full_lat}, ok={full_result['success']}) | "
            f"chunked(mem={chunked_mem}, ms={chunked_lat}, ok={chunked_result['success']}) | "
            f"same={same_output}"
        )

    payload = {
        "timestamp": int(time.time()),
        "device": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "heads": args.heads,
        "head_dim": args.head_dim,
        "topk": args.topk,
        "chunk_size": args.chunk_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
