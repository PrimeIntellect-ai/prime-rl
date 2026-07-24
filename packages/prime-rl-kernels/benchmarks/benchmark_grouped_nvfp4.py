from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F
from prime_rl_kernels import grouped_gemm
from triton.testing import do_bench


def _benchmark(operation: Callable[[], object], *, warmup: int, repetitions: int) -> float:
    operation()
    torch.cuda.synchronize()
    return float(do_bench(operation, warmup=warmup, rep=repetitions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark grouped NVFP4 GEMM on Blackwell")
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--tokens-per-group", type=int, default=256)
    parser.add_argument("--in-features", type=int, default=2048)
    parser.add_argument("--out-features", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(0)
    rows = args.groups * args.tokens_per_group
    matrix = torch.randn(rows, args.in_features, device="cuda", dtype=torch.bfloat16) * 0.1
    # Prime's parameters are physically [G, N, K]. The grouped-matmul API sees
    # the cheap transposed [G, K, N] view.
    weight_storage = (
        torch.randn(
            args.groups,
            args.out_features,
            args.in_features,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    weight = weight_storage.transpose(-2, -1)
    offsets = torch.arange(1, args.groups + 1, device="cuda", dtype=torch.int32) * args.tokens_per_group

    operations = {
        "bf16": lambda: F.grouped_mm(
            matrix,
            weight,
            offs=offsets,
            out_dtype=torch.bfloat16,
        ),
        "nvfp4": lambda: grouped_gemm(matrix, weight, offsets),
    }

    flop = 2 * rows * args.in_features * args.out_features
    print(
        f"shape: G={args.groups}, M={rows}, K={args.in_features}, "
        f"N={args.out_features}, tokens/group={args.tokens_per_group}"
    )
    for name, operation in operations.items():
        milliseconds = _benchmark(
            operation,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        tflops = flop / (milliseconds * 1e9)
        print(f"{name:>21}: {milliseconds:.4f} ms, {tflops:.1f} TFLOP/s")


if __name__ == "__main__":
    main()
