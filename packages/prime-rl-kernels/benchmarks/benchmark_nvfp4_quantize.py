from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
from prime_rl_kernels.nvfp4.quantize import quantize_activations, quantize_weights


def _benchmark(operation: Callable[[], object], warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def _report(name: str, milliseconds: float, transferred_bytes: float) -> float:
    terabytes_per_second = transferred_bytes / milliseconds / 1e9
    print(f"{name:>12}: {milliseconds:.4f} ms, {terabytes_per_second:.2f} TB/s")
    return terabytes_per_second


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SM100 NVFP4 quantization")
    parser.add_argument("--kind", choices=("activation", "weight"), default="weight")
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--rows", type=int, default=43_648)
    parser.add_argument("--in-features", type=int, default=6144)
    parser.add_argument("--out-features", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--minimum-bandwidth", type=float)
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.kind == "activation":
        source = torch.randn(
            args.rows,
            args.in_features,
            device="cuda",
            dtype=torch.bfloat16,
        )
        offsets = torch.tensor([args.rows], device="cuda", dtype=torch.int32)
        quantize = lambda: quantize_activations(source, offsets)
        quantized = quantize()
        elements = source.numel()
        metadata_bytes = args.rows * 4
    else:
        storage = torch.randn(
            args.groups,
            args.out_features,
            args.in_features,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = storage.transpose(-2, -1)
        quantize = lambda: quantize_weights(weight)
        quantized = quantize()
        elements = weight.numel()
        metadata_bytes = args.groups * 4

    # Quantization reads BF16 twice (global amax and block quantization), then
    # writes packed FP4 and one E4M3 scale per 16 values.
    quantize_bytes = elements * (4 + 0.5 + 1 / 16) + metadata_bytes
    # Dequantization reads packed FP4 and scales and writes BF16.
    dequantize_bytes = elements * (0.5 + 1 / 16 + 2) + metadata_bytes
    quantize_ms = _benchmark(quantize, args.warmup, args.repetitions)
    dequantize_ms = _benchmark(
        quantized.dequantize,
        args.warmup,
        args.repetitions,
    )

    print(f"{args.kind}: G={args.groups}, M={args.rows}, K={args.in_features}, N={args.out_features}")
    quantize_bandwidth = _report("quantize", quantize_ms, quantize_bytes)
    _report("dequantize", dequantize_ms, dequantize_bytes)
    if args.minimum_bandwidth is not None and quantize_bandwidth < args.minimum_bandwidth:
        raise RuntimeError(
            f"quantization bandwidth {quantize_bandwidth:.2f} TB/s is below {args.minimum_bandwidth:.2f} TB/s"
        )


if __name__ == "__main__":
    main()
