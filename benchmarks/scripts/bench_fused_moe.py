#!/usr/bin/env python3
"""
Time the routed-expert compute three ways: the fused MoE kernel (bf16 and mxfp8) against
the grouped-mm path a `model.moe_fused_kernel=false` run takes.

Forward and backward are timed separately, because only the forward is a kernel — the
backward is prime-rl's own closed-form bf16 gradients or, for mxfp8, autograd through the
grouped-mm reference. A speedup on the forward can be spent again on the backward.

    uv run benchmarks/scripts/bench_fused_moe.py
    uv run benchmarks/scripts/bench_fused_moe.py --tokens 16384 --num-experts 128
"""

from __future__ import annotations

import argparse

import torch

from prime_rl.trainer.models.layers.moe import _FusedMoE, _load_fused_moe_kernel, _run_experts_fused_reference

MODES = ("grouped_mm", "fused_bf16", "fused_mxfp8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=8192, help="routed tokens (batch * seq)")
    parser.add_argument("--dim", type=int, default=2048, help="model dim; must be a multiple of 256")
    parser.add_argument("--hidden-dim", type=int, default=768, help="expert intermediate size; multiple of 128")
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    randn = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    x = (randn(args.tokens, args.dim) * 0.2).requires_grad_(True)
    w1 = (randn(args.num_experts, args.hidden_dim, args.dim) * 0.05).requires_grad_(True)
    w3 = (randn(args.num_experts, args.hidden_dim, args.dim) * 0.05).requires_grad_(True)
    w2 = (randn(args.num_experts, args.dim, args.hidden_dim) * 0.05).requires_grad_(True)
    logits = torch.randn(args.tokens, args.num_experts, device="cuda", dtype=torch.float32)
    top_scores, selected = torch.softmax(logits, dim=-1).topk(args.top_k, dim=-1)
    return x, w1, w2, w3, selected, top_scores.requires_grad_(True)


def run_forward(mode: str, inputs: tuple[torch.Tensor, ...], num_experts: int) -> torch.Tensor:
    x, w1, w2, w3, selected, top_scores = inputs
    if mode == "grouped_mm":
        return _run_experts_fused_reference(x, w1, w2, w3, selected, top_scores, num_experts)
    return _FusedMoE.apply(x, w1, w2, w3, selected, top_scores, num_experts, mode == "fused_mxfp8")


def time_mode(mode: str, inputs: tuple[torch.Tensor, ...], args: argparse.Namespace) -> tuple[float, float, float]:
    """Median forward ms, median backward ms, peak MiB."""
    forward_ms, backward_ms = [], []
    grad_out = torch.randn(args.tokens, args.dim, device="cuda", dtype=torch.bfloat16)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()

    for i in range(args.warmup + args.iters):
        for tensor in inputs[:4] + (inputs[5],):
            tensor.grad = None
        torch.cuda.synchronize()

        start.record()
        out = run_forward(mode, inputs, args.num_experts)
        end.record()
        torch.cuda.synchronize()
        if i >= args.warmup:
            forward_ms.append(start.elapsed_time(end))

        start.record()
        out.backward(grad_out)
        end.record()
        torch.cuda.synchronize()
        if i >= args.warmup:
            backward_ms.append(start.elapsed_time(end))

    median = lambda values: sorted(values)[len(values) // 2]
    return median(forward_ms), median(backward_ms), torch.cuda.max_memory_allocated() / 2**20


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    kernel = _load_fused_moe_kernel()
    for mxfp8 in (False, True):
        reason = kernel.unsupported_shape_reason(args.dim, args.hidden_dim, mxfp8=mxfp8)
        if reason is not None:
            raise SystemExit(f"shape unsupported by the fused kernel: {reason}")

    inputs = make_inputs(args)
    print(
        f"tokens={args.tokens} dim={args.dim} hidden_dim={args.hidden_dim} "
        f"experts={args.num_experts} top_k={args.top_k} "
        f"({args.iters} iters, {args.warmup} warmup, median)\n"
    )
    print(f"{'mode':<12} {'fwd ms':>9} {'bwd ms':>9} {'total ms':>9} {'speedup':>9} {'peak MiB':>10}")

    baseline = None
    for mode in MODES:
        forward, backward, peak = time_mode(mode, inputs, args)
        total = forward + backward
        baseline = baseline or total
        print(f"{mode:<12} {forward:>9.3f} {backward:>9.3f} {total:>9.3f} {baseline / total:>8.2f}x {peak:>10.0f}")


if __name__ == "__main__":
    main()
