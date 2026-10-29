"""Time the mHC input side before #4216 (compiled fp32 norm, F.linear and collapse) against #4216's fused op, both compiled.

Usage: uv run --no-sync python bench_mhc_pre_history.py [--tokens 31078] [--iters 20]
"""

import argparse

import torch
import torch.nn.functional as F

from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_mhc_pre import fused_mhc_pre

HC, D, N = 4, 4096, 24


def time_ms(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def before_4216(x, fn, scale, base):
    flat = x.flatten(start_dim=-2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    mixes = F.linear(flat, fn)
    logits = mixes * scale + base
    pre = torch.sigmoid(logits[..., :HC]) + 1e-6
    return (pre.unsqueeze(-1) * x).sum(-2).to(x.dtype), logits


def with_4216(x, fn, scale, base):
    return fused_mhc_pre(x, fn, scale, base, 1e-6, 1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=31078)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(1, args.tokens, HC, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fn = (torch.randn(N, HC * D, device="cuda") * 0.02).requires_grad_()
    scale = torch.ones(N, device="cuda", requires_grad=True)
    base = torch.zeros(N, device="cuda", requires_grad=True)

    for precision in ["highest", "high"]:
        torch.set_float32_matmul_precision(precision)
        torch._dynamo.reset()
        for name, impl in [("before #4216", before_4216), ("#4216", with_4216)]:
            compiled = torch.compile(impl, fullgraph=True)

            def forward():
                return compiled(x, fn, scale, base)

            def forward_backward():
                collapsed, logits = forward()
                (collapsed.float().sum() + logits.sum()).backward()

            forward_ms = time_ms(forward, args.iters)
            forward_backward_ms = time_ms(forward_backward, args.iters)
            print(f"{precision:>7} {name:>13}: forward {forward_ms:.3f} ms, forward+backward {forward_backward_ms:.3f} ms")


if __name__ == "__main__":
    main()
