"""Time the fused mHC pre-pass at the proxy's per-rank shape under each fp32 dot precision, and vLLM's DeepGEMM projection.

Usage: uv run --no-sync python bench_mhc_pre.py [--tokens 31078] [--iters 20]
"""

import argparse

import torch

import prime_rl.trainer.models.kernels.deepseek_v4.dsv4_mhc_pre as mhc_pre

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=31078)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.tokens, HC, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fn = (torch.randn(N, HC * D, device="cuda", dtype=torch.float32) * 0.02).requires_grad_()
    scale = torch.ones(N, device="cuda", dtype=torch.float32)
    base = torch.zeros(N, device="cuda", dtype=torch.float32)
    x_bytes = x.numel() * x.element_size()
    collapsed_bytes = args.tokens * D * 2
    copy_ms = time_ms(lambda: x.detach().clone(), args.iters)
    print(f"tokens {args.tokens}: x {x_bytes / 1e9:.2f} GB, copy of x {copy_ms:.3f} ms ({2 * x_bytes / copy_ms / 1e6:.0f} GB/s)")
    print(f"forward roofline (read x, write collapsed) at copy bandwidth: {(x_bytes + collapsed_bytes) / (2 * x_bytes) * copy_ms:.3f} ms")

    import deep_gemm

    x_flat = x.detach().view(args.tokens, HC * D)
    projection = torch.empty(1, args.tokens, N, device="cuda", dtype=torch.float32)
    sqrsum = torch.empty(1, args.tokens, device="cuda", dtype=torch.float32)
    prenorm_ms = time_ms(lambda: deep_gemm.tf32_hc_prenorm_gemm(x_flat, fn.detach(), projection, sqrsum, 1), args.iters)
    exact = x_flat.double() @ fn.detach().double().T
    prenorm_err = (projection[0].double() - exact).abs().max().item()
    ieee_err = (x_flat.float() @ fn.detach().T - exact).abs().max().item()
    print(
        f"deep_gemm.tf32_hc_prenorm_gemm (projection + sum of squares only): {prenorm_ms:.3f} ms, "
        f"max |projection - fp64| {prenorm_err:.2e} (fp32 matmul at highest: {ieee_err:.2e})"
    )

    reference = None
    for precision in ["ieee", "tf32x3", "tf32"]:
        mhc_pre._precision = lambda precision=precision: precision

        def forward():
            return mhc_pre.fused_mhc_pre(x, fn, scale, base, 1e-6, 1e-6)

        def forward_backward():
            collapsed, logits = forward()
            (collapsed.float().sum() + logits.sum()).backward()

        forward_ms = time_ms(forward, args.iters)
        forward_backward_ms = time_ms(forward_backward, args.iters)
        with torch.no_grad():
            collapsed, logits = forward()
        if reference is None:
            reference = (collapsed.float(), logits)
        logits_err = (logits - reference[1]).abs().max().item()
        collapsed_err = (collapsed.float() - reference[0]).abs().max().item()
        print(
            f"{precision:>6}: forward {forward_ms:.3f} ms, forward+backward {forward_backward_ms:.3f} ms, "
            f"max |logits - ieee| {logits_err:.2e}, max |collapsed - ieee| {collapsed_err:.2e}"
        )


if __name__ == "__main__":
    main()
