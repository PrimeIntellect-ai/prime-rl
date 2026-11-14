"""Single-GPU expert MLP (DSv4-Flash routed experts) with the grouped GEMMs at hand: torch._grouped_mm (CUTLASS),
a cuBLAS matmul per expert (the original LeastLoadedEP), DeepGEMM's contiguous grouped GEMM, and a padded bmm.

    python benchmarks/llep/bench_grouped_gemm.py --routes 49152 --experts 32
"""

import argparse

import torch
import torch.nn.functional as F


def swiglu(h, limit=10.0):
    gate, up = h.chunk(2, dim=-1)
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def timed(fn, iters=10):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, default=49152)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--skew", type=float, default=0.0, help="Zipf exponent of the routes per expert")
    args = parser.parse_args()
    E, H, I = args.experts, args.hidden, args.intermediate
    weights = (torch.arange(E, dtype=torch.float64) + 1).pow(-args.skew)
    counts = (weights / weights.sum() * args.routes).floor().long()
    counts[0] += args.routes - counts.sum()
    gate_up = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.02
    down = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) * 0.02
    x = torch.randn(args.routes, H, device="cuda", dtype=torch.bfloat16)
    flops = 2 * args.routes * 3 * H * I
    results = {}

    offsets = torch.cumsum(counts, 0).to(torch.int32).cuda()

    def grouped_mm():
        h = torch._grouped_mm(x, gate_up.transpose(-2, -1), offsets)
        return torch._grouped_mm(swiglu(h), down.transpose(-2, -1), offsets)

    results["torch._grouped_mm (CUTLASS)"] = timed(grouped_mm)

    bounds = [0, *torch.cumsum(counts, 0).tolist()]

    def per_expert_mm():
        out = torch.empty_like(x)
        for e in range(E):
            s, t = bounds[e], bounds[e + 1]
            if s < t:
                out[s:t] = swiglu(x[s:t] @ gate_up[e].t()) @ down[e].t()
        return out

    results["cuBLAS mm per expert (LeastLoadedEP)"] = timed(per_expert_mm)

    try:
        import deep_gemm

        align = deep_gemm.get_mk_alignment_for_contiguous_layout()
        padded = [(int(c) + align - 1) // align * align for c in counts]
        layout = torch.cat(
            [torch.where(torch.arange(p) < c, e, -1).int() for e, (p, c) in enumerate(zip(padded, counts))]
        ).cuda()
        xp = torch.zeros(sum(padded), H, device="cuda", dtype=torch.bfloat16)
        h1 = torch.empty(sum(padded), 2 * I, device="cuda", dtype=torch.bfloat16)
        h2 = torch.empty(sum(padded), H, device="cuda", dtype=torch.bfloat16)

        def deepgemm():
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(xp, gate_up, h1, layout)
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(swiglu(h1).contiguous(), down, h2, layout)
            return h2

        results["DeepGEMM m_grouped_bf16 contiguous"] = timed(deepgemm)
    except Exception as error:  # the build may lack the kernel
        print("DeepGEMM grouped GEMM unavailable:", repr(error)[:200])

    cap = int(counts.max())
    xb = torch.randn(E, cap, H, device="cuda", dtype=torch.bfloat16)

    def padded_bmm():
        return torch.bmm(swiglu(torch.bmm(xb, gate_up.transpose(-2, -1))), down.transpose(-2, -1))

    results[f"torch.bmm padded to the busiest expert ({cap} rows)"] = timed(padded_bmm)

    print(f"routes={args.routes} experts={E} skew={args.skew} busiest={cap}")
    for name, ms in results.items():
        print(f"  {name:48s} {ms:7.3f} ms  {flops / ms / 1e9:6.0f} TFLOP/s (useful)")


if __name__ == "__main__":
    main()
