"""Kernel-level benchmark of the DeepGEMM FP8 grouped GEMM against torch._grouped_mm."""

import argparse
import importlib.metadata
import math
import os
import socket
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
from triton.testing import do_bench

import prime_rl.trainer.models.layers.fp8_grouped_gemm  # noqa: F401
from prime_rl.trainer.distributed.token_dispatcher import permute_for_grouped_gemm
from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_block_cast_to_fp8_triton,
    grouped_per_channel_cast_to_fp8_sm90_kmajor_triton,
    grouped_per_token_cast_to_fp8_triton,
    ue8m0_for_device,
    unpack_rows_triton,
)

NUM_EXPERTS = 32
SHAPES = {"gate_up": (4096, 4096), "down": (2048, 4096)}
ROWS_PER_EXPERT = [128, 256, 512, 1024, 1536, 2048, 4096, 6144, 8192]
RAGGED_MAX_OVER_MEAN = 6.0
GATE_REL_ERR = 0.1
FP8_OP = torch.ops.prime_rl.grouped_fp8_gemm.default
FP8_BWD_OP = torch.ops.prime_rl.grouped_fp8_gemm_backward.default


@dataclass
class Case:
    shape: str
    k: int
    n: int
    rows_per_expert: int
    distribution: str
    x: torch.Tensor
    param: torch.Tensor
    offs: torch.Tensor
    dy: torch.Tensor
    real_rows: int
    grouped_rows: int

    @property
    def weight(self) -> torch.Tensor:
        return self.param.transpose(-2, -1)

    @property
    def name(self) -> str:
        return f"{self.shape} rpe={self.rows_per_expert} {self.distribution}"


def ragged_counts(total: int, num_experts: int, max_over_mean: float) -> torch.Tensor:
    lo, hi = 0.0, 8.0
    for _ in range(60):
        exponent = (lo + hi) / 2
        weights = torch.arange(1, num_experts + 1, dtype=torch.float64).pow(-exponent)
        if weights.max() / weights.mean() < max_over_mean:
            lo = exponent
        else:
            hi = exponent
    counts = torch.floor(weights / weights.sum() * total).long()
    counts[0] += total - counts.sum()
    return counts


def make_case(shape: str, rows_per_expert: int, distribution: str, alignment: int) -> Case:
    k, n = SHAPES[shape]
    total = rows_per_expert * NUM_EXPERTS
    if distribution == "balanced":
        counts = torch.full((NUM_EXPERTS,), rows_per_expert, dtype=torch.long)
    else:
        counts = ragged_counts(total, NUM_EXPERTS, RAGGED_MAX_OVER_MEAN)
    counts = counts.cuda()
    tokens = torch.randn(total, k, device="cuda", dtype=torch.bfloat16)
    x, padded_counts, _ = permute_for_grouped_gemm(
        tokens, counts, experts_per_rank=NUM_EXPERTS, num_ranks=1, alignment=alignment
    )
    del tokens
    offs = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
    param = (torch.randn(NUM_EXPERTS, n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)).requires_grad_()
    dy = torch.randn(x.size(0), n, device="cuda", dtype=torch.bfloat16)
    return Case(
        shape=shape,
        k=k,
        n=n,
        rows_per_expert=rows_per_expert,
        distribution=distribution,
        x=x.contiguous().requires_grad_(),
        param=param,
        offs=offs,
        dy=dy,
        real_rows=total,
        grouped_rows=int(offs[-1].item()),
    )


def bench(fn, args, grad_to_none=None) -> tuple[float, float]:
    times = [
        do_bench(fn, warmup=args.warmup, rep=args.rep, grad_to_none=grad_to_none, return_mode="min")
        for _ in range(args.reps)
    ]
    best = min(times)
    return best, (max(times) - best) / best


def rel_err(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual.float() - reference.float()).norm() / reference.float().norm()).item()


def correctness(case: Case) -> dict[str, float]:
    rows = case.grouped_rows
    x, w, offs, dy = case.x.detach(), case.weight.detach(), case.offs, case.dy
    ref_out = torch._grouped_mm(x, w, offs=offs)
    ref_dx = torch._grouped_mm(dy, w.transpose(-2, -1), offs=offs)
    ref_dw = torch._grouped_mm(x.t(), dy, offs=offs)
    out = FP8_OP(x, w, offs)
    dx, dw = FP8_BWD_OP(dy, x, w, offs, True, True, not w.is_contiguous())
    return {
        "fwd": rel_err(out[:rows], ref_out[:rows]),
        "dgrad": rel_err(dx[:rows], ref_dx[:rows]),
        "wgrad": rel_err(dw, ref_dw),
    }


def fwd_bwd(case: Case, op):
    def run():
        case.x.grad = None
        case.param.grad = None
        out = op(case.x, case.weight, offs=case.offs)
        out.backward(case.dy)

    return run


def fp8_autograd(x, weight, *, offs):
    return FP8_OP(x, weight, offs)


def launch_wall(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    walls = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        walls.append((time.perf_counter() - start) * 1e3)
        torch.cuda.synchronize()
    return statistics.median(walls)


def count_syncs(fn) -> int:
    fn()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return sum("synchroniz" in str(w.message) for w in caught)


def copy_ceiling(num_bytes: int, args) -> float:
    half = max(num_bytes // 2, 1)
    src = torch.empty(half, device="cuda", dtype=torch.uint8)
    dst = torch.empty_like(src)
    best, _ = bench(lambda: dst.copy_(src), args)
    return best


def measure_ops(case: Case, args) -> list[tuple[str, float, float]]:
    x, w, offs, dy = case.x.detach(), case.weight.detach(), case.offs, case.dy
    xt = x.t()
    results = []

    def add(label, fn, grad_to_none=None):
        best, spread = bench(fn, args, grad_to_none)
        results.append((label, best, spread))

    bench(lambda: torch._grouped_mm(x, w, offs=offs), args)
    add("bf16 fwd", lambda: torch._grouped_mm(x, w, offs=offs))
    add("bf16 dgrad", lambda: torch._grouped_mm(dy, w.transpose(-2, -1), offs=offs))
    add("bf16 wgrad", lambda: torch._grouped_mm(xt, dy, offs=offs))
    add("fp8 fwd", lambda: FP8_OP(x, w, offs))
    add("fp8 dgrad", lambda: FP8_BWD_OP(dy, x, w, offs, True, False, not w.is_contiguous()))
    add("fp8 wgrad", lambda: FP8_BWD_OP(dy, x, w, offs, False, True, not w.is_contiguous()))
    leaves = [case.x, case.param]
    add("bf16 fwd+bwd autograd", fwd_bwd(case, torch._grouped_mm), leaves)
    add("fp8 fwd+bwd autograd", fwd_bwd(case, fp8_autograd), leaves)
    return results


def measure_casts(case: Case, args) -> list[tuple[str, float, float, int, float]]:
    x, w, offs, dy = case.x.detach(), case.weight.detach(), case.offs, case.dy
    ue8m0 = ue8m0_for_device(x.device)
    (
        total_m,
        padded_total_m,
        _,
        block_to_group,
        ks_tensor,
        starts,
        actual_ms,
        block_starts,
    ) = build_grouped_layout(offs, total_m=x.size(0))
    out_padded = torch.empty((padded_total_m, case.n), device="cuda", dtype=torch.bfloat16)
    grad_weight_fp32 = torch.zeros(w.shape, device="cuda", dtype=torch.float32)
    grouped, k, n, g = case.grouped_rows, case.k, case.n, NUM_EXPERTS
    scale_bytes = 4 * padded_total_m * triton.cdiv(k, GROUP_ALIGNMENT)
    specs = [
        ("layout build", lambda: build_grouped_layout(offs, total_m=x.size(0)), 4 * (2 * padded_total_m)),
        (
            "act cast per-token x",
            lambda: grouped_per_token_cast_to_fp8_triton(
                x, padded_total_m, block_to_group, starts, actual_ms, block_starts, ue8m0, GROUP_ALIGNMENT
            ),
            2 * grouped * k + padded_total_m * k + scale_bytes,
        ),
        (
            "weight cast fwd",
            lambda: grouped_per_block_cast_to_fp8_triton(w.transpose(1, 2), ue8m0, GROUP_ALIGNMENT),
            3 * g * k * n,
        ),
        (
            "weight cast dgrad",
            lambda: grouped_per_block_cast_to_fp8_triton(w, ue8m0, GROUP_ALIGNMENT),
            3 * g * k * n,
        ),
        (
            "kmajor cast x (wgrad)",
            lambda: grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
                x,
                padded_total_m,
                block_to_group,
                starts,
                actual_ms,
                ks_tensor,
                block_starts,
                False,
                GROUP_ALIGNMENT,
            ),
            3 * grouped * k,
        ),
        (
            "kmajor cast dy (wgrad)",
            lambda: grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(
                dy,
                padded_total_m,
                block_to_group,
                starts,
                actual_ms,
                ks_tensor,
                block_starts,
                False,
                GROUP_ALIGNMENT,
            ),
            3 * grouped * n,
        ),
        (
            "unpack rows",
            lambda: unpack_rows_triton(out_padded, total_m, block_to_group, starts, actual_ms, block_starts),
            4 * grouped * n,
        ),
        ("grad_weight zeros", lambda: torch.zeros(w.shape, device="cuda", dtype=torch.float32), 4 * g * k * n),
        ("grad_weight fp32->bf16", lambda: grad_weight_fp32.to(torch.bfloat16), 6 * g * k * n),
    ]
    results = []
    for label, fn, num_bytes in specs:
        best, spread = bench(fn, args)
        ceiling = copy_ceiling(num_bytes, args)
        results.append((label, best, spread, num_bytes, ceiling))
    return results


def device_header(args, arch: str) -> list[str]:
    props = torch.cuda.get_device_properties(0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    smi = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            visible,
            "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.memory,clocks.sm",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    clocks = smi.stdout.strip() if smi.returncode == 0 else "unavailable"
    return [
        f"# {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  host={socket.gethostname()}  arch={arch}",
        f"# label={args.label}  alignment={args.alignment}",
        f"# gpu={props.name}  sms={props.multi_processor_count}  mem={props.total_memory // 2**20} MiB",
        f"# nvidia-smi (name, driver, max sm clk, max mem clk, cur sm clk): {clocks}",
        f"# torch={torch.__version__}  triton={triton.__version__}  deep_gemm={importlib.metadata.version('deep-gemm')}",
        f"# num_experts={NUM_EXPERTS}  weight=(G, K, N)  shapes={SHAPES}  ragged max/mean={RAGGED_MAX_OVER_MEAN}",
        f"# do_bench warmup={args.warmup}ms rep={args.rep}ms return_mode=min, fastest of {args.reps} repeats",
        "",
    ]


def break_even(points: list[tuple[int, float]]) -> str:
    for (r0, q0), (r1, q1) in zip(points, points[1:]):
        if q0 > 1.0 >= q1:
            t = (q0 - 1.0) / (q0 - q1)
            return f"~{round(math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0))))}"
    if points and points[0][1] <= 1.0:
        return f"<= {points[0][0]}"
    return f"> {points[-1][0]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="tag in the results filename, e.g. base or item1")
    parser.add_argument("--alignment", type=int, default=8, help="dispatcher token_group_alignment")
    parser.add_argument("--rows-per-expert", type=int, nargs="+", default=ROWS_PER_EXPERT)
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=list(SHAPES))
    parser.add_argument("--distributions", nargs="+", default=["balanced", "ragged"], choices=["balanced", "ragged"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=40)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--no-casts", action="store_true", help="skip the per-cast isolation arm")
    args = parser.parse_args()

    torch.manual_seed(0)
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm{major}{minor}"
    gpu = torch.cuda.get_device_name().replace(" ", "_")
    out_path = Path(__file__).parent / f"results-{args.label}-{socket.gethostname().split('.')[0]}-{gpu}-{arch}.txt"
    lines = device_header(args, arch)
    sweep_start = time.perf_counter()

    configs = [
        (shape, rpe, dist) for shape in args.shapes for dist in args.distributions for rpe in args.rows_per_expert
    ]

    lines.append("## correctness gate (relative L2 error vs torch._grouped_mm, grouped rows only)")
    failed = False
    for shape, rpe, dist in configs:
        case = make_case(shape, rpe, dist, args.alignment)
        errs = correctness(case)
        ok = all(v < GATE_REL_ERR for v in errs.values())
        failed |= not ok
        lines.append(
            f"  {'ok  ' if ok else 'FAIL'} {case.name:32s} "
            + "  ".join(f"{key}={val:.4f}" for key, val in errs.items())
        )
        del case
        torch.cuda.empty_cache()
    if failed:
        lines.append(f"CORRECTNESS FAILED (bound {GATE_REL_ERR}), refusing to report timings")
        out_path.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        sys.exit(1)
    lines.append("")
    print("\n".join(lines), flush=True)

    noise_case = make_case(args.shapes[0], args.rows_per_expert[0], args.distributions[0], args.alignment)
    noise_x, noise_w, noise_offs = noise_case.x.detach(), noise_case.weight.detach(), noise_case.offs
    noise_start, _ = bench(lambda: torch._grouped_mm(noise_x, noise_w, offs=noise_offs), args)
    del noise_case, noise_x, noise_w, noise_offs

    ratios: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for shape, rpe, dist in configs:
        case = make_case(shape, rpe, dist, args.alignment)
        block = [
            f"## {case.name}  x.size(0)={case.x.size(0)} grouped={case.grouped_rows} real={case.real_rows}"
            f"  tail={case.x.size(0) - case.grouped_rows}"
        ]
        ops = measure_ops(case, args)
        timings = {label: best for label, best, _ in ops}
        for label, best, spread in ops:
            block.append(f"  {label:28s} {best:9.4f} ms  spread {100 * spread:4.1f}%")
        bf16_total = timings["bf16 fwd"] + timings["bf16 dgrad"] + timings["bf16 wgrad"]
        fp8_total = timings["fp8 fwd"] + timings["fp8 dgrad"] + timings["fp8 wgrad"]
        ratio = timings["fp8 fwd+bwd autograd"] / timings["bf16 fwd+bwd autograd"]
        block.append(
            f"  sum fwd+dgrad+wgrad: bf16 {bf16_total:.4f} ms  fp8 {fp8_total:.4f} ms"
            f"  fp8/bf16 {fp8_total / bf16_total:.3f}   autograd fp8/bf16 {ratio:.3f}"
        )
        ratios.setdefault((shape, dist), []).append((rpe, ratio))

        for label, op in (("bf16", torch._grouped_mm), ("fp8", fp8_autograd)):
            fn = fwd_bwd(case, op)
            wall = launch_wall(fn)
            gpu_ms = timings[f"{label} fwd+bwd autograd"]
            syncs = count_syncs(fn)
            block.append(
                f"  host launch wall {label:4s} fwd+bwd {wall:8.4f} ms  (gpu {gpu_ms:.4f} ms,"
                f" wall/gpu {wall / gpu_ms:.2f})  device-to-host syncs {syncs}"
            )

        if not args.no_casts:
            block.append(f"  {'cast / helper':28s} {'ms':>9s}  {'GB/s':>7s}  {'copy ms':>8s}  {'x off copy':>10s}")
            for label, best, spread, num_bytes, ceiling in measure_casts(case, args):
                block.append(
                    f"  {label:28s} {best:9.4f}  {num_bytes / best / 1e6:7.0f}  {ceiling:8.4f}"
                    f"  {best / ceiling:9.2f}x  spread {100 * spread:4.1f}%"
                )
        block.append("")
        print("\n".join(block), flush=True)
        lines.extend(block)
        case.x.grad = None
        case.param.grad = None
        del case
        torch.cuda.empty_cache()

    noise_case = make_case(args.shapes[0], args.rows_per_expert[0], args.distributions[0], args.alignment)
    noise_x, noise_w, noise_offs = noise_case.x.detach(), noise_case.weight.detach(), noise_case.offs
    noise_end, _ = bench(lambda: torch._grouped_mm(noise_x, noise_w, offs=noise_offs), args)
    summary = [
        "## break-even rows/expert (autograd fwd+bwd, fp8/bf16 crosses 1.0)",
        *(
            f"  {shape:8s} {dist:9s} {break_even(points):>8s}   "
            + "  ".join(f"{rpe}:{ratio:.3f}" for rpe, ratio in points)
            for (shape, dist), points in ratios.items()
        ),
        "",
        f"## noise control: bf16 fwd {noise_case.name} start {noise_start:.4f} ms end {noise_end:.4f} ms"
        f" drift {100 * (noise_end - noise_start) / noise_start:+.1f}%",
        f"## sweep wall time {time.perf_counter() - sweep_start:.0f} s",
    ]
    lines.extend(summary)
    print("\n".join(summary))
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
