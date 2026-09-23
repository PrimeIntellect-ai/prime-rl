"""Compare the static-`topk` sparse attention kernels with copies whose slot count is dynamic.

`dynamic` makes `topk` itself a dynamic shape; `tiled` instead hands the kernels `Indices` viewed
as `(..., n_tiles, tile)` with only `n_tiles` dynamic. Checks that each produces the same outputs
and gradients as the static kernels, then times each at a range of gather
widths on HCA-like indices: every query's valid slots are a prefix of the slot axis, of a length
drawn uniformly from `[0, width]`, and the rest are `IGNORE_SLOT`.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_bwd import bwd, postprocess, preprocess
from prime_rl.trainer.models.kernels.deepseek_v4.dsv4_sparse_attn_fwd import dsv4_sparse_attn_fwd

sys.path.insert(0, str(Path(__file__).parent))
from dyn_bwd import bwd_dyn  # noqa: E402
from dyn_fwd import dsv4_sparse_attn_fwd_dyn  # noqa: E402
from tiled_bwd import bwd_tiled  # noqa: E402
from tiled_fwd import dsv4_sparse_attn_fwd_tiled  # noqa: E402

HEADS = 64
DIM = 512
SCALE = DIM**-0.5


def make_inputs(seq_len: int, width: int, device: torch.device):
    q = torch.randn(1, seq_len, HEADS, DIM, device=device, dtype=torch.bfloat16)
    kv = torch.randn(1, seq_len, 1, DIM, device=device, dtype=torch.bfloat16)
    indices = torch.randint(0, seq_len, (1, seq_len, 1, width), device=device, dtype=torch.int32)
    n_valid = torch.randint(0, width + 1, (1, seq_len, 1, 1), device=device)
    indices = torch.where(torch.arange(width, device=device) < n_valid, indices, -1).contiguous()
    sinks = torch.randn(HEADS, device=device, dtype=torch.float32)
    grad_out = torch.randn_like(q)
    return q, kv, indices, sinks, grad_out


VARIANTS = ("static", "dynamic", "tiled")
FWD_TILE = 64
BWD_TILE = 32


def kernels(variant: str, width: int):
    """The fwd and bwd kernels of `variant`, each paired with the view of `Indices` it reads."""
    if variant == "static":
        fwd_kernel = dsv4_sparse_attn_fwd(HEADS, DIM, width, 1, SCALE, True)
        bwd_kernel = bwd(HEADS, DIM, width, 1, SCALE, True)
        return (fwd_kernel, lambda idx: idx), (bwd_kernel, lambda idx: idx)
    if variant == "dynamic":
        fwd_kernel = dsv4_sparse_attn_fwd_dyn(HEADS, DIM, 1, SCALE, True)
        bwd_kernel = bwd_dyn(HEADS, DIM, 1, SCALE, True)
        return (fwd_kernel, lambda idx: idx), (bwd_kernel, lambda idx: idx)
    fwd_kernel = dsv4_sparse_attn_fwd_tiled(HEADS, DIM, 1, SCALE, True)
    bwd_kernel = bwd_tiled(HEADS, DIM, 1, SCALE, True)
    return (
        (fwd_kernel, lambda idx: idx.view(*idx.shape[:-1], -1, FWD_TILE)),
        (bwd_kernel, lambda idx: idx.view(*idx.shape[:-1], -1, BWD_TILE)),
    )


def run(variant: str, q, kv, indices, sinks, grad_out):
    (fwd_kernel, fwd_view), (bwd_kernel, bwd_view) = kernels(variant, indices.shape[-1])
    out, lse = fwd_kernel(q, kv, fwd_view(indices), sinks)
    delta = preprocess(HEADS, DIM)(out, grad_out)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd_kernel(q, kv, grad_out, bwd_view(indices), lse, delta, dkv)
    return out, lse, dq, postprocess(DIM, 1)(dkv)


def time_ms(fn, repeats: int) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("seq_len", type=int)
    parser.add_argument("widths", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)

    for variant in ("dynamic", "tiled"):
        start = time.perf_counter()
        kernels(variant, 0)
        print(json.dumps({f"{variant}_compile_s": time.perf_counter() - start}), flush=True)

    for width in args.widths:
        inputs = make_inputs(args.seq_len, width, device)
        q, kv, indices, sinks, grad_out = inputs
        static_results = run("static", *inputs)
        row = dict(seq_len=args.seq_len, width=width)
        for variant in VARIANTS[1:]:
            row[f"{variant}_max_abs_diff"] = {
                name: (a.float() - b.float()).abs().max().item()
                for name, a, b in zip(("out", "lse", "dq", "dkv"), static_results, run(variant, *inputs))
            }
        for variant in VARIANTS:
            (fwd_kernel, fwd_view), (bwd_kernel, bwd_view) = kernels(variant, width)
            fwd_idx, bwd_idx = fwd_view(indices), bwd_view(indices)
            out, lse = fwd_kernel(q, kv, fwd_idx, sinks)
            delta = preprocess(HEADS, DIM)(out, grad_out)
            dkv = torch.zeros_like(kv, dtype=torch.float32)
            row[f"{variant}_fwd_ms"] = time_ms(lambda: fwd_kernel(q, kv, fwd_idx, sinks), args.repeats)
            row[f"{variant}_bwd_ms"] = time_ms(
                lambda: bwd_kernel(q, kv, grad_out, bwd_idx, lse, delta, dkv), args.repeats
            )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
