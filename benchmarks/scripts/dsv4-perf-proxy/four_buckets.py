"""Sum a DSv4 trace's kernel time into attention, MoE, mHC and elementwise, plus the lm_head and optimizer.

Reads the profiling skill's `top_kernels.py --by both --csv` output for one rank. GEMM and attention kernels go to the
module that launched them (attention projections have 3-D inputs, the shared expert 2-D inputs, the router
`--router-experts` outputs), every other non-NCCL kernel (casts, FP8 quantize, norms, RoPE, permute, SwiGLU, pooling,
copies) to elementwise.

The defaults match DSv4 Flash. For another MoE model, pass its Muon expert-matrix shapes, router width and (if it has
dense MLP layers) their intermediate width, e.g. GLM-4.5-Air: `--muon-shape '\\[16,(1408|2816|4096),' --router-experts 128
--dense-mlp-width 10944`.

Usage: python four_buckets.py <rank.csv> [--scale 0.5] [--muon-shape REGEX] [--router-experts N] [--dense-mlp-width N]
"""

import argparse
import csv
import re
from collections import Counter

GEMM_KERNEL = r"gemm|nvjet|cutlass|xmma|wgmma"


def bucket(label: str, kernel: str, muon_shape: str, router_experts: int, dense_mlp_width: int | None) -> str | None:
    text = f"{label} {kernel}"
    if re.search(r"nccl", kernel, re.IGNORECASE):
        return None
    if re.search(r"_SequenceChunkedLogProbEntropyFn", label):
        return "lm_head + logprob/entropy"
    if re.search(r"grouped_fp8_gemm", label):
        return "MoE" if re.search(GEMM_KERNEL, kernel, re.IGNORECASE) else "elementwise"
    if re.search(muon_shape, text) or re.search(r"_foreach_|Optimizer|muon", text, re.IGNORECASE):
        return "optimizer (Muon)"
    if re.search(r"mhc|hpb|sinkhorn|post_bda", text, re.IGNORECASE):
        return "mHC"
    if re.search(r"dsv4_sparse_attn|fp8_indexer", label):
        return "attention"
    if not re.search(GEMM_KERNEL, kernel, re.IGNORECASE):
        return "elementwise"
    if re.search(rf"grouped_fp8_gemm|\[\d+,{router_experts}\]|\[{router_experts},", label):
        return "MoE"
    if dense_mlp_width is not None and re.search(rf",{dense_mlp_width}\]|\[{dense_mlp_width},", label):
        return "dense MLP"
    if re.search(r"fp8_blockwise_mm(_backward)? \[1,", label) or re.search(r"aten::bmm \[8,", label):
        return "attention"
    if re.search(r"fp8_blockwise_mm(_backward)? \[\d+,", label):
        return "MoE"
    return "attention"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply totals, e.g. 0.5 for a two-micro-batch window")
    parser.add_argument("--muon-shape", default=r"\[32,(2048|4096),", help="label regex of Muon expert matrices")
    parser.add_argument("--router-experts", type=int, default=256, help="router output width")
    parser.add_argument("--dense-mlp-width", type=int, default=None, help="intermediate width of dense MLP layers")
    args = parser.parse_args()
    totals: Counter = Counter()
    for row in csv.DictReader(open(args.csv)):
        name = bucket(row["label"], row["kernel_names"], args.muon_shape, args.router_experts, args.dense_mlp_width)
        if name is not None:
            totals[name] += float(row["total_us"]) / 1000 * args.scale
    for name, ms in totals.most_common():
        print(f"{name}\t{ms:.1f}")


if __name__ == "__main__":
    main()
