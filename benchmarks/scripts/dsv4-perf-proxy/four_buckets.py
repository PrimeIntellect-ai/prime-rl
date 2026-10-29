"""Sum a DSv4 trace's kernel time into attention, MoE, mHC and elementwise, plus the lm_head and optimizer.

Reads the profiling skill's `top_kernels.py --by both --csv` output for one rank. GEMM and attention kernels go to the
module that launched them (attention projections have 3-D inputs, the shared expert 2-D inputs, the router 256 outputs),
every other non-NCCL kernel (casts, FP8 quantize, norms, RoPE, permute, SwiGLU, pooling, copies) to elementwise.

Usage: python four_buckets.py <rank.csv> [--scale 0.5]
"""

import argparse
import csv
import re
from collections import Counter

GEMM_KERNEL = r"gemm|nvjet|cutlass|xmma|wgmma"


def bucket(label: str, kernel: str) -> str | None:
    text = f"{label} {kernel}"
    if re.search(r"nccl", kernel, re.IGNORECASE):
        return None
    if re.search(r"_SequenceChunkedLogProbEntropyFn", label):
        return "lm_head + logprob/entropy"
    if re.search(r"grouped_fp8_gemm", label):
        return "MoE" if re.search(GEMM_KERNEL, kernel, re.IGNORECASE) else "elementwise"
    if re.search(r"\[32,(2048|4096),|_foreach_|Optimizer|muon", text, re.IGNORECASE):
        return "optimizer (Muon)"
    if re.search(r"mhc|hpb|sinkhorn|post_bda", text, re.IGNORECASE):
        return "mHC"
    if re.search(r"dsv4_sparse_attn|fp8_indexer", label):
        return "attention"
    if not re.search(GEMM_KERNEL, kernel, re.IGNORECASE):
        return "elementwise"
    if re.search(r"grouped_fp8_gemm|\[\d+,256\]|\[256,", label):
        return "MoE"
    if re.search(r"fp8_blockwise_mm(_backward)? \[1,", label) or re.search(r"aten::bmm \[8,", label):
        return "attention"
    if re.search(r"fp8_blockwise_mm(_backward)? \[\d+,", label):
        return "MoE"
    return "attention"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply totals, e.g. 0.5 for a two-micro-batch window")
    args = parser.parse_args()
    totals: Counter = Counter()
    for row in csv.DictReader(open(args.csv)):
        name = bucket(row["label"], row["kernel_names"])
        if name is not None:
            totals[name] += float(row["total_us"]) / 1000 * args.scale
    for name, ms in totals.most_common():
        print(f"{name}\t{ms:.1f}")


if __name__ == "__main__":
    main()
