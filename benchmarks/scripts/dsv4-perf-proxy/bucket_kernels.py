"""Sum each rank's kernel time into DSv4 op buckets from the profiling skill's `top_kernels.py --by both --csv` output.

Buckets match the call site plus kernel name, first match wins. NCCL time includes waiting for peers.

Usage: python bucket_kernels.py <rank0.csv> [<rank1.csv> ...] [--show-other 15]
"""

import argparse
import csv
import re
from collections import Counter

BUCKETS = [
    ("comm: EP all-to-all (SendRecv)", r"ncclDevKernel_SendRecv"),
    ("comm: all-gather", r"ncclDevKernel_AllGather"),
    ("comm: reduce-scatter", r"ncclDevKernel_ReduceScatter"),
    ("comm: all-reduce", r"ncclDevKernel_AllReduce"),
    ("MoE grouped FP8 GEMM", r"grouped_fp8_gemm"),
    ("lm_head + logprobs/entropy", r"_SequenceChunkedLogProbEntropyFn"),
    ("Muon optimizer", r"muon|newton|zeropower|compiled 2/\d+ > aten::bmm \[32,"),
    ("mHC pre-pass fwd", r"dsv4_mhc_pre \["),
    ("mHC pre-pass bwd", r"dsv4_mhc_pre_backward"),
    ("mHC sinkhorn/post", r"mhc|hpb|sinkhorn|post_bda"),
    ("sparse attention fwd", r"dsv4_sparse_attn \["),
    ("sparse attention bwd", r"dsv4_sparse_attn_backward"),
    ("indexer", r"fp8_indexer"),
    ("dense FP8 GEMM", r"fp8_blockwise_(b)?mm|deep_gemm::sm90_fp8_gemm"),
    ("FP8 quantize", r"per_token_fp8|per_block_fp8|per_channel_fp8|per_token_group_quant"),
    ("FSDP copies", r"fsdp::|chunk_cat|RegisterPostBackward|all_gather_copy|split_with_sizes_copy"),
    ("BF16 GEMM (o_a_proj, router, compressor, ...)", r"nvjet|sm90_xmma|cutlass.*gemm|aten::(mm|bmm|addmm)\b"),
    ("RoPE / norms", r"rope|rmsnorm|RMSNorm|layer_norm"),
    ("MoE routing / permute", r"moe|topk|sort|histc|bincount|index_add|scatter|gather|dispatch|combine"),
]


def bucket_of(text: str) -> str:
    for name, pattern in BUCKETS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return name
    return "elementwise / other"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--show-other", type=int, default=0, help="list the largest unbucketed sites of the first CSV")
    args = parser.parse_args()

    per_rank: list[Counter] = []
    other_rows = []
    for index, path in enumerate(args.csvs):
        totals: Counter = Counter()
        for row in csv.DictReader(open(path)):
            text = f"{row['label']} {row['kernel_names']}"
            name = bucket_of(text)
            ms = float(row["total_us"]) / 1000
            totals[name] += ms
            if index == 0 and name == "elementwise / other":
                other_rows.append((ms, row["label"][:150]))
        per_rank.append(totals)

    names = [name for name, _ in BUCKETS] + ["elementwise / other"]
    names.sort(key=lambda name: -max(totals[name] for totals in per_rank))
    header = " | ".join(f"r{i}" for i in range(len(per_rank)))
    print(f"| bucket (ms per window) | {header} | min | max |")
    print("|---|" + "---|" * (len(per_rank) + 2))
    for name in names:
        values = [totals[name] for totals in per_rank]
        print(f"| {name} | " + " | ".join(f"{v:.0f}" for v in values) + f" | {min(values):.0f} | {max(values):.0f} |")
    sums = [sum(totals.values()) for totals in per_rank]
    print("| sum of kernel durations | " + " | ".join(f"{v:.0f}" for v in sums) + f" | {min(sums):.0f} | {max(sums):.0f} |")
    if args.show_other:
        print()
        for ms, label in sorted(other_rows, reverse=True)[: args.show_other]:
            print(f"{ms:8.1f}  {label}")


if __name__ == "__main__":
    main()
