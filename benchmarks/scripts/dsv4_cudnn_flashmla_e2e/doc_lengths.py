"""Rendered token lengths of a random sample of SFT documents per dataset subset, as the DeepSeek V4 SFT run sees them.

Each sampled row goes through `SFTDataset._process` with the `deepseek-v4` renderer (thinking on unless `--no-thinking`), so lengths include the
chat scaffolding the trainer packs. Per subset it prints the length percentiles and, for packing into rows of
`--seq-len` tokens, the share of packed tokens that come from documents at least 16k and 32k long (a document
longer than a row is counted at the row length, since packing truncates it).

usage (on a compute node; CPU only):
  env -u HF_HOME HF_HUB_CACHE=/home/huggingface/hub uv run --no-sync python \
      benchmarks/scripts/dsv4_cudnn_flashmla_e2e/doc_lengths.py SNAPSHOT_DATA_DIR SUBSET [SUBSET ...] \
      [--rows 200] [--seq-len 65536] [--no-thinking]
"""

import argparse
import random
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset
from renderers.configs import DeepSeekV4RendererConfig

from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.trainer.sft.data import RendererResolver, SFTDataset

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
LONG_THRESHOLDS = (16384, 32768)


def sample_rows(subset_dir: Path, rows: int, rng: random.Random) -> list[dict]:
    files = sorted(subset_dir.glob("*.parquet"))
    counts = [pq.ParquetFile(file).metadata.num_rows for file in files]
    picks = sorted(rng.sample(range(sum(counts)), min(rows, sum(counts))))
    sampled, offset = [], 0
    for file, count in zip(files, counts):
        local = [pick - offset for pick in picks if offset <= pick < offset + count]
        if local:
            sampled.extend(pq.read_table(file).take(local).to_pylist())
        offset += count
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("subsets", nargs="+")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--no-thinking", action="store_true", help="render with enable_thinking = false")
    args = parser.parse_args()

    tokenizer = setup_tokenizer(TokenizerConfig(name=MODEL))
    resolver = RendererResolver(tokenizer, DeepSeekV4RendererConfig(enable_thinking=not args.no_thinking))
    rng = random.Random(0)
    header = "| subset | rows | p10 | p50 | p90 | max | tokens from docs >= 16k | >= 32k |"
    print(header)
    print("|---" * (header.count("|") - 1) + "|")
    for subset in args.subsets:
        rows = sample_rows(args.data_dir / subset, args.rows, rng)
        dataset = SFTDataset(Dataset.from_list(rows), resolver, seq_len=args.seq_len)
        lengths = sorted(len(sample["input_ids"]) for row in rows if (sample := dataset._process(row)) is not None)
        packed = [min(length, args.seq_len) for length in lengths]
        shares = [sum(n for n in packed if n >= threshold) / sum(packed) for threshold in LONG_THRESHOLDS]

        def percentile(fraction: float) -> int:
            return lengths[min(len(lengths) - 1, int(fraction * len(lengths)))]

        print(
            f"| {subset} | {len(lengths)} | {percentile(0.1)} | {percentile(0.5)} | {percentile(0.9)} | "
            f"{lengths[-1]} | {shares[0]:.0%} | {shares[1]:.0%} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
