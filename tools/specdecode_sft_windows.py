"""Sample speculative-draft training windows from an SFT run's data, rendered as that run trained on it.

Reads the run's resolved config (data, renderer, tokenizer), loads the rows of `--files` random parquet
files of its dataset (subsets in proportion to their files), renders them in shuffled order with
prime-rl's SFT dataset, and keeps one window of up to `--window` tokens per row with its exact prefix,
in `specdecode_windows.py`'s row format. A fraction of rows is held out.

Usage (from the prime-rl repo; pass the run's launch directory when its renderer paths are relative):
    uv run python tools/specdecode_sft_windows.py <sft_run_dir> <out_dir> [--launch-dir DIR] [--tokens 6e6]
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from pydantic import TypeAdapter
from renderers import RendererConfig

from prime_rl.configs.sft import DataConfig, SFTDataConfig
from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.specdecode.training import draft_loss_mask
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.trainer.sft.data import setup_sft_dataset
from prime_rl.utils.pathing import get_config_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--launch-dir", type=Path, help="directory the run was launched from")
    parser.add_argument("--tokens", type=float, default=6e6, help="window tokens to sample in total")
    parser.add_argument("--files", type=int, default=64, help="random parquet files of the dataset to draw rows from")
    parser.add_argument("--window", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=8, help="DSpark block size")
    parser.add_argument("--heldout", type=float, default=0.03, help="fraction of rows held out")
    parser.add_argument("--min-loss-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_config = json.loads((get_config_dir(args.run_dir.resolve()) / "sft.json").read_text())
    out_dir = args.out_dir.resolve()
    if args.launch_dir is not None:
        os.chdir(args.launch_dir)

    # Older runs carry data fields main has since dropped (e.g. packing); rendering needs none of them.
    data = {k: v for k, v in run_config["data"].items() if k in SFTDataConfig.model_fields}
    data_config = TypeAdapter(DataConfig).validate_python({**data, "seed": args.seed})
    tokenizer = setup_tokenizer(TokenizerConfig(**run_config["tokenizer"]))
    renderer_config = TypeAdapter(RendererConfig).validate_python(run_config["renderer"])
    rng = random.Random(args.seed)
    files = rng.sample(sorted(Path(data_config.name).glob("data/**/*.parquet")), args.files)
    columns = {data_config.columns.messages, data_config.columns.tools, data_config.columns.message_loss_mask}
    columns |= set(data_config.columns.renderer.values())
    raw = Dataset(pa.concat_tables([pq.read_table(f, columns=sorted(columns)) for f in files]))
    dataset = setup_sft_dataset(tokenizer, data_config, renderer_config=renderer_config, max_epochs=1, raw_dataset=raw)

    splits = {"train": [], "heldout": []}
    taken = 0
    for i, sample in enumerate(dataset):
        if taken >= args.tokens:
            break
        # The SFT sample is causally shifted: loss_mask[i] marks predicting token i + 1, as DSpark anchors do.
        token_ids = np.asarray([*sample["input_ids"], sample["target_ids"][-1]], dtype=np.int64)
        loss_mask = torch.as_tensor(sample["loss_mask"], dtype=torch.bool)
        end = rng.randint(min(args.window, len(loss_mask)), len(loss_mask))
        start = max(0, end - args.window)
        mask = draft_loss_mask(loss_mask[start:end][None], torch.tensor([end - start]), args.block_size, False)[0]
        if mask.sum() < args.min_loss_frac * (end - start):
            continue
        split = "heldout" if rng.random() < args.heldout else "train"
        splits[split].append(
            {
                "input_ids": token_ids[start:end],
                "loss_mask": mask.numpy(),
                "seq_len": end - start,
                "prefix_ids": token_ids[:end].astype(np.int32),
                "env": "sft",
                "episode": str(i),
            }
        )
        taken += end - start
    for split, rows in splits.items():
        Dataset.from_list(rows).save_to_disk(str(out_dir / split))
        print(f"{split}: {len(rows)} windows, {sum(r['seq_len'] for r in rows) / 1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
