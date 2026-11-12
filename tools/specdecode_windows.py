"""Sample speculative-draft training windows from an RL run's episode traces.

Each sampled episode contributes one window of up to `--window` tokens from one of its trainable
branches (verifiers' `Trace.branches`: the trainer's token ids and sampled-token masks). A window
keeps its exact token prefix, so the target can prefill the whole context before the window.
Episodes are the run's successful train episodes (all, or the most recent `--recent`); each env's
token budget follows the square root of its decoded tokens there. Rows follow the Speculators schema
(`input_ids`, `loss_mask` aligned to DSpark anchors, `seq_len`) plus `prefix_ids`. Whole task groups
(the rollouts of one task) are held out, so validation never shares a task's context with training.

Usage (from the prime-rl repo; pass the run's launch directory when its renderer path is relative):
    uv run python tools/specdecode_windows.py <run_dir> <out_dir> [--launch-dir DIR] [--tokens 10e6]
"""

import argparse
import collections
import hashlib
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from verifiers.v1 import Trace

from prime_rl.monitors.file.traces.chunks import open_chunk
from prime_rl.specdecode.training import draft_loss_mask


def sample_window(task: tuple) -> dict | None:
    stream_dir, entry, window, block_size, min_loss_frac, seed = task
    rng = random.Random(seed)
    with open_chunk(stream_dir, entry["chunk"]) as f:
        f.seek(entry["offset"])
        record = json.loads(f.readline())
    branches = [
        branch
        for trace in record["traces"]
        for branch in Trace.model_validate(trace).branches
        if branch.trainable and branch.sampled_mask.any()
    ]
    if not branches:
        return None
    branch = rng.choices(branches, weights=[len(b.token_ids) for b in branches])[0]
    token_ids, sampled = branch.token_ids, branch.sampled_mask
    for _ in range(5):
        end = rng.randint(min(window, len(token_ids)), len(token_ids))
        start = max(0, end - window)
        sampled_window = torch.from_numpy(sampled[start:end])[None]
        mask = draft_loss_mask(sampled_window, torch.tensor([end - start]), block_size, predict_next=True)[0]
        if mask.sum() >= min_loss_frac * (end - start):
            return {
                "input_ids": token_ids[start:end].astype(np.int64),
                "loss_mask": mask.numpy(),
                "seq_len": end - start,
                "prefix_ids": token_ids[:end],
                "env": entry["env"],
                "group": entry["group"],
                "episode": entry["id"],
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--launch-dir", type=Path, help="directory the run was launched from")
    parser.add_argument("--tokens", type=float, default=10e6, help="window tokens to sample in total")
    parser.add_argument("--window", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=8, help="DSpark block size")
    parser.add_argument("--recent", type=int, default=0, help="most recent successful train episodes; 0 for all")
    parser.add_argument("--heldout", type=float, default=0.02, help="fraction of task groups held out")
    parser.add_argument("--min-loss-frac", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    traces_dir = args.run_dir.resolve() / "monitors/file/traces"
    out_dir = args.out_dir.resolve()
    if args.launch_dir is not None:
        os.chdir(args.launch_dir)

    with open(traces_dir / "stream.index.jsonl") as f:
        entries = [json.loads(line) for line in f]
    entries = [e for e in entries if e.get("kind") == "train" and e.get("ok") and e.get("output_tokens")]
    episodes = collections.defaultdict(list)
    for entry in entries[-args.recent :] if args.recent else entries:
        episodes[entry["env"]].append(entry)
    weights = {env: math.sqrt(sum(e["output_tokens"] for e in eps)) for env, eps in episodes.items()}
    budgets = {env: args.tokens * w / sum(weights.values()) for env, w in weights.items()}

    rng = random.Random(args.seed)
    rows = collections.defaultdict(list)
    with ProcessPoolExecutor(args.workers) as pool:
        for env, eps in sorted(episodes.items()):
            rng.shuffle(eps)
            taken = 0
            for i in range(0, len(eps), args.workers):
                if taken >= budgets[env]:
                    break
                tasks = [
                    (traces_dir / "stream", e, args.window, args.block_size, args.min_loss_frac, rng.random())
                    for e in eps[i : i + args.workers]
                ]
                for row in pool.map(sample_window, tasks):
                    if row is not None and taken < budgets[env]:
                        rows[env].append(row)
                        taken += row["seq_len"]
            print(f"{env}: {len(rows[env])} windows, {taken / 1e6:.2f}M tokens (budget {budgets[env] / 1e6:.2f}M)")

    def held_out(group: str) -> bool:
        return int(hashlib.sha256(group.encode()).hexdigest(), 16) % 10_000 < args.heldout * 10_000

    train = [row for env_rows in rows.values() for row in env_rows if not held_out(row["group"])]
    heldout = [row for env_rows in rows.values() for row in env_rows if held_out(row["group"])]
    rng.shuffle(train)
    for split, split_rows in (("train", train), ("heldout", heldout)):
        Dataset.from_list(split_rows).save_to_disk(str(out_dir / split))
        print(f"{split}: {len(split_rows)} windows, {sum(r['seq_len'] for r in split_rows) / 1e6:.2f}M tokens")


if __name__ == "__main__":
    main()
