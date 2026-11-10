"""Extract target hidden states for `specdecode_windows.py` windows into a Speculators dataset.

The training windows of every given directory are shuffled together and the `--val` windows follow
them, so Speculators' sequential split validates on exactly those (`--finalize` prints the
`train.data.train_data_ratio` to use). The extraction server prefills each window's full prefix and
`TailHiddenStatesConnector` writes the window's positions only
(examples/specdecode/deepseek-v4/extract.toml). Each file is checked against the window's token ids
and staged; `--finalize` numbers the extracted windows densely into `<out_dir>/hidden_states` and saves
their rows to `<out_dir>`. Rerunning with the same directories resumes.

Usage (from the prime-rl repo, next to a running extraction server):
    uv run python tools/specdecode_extract.py <out_dir> <windows>/train ... --val <windows>/heldout ... --model <served model>
    (the same command with --finalize once extraction is done)
"""

import argparse
import asyncio
import fcntl
import os
from pathlib import Path

import openai
from datasets import Sequence, Value, concatenate_datasets, load_from_disk
from safetensors import safe_open
from speculators.data_generation.vllm_client import generate_hidden_states_async


def claim(handle: str, dest: Path, window_ids: list[int]) -> None:
    with open(handle + ".lock") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)  # released once the connector has written the file
    with safe_open(handle, "pt") as f:
        token_ids = f.get_tensor("token_ids").tolist()
    if token_ids != window_ids:
        raise ValueError(f"{handle}: extracted token ids do not match the window")
    os.replace(handle, dest)
    os.remove(handle + ".lock")


async def extract(args: argparse.Namespace, windows) -> None:
    staging = args.out_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    client = openai.AsyncOpenAI(base_url=args.url, api_key="EMPTY", timeout=None)
    semaphore = asyncio.Semaphore(args.concurrency)
    counts = {"done": 0, "failed": 0}

    async def one(i: int) -> None:
        dest = staging / f"win_{i}.safetensors"
        if dest.exists():
            return
        row = windows[i]
        async with semaphore:
            try:
                handle, _ = await generate_hidden_states_async(
                    client, args.model, {"input_ids": row["prefix_ids"]}, timeout=args.timeout
                )
                await asyncio.to_thread(claim, handle, dest, row["input_ids"])
                counts["done"] += 1
            except Exception as e:
                counts["failed"] += 1
                print(f"window {i} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        if sum(counts.values()) % 50 == 0:
            print(f"{counts} of {len(windows)}", flush=True)

    await asyncio.gather(*(one(i) for i in range(len(windows))))
    print(f"{counts} of {len(windows)}", flush=True)


def finalize(args: argparse.Namespace, windows, num_train_windows: int) -> None:
    staging, hidden_states = args.out_dir / "staging", args.out_dir / "hidden_states"
    hidden_states.mkdir(parents=True, exist_ok=True)
    keep = []
    for i in range(len(windows)):
        source = staging / f"win_{i}.safetensors"
        if source.exists():
            os.replace(source, hidden_states / f"hs_{len(keep)}.safetensors")
            keep.append(i)
    rows = windows.select(keep).select_columns(["input_ids", "loss_mask", "seq_len"])
    rows = rows.cast_column("input_ids", Sequence(Value("int64")))  # as the extracted token_ids
    rows.set_format("torch")
    rows.save_to_disk(str(args.out_dir))
    num_train = sum(i < num_train_windows for i in keep)
    print(f"{len(keep)} of {len(windows)} windows, {sum(rows['seq_len']).item() / 1e6:.2f}M tokens")
    print(f"train.data.train_data_ratio = {num_train / len(keep)} ({num_train} train, {len(keep) - num_train} val)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("windows_dirs", type=Path, nargs="+", help="training windows")
    parser.add_argument("--val", type=Path, nargs="*", default=[], help="validation windows, placed last")
    parser.add_argument("--model", required=True, help="model name served by the extraction server")
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    columns = ["input_ids", "loss_mask", "seq_len", "prefix_ids", "env"]
    train = concatenate_datasets([load_from_disk(str(d)).select_columns(columns) for d in args.windows_dirs])
    train = train.shuffle(seed=0)
    val = [load_from_disk(str(d)).select_columns(columns) for d in args.val]
    windows = concatenate_datasets([train, *val])
    if args.finalize:
        finalize(args, windows, len(train))
    else:
        asyncio.run(extract(args, windows))


if __name__ == "__main__":
    main()
