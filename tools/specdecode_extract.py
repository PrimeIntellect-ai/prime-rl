"""Extract target hidden states for `specdecode_windows.py` windows into a Speculators dataset.

The extraction server prefills each window's full prefix and `TailHiddenStatesConnector` writes the
window's positions only (examples/specdecode/deepseek-v4/extract.toml). Each file is checked
against the window's token ids and staged; `--finalize` numbers the extracted windows densely into
`<out_dir>/hidden_states` and saves their rows to `<out_dir>`. Rerunning resumes.

Usage (from the prime-rl repo, next to a running extraction server):
    uv run python tools/specdecode_extract.py <windows_dir>/train <out_dir> --model <served model>
    uv run python tools/specdecode_extract.py <windows_dir>/train <out_dir> --model <served model> --finalize
"""

import argparse
import asyncio
import fcntl
import os
from pathlib import Path

import openai
from datasets import load_from_disk
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


def finalize(args: argparse.Namespace, windows) -> None:
    staging, hidden_states = args.out_dir / "staging", args.out_dir / "hidden_states"
    hidden_states.mkdir(parents=True, exist_ok=True)
    keep = []
    for i in range(len(windows)):
        source = staging / f"win_{i}.safetensors"
        if source.exists():
            os.replace(source, hidden_states / f"hs_{len(keep)}.safetensors")
            keep.append(i)
    rows = windows.select(keep).select_columns(["input_ids", "loss_mask", "seq_len"])
    rows.set_format("torch")
    rows.save_to_disk(str(args.out_dir))
    print(f"{len(keep)} of {len(windows)} windows, {sum(rows['seq_len']).item() / 1e6:.2f}M tokens")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("windows_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--model", required=True, help="model name served by the extraction server")
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    windows = load_from_disk(str(args.windows_dir))
    if args.finalize:
        finalize(args, windows)
    else:
        asyncio.run(extract(args, windows))


if __name__ == "__main__":
    main()
