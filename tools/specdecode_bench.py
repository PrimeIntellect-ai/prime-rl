"""Decode throughput and speculative acceptance on held-out `specdecode_windows.py` episodes.

Each held-out window gives one prompt: its exact token prefix up to the window's first generated
turn. Sampling defaults to temperature 1.0, top-p 0.95 and top-k 128. Run it against the same
server with and without the draft and compare the JSON lines appended to `--out`.

Usage (from the prime-rl repo, next to a running inference server):
    uv run python tools/specdecode_bench.py <windows_dir>/heldout --model <served model> --label dspark
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import openai
from datasets import load_from_disk

SPEC_COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")


def prompts_from(windows, limit: int) -> list[list[int]]:
    prompts = []
    for row in windows:
        mask = row["loss_mask"]
        # The DSpark mask marks the position before each sampled token: a run starting at i means
        # the turn's first sampled token is i + 1, so the prompt ends at i + 1.
        start = next((i for i, m in enumerate(mask) if m and (i == 0 or not mask[i - 1])), None)
        if start is not None:
            prompts.append(row["prefix_ids"][: len(row["prefix_ids"]) - row["seq_len"] + start + 1])
        if len(prompts) == limit:
            break
    return prompts


def spec_counters(url: str) -> dict[str, float]:
    text = httpx.get(url.removesuffix("/v1") + "/metrics", timeout=30).text
    return {
        name: sum(float(v) for v in re.findall(rf"^vllm:spec_decode_{name}_total(?:{{[^}}]*}})? (\S+)$", text, re.M))
        for name in SPEC_COUNTERS
    }


async def run(client, args, prompts: list[list[int]], concurrency: int) -> tuple[int, float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(prompt: list[int]) -> int:
        async with semaphore:
            response = await client.completions.create(
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body={"top_k": args.top_k},
            )
            return response.usage.completion_tokens

    start = time.perf_counter()
    tokens = await asyncio.gather(*(one(p) for p in prompts))
    return sum(tokens), time.perf_counter() - start


async def bench(args: argparse.Namespace) -> None:
    prompts = prompts_from(load_from_disk(str(args.heldout)), args.prompts)
    client = openai.AsyncOpenAI(base_url=args.url, api_key="EMPTY", timeout=None)
    for concurrency in args.concurrency:
        level = (prompts * (1 + 2 * concurrency // len(prompts)))[: max(len(prompts), 2 * concurrency)]
        before = spec_counters(args.url)
        tokens, seconds = await run(client, args, level, concurrency)
        after = spec_counters(args.url)
        drafts = after["num_drafts"] - before["num_drafts"]
        accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
        result = {
            "label": args.label,
            "concurrency": concurrency,
            "requests": len(level),
            "output_tokens": tokens,
            "seconds": round(seconds, 1),
            "output_tokens_per_s": round(tokens / seconds, 1),
            "mean_acceptance_length": round(1 + accepted / drafts, 3) if drafts else None,
        }
        print(json.dumps(result), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("heldout", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument("--prompts", type=int, default=256)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 128])
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("specdecode_bench.jsonl"))
    asyncio.run(bench(parser.parse_args()))


if __name__ == "__main__":
    main()
