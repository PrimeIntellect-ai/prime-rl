"""Correctness of Mooncake-loaded KV: rank A computes a fresh long prompt (and stores it); rank B loads the prefix from
the store. Both decode greedily; B's tokens are teacher-force scored on A (whose prefix sits in its GPU cache).

    uv run --no-sync python mooncake_load_check.py <prefill_url_A> <prefill_url_B>
"""

import asyncio
import random
import sys

import httpx

MODEL = "PrimeIntellect/DeepSeek-V4-Flash-0731-bf16"


async def post(client, url, body):
    r = await client.post(f"{url}/v1/completions", json={"model": MODEL, **body}, timeout=600)
    r.raise_for_status()
    return r.json()


async def main() -> None:
    a, b = sys.argv[1:]
    salt = " ".join(str(random.randrange(10**6)) for _ in range(8))
    prompt = f"[{salt}] Here is a list of facts about lighthouses.\n" + " ".join(
        f"Fact {i}: lighthouse number {i} stands {i * 3 % 97} meters tall." for i in range(500)
    ) + "\nNow summarize the facts above in three sentences:"
    gen = {"prompt": prompt, "max_tokens": 64, "temperature": 0.0, "logprobs": 0, "return_token_ids": True}
    async with httpx.AsyncClient() as client:
        ra = (await post(client, a, gen))["choices"][0]
        await asyncio.sleep(3)
        for name, url in (("A again (GPU prefix hit)", a), ("B (Mooncake load)", b)):
            resp = await post(client, url, gen)
            rb = resp["choices"][0]
            cached = (resp["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            ids = rb["token_ids"]
            body = {"prompt": rb["prompt_token_ids"] + ids, "max_tokens": 1, "prompt_logprobs": 0}
            rows = (await post(client, a, body))["choices"][0]["prompt_logprobs"][len(rb["prompt_token_ids"]) :]
            lps = rb["logprobs"]["token_logprobs"]
            diffs = [abs(row[str(t)]["logprob"] - lp) for row, t, lp in zip(rows, ids, lps)]
            same = next((i for i, (x, y) in enumerate(zip(ra["token_ids"], ids)) if x != y), len(ids))
            print(f"{name}: prompt {len(rb['prompt_token_ids'])}, cached {cached}; same as A for {same}/{len(ids)}; "
                  f"scored on A: mean|dlp| {sum(diffs) / len(diffs):.4f} max {max(diffs):.3f}; first lp {lps[0]:.3f} "
                  f"vs A {ra['logprobs']['token_logprobs'][0]:.3f}")


asyncio.run(main())
