"""Mooncake KV offload checks under P/D, read from the prefill engines' external prefix-cache counters.

1. Cross-rank reuse: prefill rank A computes a long prompt; prefill rank B (no local copy) loads it from the store.
2. Decode save: a turn goes through the router (prefill on rank C, decode on a decode rank); the next turn's prefix
   (prompt + completion) is prefilled on rank D, which should load the completion's KV that only decode computed.

    uv run --no-sync python mooncake_check.py <router_url> <prefill_url_A> <prefill_url_B> <prefill_url_C> <prefill_url_D>
"""

import asyncio
import random
import re
import sys

import httpx

MODEL = "PrimeIntellect/DeepSeek-V4-Flash-0731-bf16"


async def external_hits(client: httpx.AsyncClient, url: str) -> tuple[float, float]:
    text = (await client.get(f"{url}/metrics", timeout=30)).text

    def total(name: str) -> float:
        return sum(float(v) for v in re.findall(rf"^vllm:{name}_total\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M))

    return total("external_prefix_cache_queries"), total("external_prefix_cache_hits")


async def complete(client: httpx.AsyncClient, url: str, prompt, max_tokens: int) -> dict:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "return_token_ids": True,
            "ignore_eos": True}
    r = await client.post(f"{url}/v1/completions", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]


async def delta(client, url, fn):
    q0, h0 = await external_hits(client, url)
    result = await fn()
    await asyncio.sleep(1)
    q1, h1 = await external_hits(client, url)
    return result, q1 - q0, h1 - h0


async def main() -> None:
    router, a, b, c, d = sys.argv[1:]
    rng = random.Random()
    salt = " ".join(str(rng.randrange(10**6)) for _ in range(8))  # fresh prefix, never cached before
    prompt = f"[{salt}] Summarize the following text.\n\n" + " ".join(
        f"Paragraph {i}: the quick brown fox jumps over the lazy dog." for i in range(600)
    )
    async with httpx.AsyncClient() as client:
        first, q, h = await delta(client, a, lambda: complete(client, a, prompt, 1))
        n = len(first["prompt_token_ids"])
        print(f"1a. rank A computes {n} prompt tokens: external queries {q:.0f}, hits {h:.0f}")
        await asyncio.sleep(3)  # let the async store put land
        _, q, h = await delta(client, b, lambda: complete(client, b, prompt, 1))
        print(f"1b. rank B, same prompt: external queries {q:.0f}, hits {h:.0f} (expect ~{n // 256 * 256})")

        context = " ".join(f"Log line {i}: the lighthouse lamp turned {i % 7} times." for i in range(90))
        turn = f"[{salt} turn] {context}\nWrite a long story about a lighthouse keeper."
        first, q, h = await delta(client, c, lambda: complete(client, router, turn, 1000))
        n_prompt, n_out = len(first["prompt_token_ids"]), len(first["token_ids"])
        print(f"2a. router turn: {n_prompt} prompt + {n_out} completion tokens")
        await asyncio.sleep(5)
        follow_up = first["prompt_token_ids"] + first["token_ids"] + first["prompt_token_ids"][:16]
        _, q, h = await delta(client, d, lambda: complete(client, d, follow_up, 1))
        print(f"2b. rank D, next turn ({len(follow_up)} tokens): external queries {q:.0f}, hits {h:.0f} "
              f"(prompt blocks only: {n_prompt // 256 * 256}; with decode-saved KV: {(n_prompt + n_out) // 256 * 256})")


asyncio.run(main())
