"""P/D smoke test: greedy completions through the P/D router, teacher-force scored on single engines.

The router path prefills on a prefill engine and decodes on a decode engine from the transferred KV. Each direct
engine then scores the router's tokens with its own prefill (prompt_logprobs). With a correct KV transfer the
router's logprobs match these scores up to numerics; a corrupted transfer gives the decode-side tokens much
lower logprobs under the direct engines.

    uv run --no-sync python pd_smoke.py <router_url> <direct_url> [<direct_url> ...]
"""

import asyncio
import sys

import httpx

MODEL = "PrimeIntellect/DeepSeek-V4-Flash-0731-bf16"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Explain the difference between TCP and UDP in one paragraph:",
    "1, 1, 2, 3, 5, 8, 13,",
    # Long prompts span many KV blocks (block_size 256) and the compressed/indexer caches.
    "Summarize the following text.\n\n" + " ".join(f"Paragraph {i}: the quick brown fox jumps over the lazy dog." for i in range(400)),
    "Count upward: " + " ".join(str(i) for i in range(3000)),
]


async def post(client: httpx.AsyncClient, url: str, body: dict) -> dict:
    r = await client.post(f"{url}/v1/completions", json={"model": MODEL, **body}, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]


async def score(client: httpx.AsyncClient, url: str, prompt_ids: list[int], ids: list[int]) -> list[float]:
    choice = await post(client, url, {"prompt": prompt_ids + ids, "max_tokens": 1, "prompt_logprobs": 0})
    rows = choice["prompt_logprobs"][len(prompt_ids) :]
    return [row[str(t)]["logprob"] for row, t in zip(rows, ids)]


async def main() -> None:
    router, *directs = sys.argv[1:]
    worst = 0.0
    async with httpx.AsyncClient() as client:
        for prompt in PROMPTS:
            body = {"prompt": prompt, "max_tokens": 48, "temperature": 0.0, "logprobs": 0, "return_token_ids": True}
            choice = await post(client, router, body)
            ids, logprobs = choice["token_ids"], choice["logprobs"]["token_logprobs"]
            scores = await asyncio.gather(*(score(client, url, choice["prompt_token_ids"], ids) for url in directs))
            line = f"prompt_tokens={len(choice['prompt_token_ids']):>5} out={len(ids):>3}"
            for url, s in zip(directs, scores):
                diffs = [abs(a - b) for a, b in zip(logprobs, s)]
                worst = max(worst, max(diffs))
                line += f" | {url.rsplit(':', 1)[-1]}: mean|dlp| {sum(diffs) / len(diffs):.4f} max {max(diffs):.3f}"
            print(line, "|", repr(choice["text"][:50]))
    print("PASS" if worst < 1.0 else "FAIL", f"(max |dlogprob| {worst:.3f})")


asyncio.run(main())
