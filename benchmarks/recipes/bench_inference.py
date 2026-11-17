"""Drive an inference server with synthetic agentic sessions and report sustained throughput.

Each session is a multi-turn rollout: a prompt, then turns of (model output, think time, tool
output) appended to one growing context, until the context reaches the session's target length.
Target lengths are log-normal and capped at the max model length, so most sessions stay short
and a heavy tail reaches the cap. A fixed number of sessions runs concurrently (closed loop).
The first session of each worker starts at a random point of its rollout, so long sessions are
in flight from the start. That start is a burst of cold prefill, so the warmup lasts at least
--warmup seconds and then until the queue drains (capped at --max-warmup); only requests that
finish after the warmup count.

    python benchmarks/recipes/bench_inference.py --base-url http://localhost:8000 \
        --metrics-url http://localhost:8100/metrics --model Qwen/Qwen3.6-35B-A3B \
        --concurrency 128 --duration 300 --out result.json
"""

import argparse
import asyncio
import json
import math
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field

import httpx

VOCAB = 100_000  # random token ids stay inside every modern vocab


@dataclass
class Workload:
    max_len: int = 262_144
    target_median: float = 24_000
    target_sigma: float = 1.2
    full_len_prob: float = 0.03
    prompt_median: float = 4_000
    prompt_sigma: float = 0.5
    output_median: float = 600
    output_sigma: float = 0.8
    output_max: int = 8_192
    tool_median: float = 1_200
    tool_sigma: float = 1.0
    tool_max: int = 16_384
    think_median_s: float = 2.0
    think_sigma: float = 0.8


@dataclass
class Stats:
    requests: int = 0
    errors: int = 0
    sessions_done: int = 0
    output_tokens: int = 0
    prompt_tokens: int = 0
    full_len_sessions_done: int = 0
    ttft_s: list[float] = field(default_factory=list)
    itl_s: list[float] = field(default_factory=list)
    final_lens: list[int] = field(default_factory=list)


def lognormal(rng: random.Random, median: float, sigma: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, rng.lognormvariate(math.log(median), sigma))))


def random_tokens(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(VOCAB) for _ in range(n)]


async def run_session(client, args, wl: Workload, rng: random.Random, stats: Stats, deadline: float, warm: bool):
    full = rng.random() < wl.full_len_prob
    target = wl.max_len - 16 if full else lognormal(rng, wl.target_median, wl.target_sigma, 2_048, wl.max_len - 16)
    prompt = min(target // 2, lognormal(rng, wl.prompt_median, wl.prompt_sigma, 256, 32_768))
    context = random_tokens(rng, max(prompt, int(rng.random() * target)) if warm else prompt)
    session = uuid.uuid4().hex
    while time.monotonic() < deadline:
        max_tokens = min(lognormal(rng, wl.output_median, wl.output_sigma, 16, wl.output_max), target - len(context))
        if max_tokens < 16:
            break
        body = {
            "model": args.model,
            "prompt": context,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        }
        start = time.monotonic()
        first = last = None
        out_ids: list[int] = []
        try:
            async with client.stream("POST", "/v1/completions", json=body, headers={"x-session-id": session}) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    for choice in chunk.get("choices", []):
                        now = time.monotonic()
                        if first is None:
                            first = now
                        elif last is not None and now >= args.measure_from:
                            stats.itl_s.append(now - last)
                        last = now
                        out_ids.extend(choice.get("token_ids") or [])
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            stats.errors += 1
            if args.verbose:
                print(f"request error: {e!r}")
            await asyncio.sleep(1)
            return
        if time.monotonic() >= args.measure_from:
            stats.requests += 1
            stats.prompt_tokens += len(context)
            stats.output_tokens += max_tokens
            if first is not None:
                stats.ttft_s.append(first - start)
        # Without returned ids the next turn misses the prefix cache on this turn's output.
        context = context + (out_ids if len(out_ids) == max_tokens else random_tokens(rng, max_tokens))
        if len(context) >= target - 16:
            break
        await asyncio.sleep(lognormal(rng, wl.think_median_s * 1000, wl.think_sigma, 50, 60_000) / 1000)
        tool = min(lognormal(rng, wl.tool_median, wl.tool_sigma, 16, wl.tool_max), target - len(context) - 16)
        if tool <= 0:
            break
        context = context + random_tokens(rng, tool)
    else:
        return
    if time.monotonic() < args.measure_from:
        return
    stats.sessions_done += 1
    stats.final_lens.append(len(context))
    if full:
        stats.full_len_sessions_done += 1


async def worker(client, args, wl, rng, stats, deadline):
    warm = True
    while time.monotonic() < deadline:
        await run_session(client, args, wl, rng, stats, deadline, warm)
        warm = False


METRIC = re.compile(r"^(vllm:[a-z_]+)(\{[^}]*\})? ([0-9.eE+-]+)$")


async def scrape(client: httpx.AsyncClient, url: str) -> dict[str, float]:
    """Sum each vLLM metric over its label sets (DP ranks, finish reasons, ...)."""
    try:
        text = (await client.get(url)).text
    except httpx.HTTPError:
        return {}
    totals: dict[str, float] = {}
    for line in text.splitlines():
        m = METRIC.match(line)
        if m:
            totals[m.group(1)] = totals.get(m.group(1), 0.0) + float(m.group(3))
    return totals


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


async def main(args):
    wl = Workload(max_len=args.max_len)
    stats = Stats()
    limits = httpx.Limits(max_connections=args.concurrency + 8, max_keepalive_connections=args.concurrency + 8)
    timeout = httpx.Timeout(3600, connect=30)
    async with (
        httpx.AsyncClient(base_url=args.base_url, limits=limits, timeout=timeout) as client,
        httpx.AsyncClient(timeout=10) as metrics_client,
    ):
        kv_samples, waiting_samples = [], []
        start = time.monotonic()
        args.measure_from = math.inf
        deadline = math.inf
        rngs = [random.Random(args.seed * 100_003 + i) for i in range(args.concurrency)]
        tasks = [
            asyncio.create_task(worker(client, args, wl, rngs[i], stats, deadline)) for i in range(args.concurrency)
        ]
        await asyncio.sleep(args.warmup)
        drained = 0
        while drained < 2 and time.monotonic() - start < args.max_warmup:
            await asyncio.sleep(10)
            waiting = (await scrape(metrics_client, args.metrics_url)).get("vllm:num_requests_waiting", 0.0)
            drained = drained + 1 if waiting <= max(2, 0.01 * args.concurrency) else 0
        warmup_s = time.monotonic() - start
        before = await scrape(metrics_client, args.metrics_url)
        start = args.measure_from = time.monotonic()
        deadline = start + args.duration
        while time.monotonic() < deadline:
            await asyncio.sleep(10)
            snap = await scrape(metrics_client, args.metrics_url)
            if "vllm:kv_cache_usage_perc" in snap:
                kv_samples.append(snap["vllm:kv_cache_usage_perc"] / max(1, args.dp))
            waiting_samples.append(snap.get("vllm:num_requests_waiting", 0.0))
            if args.verbose:
                print(
                    f"t={time.monotonic() - start:.0f}s reqs={stats.requests} out={stats.output_tokens} "
                    f"kv={kv_samples[-1] if kv_samples else None} waiting={waiting_samples[-1]}"
                )
        elapsed = time.monotonic() - start
        # In-flight requests past the deadline are not counted; drop them.
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        after = await scrape(metrics_client, args.metrics_url)

    def delta(key):
        return after.get(key, 0.0) - before.get(key, 0.0)

    hits, queries = delta("vllm:prefix_cache_hits_total"), delta("vllm:prefix_cache_queries_total")
    result = {
        "concurrency": args.concurrency,
        "duration_s": elapsed,
        "warmup_s": warmup_s,
        "warmup_drained": drained >= 2,
        "gpus": args.gpus,
        "requests": stats.requests,
        "errors": stats.errors,
        "sessions_done": stats.sessions_done,
        "full_len_sessions_done": stats.full_len_sessions_done,
        "output_tok_s": stats.output_tokens / elapsed,
        "output_tok_s_per_gpu": stats.output_tokens / elapsed / args.gpus,
        "prompt_tok_s": stats.prompt_tokens / elapsed,
        "engine_prompt_tok_s": delta("vllm:prompt_tokens_total") / elapsed,
        "engine_generation_tok_s": delta("vllm:generation_tokens_total") / elapsed,
        "prefix_cache_hit_rate": hits / queries if queries else None,
        "preemptions": delta("vllm:num_preemptions_total"),
        "kv_usage_mean": sum(kv_samples) / len(kv_samples) if kv_samples else None,
        "kv_usage_max": max(kv_samples) if kv_samples else None,
        "waiting_mean": sum(waiting_samples) / len(waiting_samples) if waiting_samples else None,
        "ttft_p50_s": pct(stats.ttft_s, 0.5),
        "ttft_p99_s": pct(stats.ttft_s, 0.99),
        "itl_p50_ms": (pct(stats.itl_s, 0.5) or 0) * 1000,
        "itl_p99_ms": (pct(stats.itl_s, 0.99) or 0) * 1000,
        "final_len_p50": pct(stats.final_lens, 0.5),
        "final_len_p99": pct(stats.final_lens, 0.99),
        "workload": asdict(wl),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--metrics-url", default="http://localhost:8100/metrics")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--duration", type=float, default=300, help="measured seconds, after warmup")
    p.add_argument("--warmup", type=float, default=90, help="minimum warmup seconds")
    p.add_argument("--max-warmup", type=float, default=360)
    p.add_argument("--max-len", type=int, default=262_144)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--dp", type=int, default=1, help="DP ranks, to average kv_cache_usage_perc over")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.add_argument("--verbose", action="store_true")
    asyncio.run(main(p.parse_args()))
