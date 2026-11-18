"""Concurrency staircase for one stress arm: pins the eval's episode cap level by level through its control API and
records the engines' load every window.

    stress_driver.py --control http://localhost:8045 --schedule 128:60,256:60,512:90 --out arm.jsonl \
        --engine http://host:8100=prefill ... --engine http://host:8200=decode ...

Each window row sums every engine's Prometheus counters into rates: generated and prompt tokens/s, prefill tokens
actually computed (prefix-cache queries minus GPU and external (Mooncake) hits), hit rates, preemptions/s, plus
running/waiting gauges and the TTFT / inter-token / end-to-end latency means of the requests finished in the window.
Rows are per role ("all" is the total). A lower cap cancels no episode, so a level below the episodes in flight
first drains (``phase: drain``, at most ``--max-drain`` minutes) and its hold timer starts once they are down to
the level. A summary row closes each level, averaged over the second half of its hold.
"""

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict

import httpx

SAMPLE = re.compile(r"^vllm:([a-z_:]+?)(?:\{[^}]*\})? ([0-9.eE+-]+|NaN)$", re.M)
COUNTERS = {
    "generation_tokens_total": "gen",
    "prompt_tokens_total": "prompt",
    "prefix_cache_queries_total": "pc_q",
    "prefix_cache_hits_total": "pc_h",
    "external_prefix_cache_queries_total": "ext_q",
    "external_prefix_cache_hits_total": "ext_h",
    "num_preemptions_total": "preempt",
    "request_success_total": "done",
    "time_to_first_token_seconds_sum": "ttft_s",
    "time_to_first_token_seconds_count": "ttft_n",
    "inter_token_latency_seconds_sum": "itl_s",
    "inter_token_latency_seconds_count": "itl_n",
    "e2e_request_latency_seconds_sum": "e2e_s",
    "e2e_request_latency_seconds_count": "e2e_n",
}
GAUGES = {"num_requests_running": "running", "num_requests_waiting": "waiting", "kv_cache_usage_perc": "kv"}


async def scrape(client: httpx.AsyncClient, url: str) -> dict[str, float] | None:
    try:
        text = (await client.get(f"{url}/metrics", timeout=20)).text
    except httpx.HTTPError:
        return None
    values: dict[str, float] = defaultdict(float)
    for name, value in SAMPLE.findall(text):
        key = COUNTERS.get(name) or GAUGES.get(name)
        if key is not None and value != "NaN":
            values[key] += float(value)
    return values


def window_row(prev: dict, cur: dict, dt: float, engines: dict[str, str]) -> dict:
    """Rates and gauges per role over one window; engines that failed a scrape are left out of both ends."""
    out = {}
    for role in sorted(set(engines.values())) + ["all"]:
        urls = [u for u, r in engines.items() if (role == "all" or r == role) and prev.get(u) and cur.get(u)]
        d = {k: sum(cur[u].get(k, 0.0) - prev[u].get(k, 0.0) for u in urls) for k in COUNTERS.values()}
        g = {k: sum(cur[u].get(k, 0.0) for u in urls) for k in GAUGES.values()}
        computed = d["pc_q"] - d["pc_h"] - d["ext_h"]
        out[role] = {
            "engines": len(urls),
            "gen_tok_s": d["gen"] / dt,
            "prompt_tok_s": d["prompt"] / dt,
            "prefill_computed_tok_s": computed / dt,
            "gpu_hit_rate": d["pc_h"] / d["pc_q"] if d["pc_q"] else None,
            "ext_hit_rate": d["ext_h"] / d["ext_q"] if d["ext_q"] else None,
            "preempt_s": d["preempt"] / dt,
            "req_done_s": d["done"] / dt,
            "ttft_mean": d["ttft_s"] / d["ttft_n"] if d["ttft_n"] else None,
            "itl_mean": d["itl_s"] / d["itl_n"] if d["itl_n"] else None,
            "e2e_mean": d["e2e_s"] / d["e2e_n"] if d["e2e_n"] else None,
            "running": g["running"],
            "waiting": g["waiting"],
            "kv_mean": g["kv"] / len(urls) if urls else None,
        }
    return out


def summarize(rows: list[dict]) -> dict:
    """Mean of every numeric field over the level's second half."""
    half = rows[len(rows) // 2 :]
    summary = {}
    for role in half[0]["roles"]:
        fields = defaultdict(list)
        for row in half:
            for k, v in row["roles"][role].items():
                if v is not None:
                    fields[k].append(v)
        summary[role] = {k: sum(v) / len(v) for k, v in fields.items()}
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--schedule", required=True, help="level:minutes,...")
    parser.add_argument("--engine", action="append", required=True, help="url[=role]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=float, default=60.0)
    parser.add_argument(
        "--max-drain", type=float, default=60.0, help="minutes a level waits for in-flight episodes to drain"
    )
    args = parser.parse_args()
    engines = dict(e.split("=", 1) if "=" in e else (e, "agg") for e in args.engine)
    schedule = [tuple(map(int, step.split(":"))) for step in args.schedule.split(",")]

    async with httpx.AsyncClient() as client:
        while True:
            try:
                (await client.get(f"{args.control}/v1/concurrency", timeout=10)).raise_for_status()
                break
            except httpx.HTTPError:
                await asyncio.sleep(10)
        out = open(args.out, "a")
        prev = dict(zip(engines, await asyncio.gather(*(scrape(client, u) for u in engines))))
        prev_t = time.time()
        for level, minutes in schedule:
            r = await client.put(f"{args.control}/v1/concurrency", json={"target": level}, timeout=10)
            r.raise_for_status()
            phase = "drain" if r.json()["inflight"] > level else "hold"
            begin = start = time.time()
            level_rows = []
            print(f"=== level {level} for {minutes} min ({phase}): {r.json()}", flush=True)
            while phase == "drain" or time.time() - start < minutes * 60:
                await asyncio.sleep(args.window)
                cur = dict(zip(engines, await asyncio.gather(*(scrape(client, u) for u in engines))))
                now = time.time()
                try:
                    conc = (await client.get(f"{args.control}/v1/concurrency", timeout=10)).json()
                except httpx.HTTPError:
                    conc = {}
                row = {
                    "kind": "window",
                    "t": now,
                    "level": level,
                    "phase": phase,
                    "level_min": (now - start) / 60,
                    "cap": conc.get("cap"),
                    "inflight": conc.get("inflight"),
                    "roles": window_row(prev, cur, now - prev_t, engines),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                a = row["roles"]["all"]
                print(
                    f"L{level} {phase} {row['level_min']:5.1f}m inflight {conc.get('inflight')} | gen {a['gen_tok_s']:7.0f} "
                    f"prefill {a['prefill_computed_tok_s']:7.0f} tok/s | hit {a['gpu_hit_rate'] or 0:.3f} "
                    f"ext {a['ext_hit_rate'] or 0:.3f} | run {a['running']:.0f} wait {a['waiting']:.0f} "
                    f"preempt/s {a['preempt_s']:.2f} | itl {a['itl_mean'] or 0:.3f}s",
                    flush=True,
                )
                prev = cur  # an engine that missed this scrape sits out the next window too
                prev_t = now
                if phase == "hold":
                    level_rows.append(row)
                elif (conc.get("inflight") or 0) <= level or now - begin > args.max_drain * 60:
                    phase, start = "hold", now
                    print(f"=== level {level}: hold starts after {(now - begin) / 60:.0f} min of drain", flush=True)
            if level_rows:
                summary = {
                    "kind": "level",
                    "t": time.time(),
                    "level": level,
                    "minutes": minutes,
                    "roles": summarize(level_rows),
                }
                out.write(json.dumps(summary) + "\n")
                out.flush()


asyncio.run(main())
print("=== schedule done", flush=True)
