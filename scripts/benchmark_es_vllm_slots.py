import argparse
import asyncio
import json
import statistics
import time
import tomllib
from pathlib import Path

import httpx
import torch
from datasets import load_dataset

from prime_rl.configs.es import ESConfig
from prime_rl.trainer.es.lora_materialize import build_adapter_template


def serialize_specs(template) -> list[dict]:
    return [
        {
            "name": spec.name,
            "shape": list(spec.shape),
            "dtype": str(spec.dtype),
            "numel": spec.numel,
        }
        for spec in template.specs
    ]


def slot_definitions(count: int) -> list[dict]:
    return [{"lora_name": f"es_slot_{idx}", "lora_int_id": 10_000 + idx, "slot": idx} for idx in range(count)]


async def wait_ready(client: httpx.AsyncClient, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    last_error = None
    while time.perf_counter() < deadline:
        try:
            response = await client.get("/liveness", timeout=5)
            if response.status_code == 200:
                return
            last_error = response.text
        except Exception as exc:
            last_error = repr(exc)
        await asyncio.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout_s}s: {last_error}")


async def init_slots(client: httpx.AsyncClient, config: ESConfig, slots: list[dict], work_dir: Path) -> None:
    template = build_adapter_template(work_dir, config.model, device=torch.device("cpu"))
    theta_path = work_dir / "bench_theta_init.pt"
    torch.save({"theta": template.theta.detach().cpu()}, theta_path)
    response = await client.post(
        "/es/init_lora_slots",
        json={
            "theta_path": theta_path.resolve().as_posix(),
            "specs": serialize_specs(template),
            "adapter_config": template.adapter_config,
            "slots": slots,
        },
        timeout=None,
    )
    response.raise_for_status()
    response = await client.post(
        "/es/materialize_lora_slots",
        json={"slots": [{**slot, "candidate_idx": slot["slot"], "seed": slot["slot"], "sign": 1} for slot in slots], "sigma": 0.0},
        timeout=None,
    )
    response.raise_for_status()
    theta_path.unlink(missing_ok=True)


def gsm8k_messages(count: int) -> list[list[dict[str, str]]]:
    dataset = load_dataset("openai/gsm8k", "main", split=f"train[:{count}]")
    return [[{"role": "user", "content": row["question"]}] for row in dataset]


async def run_batch(
    client: httpx.AsyncClient,
    messages: list[list[dict[str, str]]],
    *,
    candidates: int,
    concurrency: int,
    max_tokens: int,
    top_k: int | None,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    completion_tokens = 0
    prompt_tokens = 0
    failures = 0

    async def one_request(idx: int, message: list[dict[str, str]]) -> None:
        nonlocal completion_tokens, prompt_tokens, failures
        candidate_idx = idx % candidates
        payload = {
            "model": f"es_slot_{candidate_idx}",
            "messages": message,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "logprobs": False,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        async with semaphore:
            t0 = time.perf_counter()
            try:
                response = await client.post("/v1/chat/completions", json=payload, timeout=None)
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                completion_tokens += int(usage.get("completion_tokens") or 0)
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
            except Exception:
                failures += 1
            finally:
                latencies.append(time.perf_counter() - t0)

    started = time.perf_counter()
    await asyncio.gather(*(one_request(idx, message) for idx, message in enumerate(messages)))
    elapsed = time.perf_counter() - started
    latencies_sorted = sorted(latencies)
    return {
        "requests": len(messages),
        "failures": failures,
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tps": completion_tokens / elapsed if elapsed > 0 else 0.0,
        "total_tps": (completion_tokens + prompt_tokens) / elapsed if elapsed > 0 else 0.0,
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "latency_p50_s": latencies_sorted[int(0.50 * (len(latencies_sorted) - 1))] if latencies_sorted else 0.0,
        "latency_p95_s": latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))] if latencies_sorted else 0.0,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--es-config", type=Path, default=Path("configs/gsm8k/es.toml"))
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/gsm8k/es_qwen3_0_6b/bench"))
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--examples-per-candidate", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--skip-init-slots", action="store_true")
    parser.add_argument("--ready-timeout-s", type=float, default=240)
    args = parser.parse_args()

    with open(args.es_config, "rb") as f:
        config = ESConfig(**tomllib.load(f))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    slots = slot_definitions(args.candidates)
    messages = gsm8k_messages(args.candidates * args.examples_per_candidate)

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        await wait_ready(client, args.ready_timeout_s)
        if not args.skip_init_slots:
            await init_slots(client, config, slots, args.work_dir)
        warmup_messages = messages[: min(32, len(messages))]
        await run_batch(
            client,
            warmup_messages,
            candidates=args.candidates,
            concurrency=min(args.concurrency, len(warmup_messages)),
            max_tokens=min(args.max_tokens, 256),
            top_k=args.top_k,
        )
        result = await run_batch(
            client,
            messages,
            candidates=args.candidates,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
