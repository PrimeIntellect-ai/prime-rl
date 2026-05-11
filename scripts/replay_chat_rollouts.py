#!/usr/bin/env python3
"""Replay saved MITO rollout prompts against a prime-rl/vLLM inference server.

Default settings target the local Nemotron Super math NaN repro artifacts:

  rollouts: /beegfs/daniel/nemotron-super-math-minimal/run_default/rollouts/step_2/train_rollouts.jsonl
  endpoint: http://localhost:8000/v1/chat/completions
  model:    nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
  mode:     MITO chat completions only; no trainer, orchestrator, env, or weight broadcast.

Start an inference server separately, then run this script from the prime-rl
checkout with `uv run python scripts/replay_chat_rollouts.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx


DEFAULT_ROLLOUTS = Path(
    "/beegfs/daniel/nemotron-super-math-minimal/run_default/rollouts/step_2/train_rollouts.jsonl"
)
DEFAULT_OUTPUT = Path("/tmp/nemotron-chat-rollout-replay.jsonl")
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default=os.environ.get("REPLAY_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("REPLAY_MODEL", DEFAULT_MODEL))
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--api-key-var", default="VLLM_API_KEY")
    return parser.parse_args()


def load_rollouts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_repeated(rows: list[dict[str, Any]], repeat: int) -> Iterable[tuple[int, dict[str, Any]]]:
    for repeat_idx in range(repeat):
        for row in rows:
            yield repeat_idx, row


def chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def build_request(row: dict[str, Any], model: str) -> dict[str, Any]:
    sampling_args = copy.deepcopy(row.get("sampling_args") or {})
    extra_body = sampling_args.pop("extra_body", None) or {}

    body: dict[str, Any] = {
        "model": model,
        "messages": row["prompt"],
        "stream": False,
    }
    body.update(sampling_args)
    body.update(extra_body)
    return body


def request_headers(row: dict[str, Any], api_key_var: str) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        # Match orchestrator.client.extra_headers_from_state for this run.
        "X-Session-ID": str(row.get("example_id", "")),
    }
    api_key = os.environ.get(api_key_var)
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def find_nonfinite(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out.extend(find_nonfinite(item, f"{path}.{key}"))
        return out
    if isinstance(value, list):
        out = []
        for idx, item in enumerate(value):
            out.extend(find_nonfinite(item, f"{path}[{idx}]"))
        return out
    return []


def sanitize_nonfinite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: sanitize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_nonfinite(item) for item in value]
    return value


async def replay_one(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    api_key_var: str,
    repeat_idx: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    request = build_request(row, model)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "repeat": repeat_idx,
        "line_no": row.get("_line_no"),
        "example_id": row.get("example_id"),
        "env_name": row.get("env_name"),
        "request": request,
    }
    try:
        response = await client.post(url, headers=request_headers(row, api_key_var), json=request)
        result["status_code"] = response.status_code
        result["ok"] = response.is_success
        text = response.text
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
            result["response_text"] = text[:4000]

        if payload is not None:
            nonfinite_paths = find_nonfinite(payload)
            result["nonfinite_paths"] = nonfinite_paths
            payload = sanitize_nonfinite(payload)
            if response.is_success:
                choice = (payload.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                result["finish_reason"] = choice.get("finish_reason")
                result["completion_chars"] = len(message.get("content") or "")
                result["usage"] = payload.get("usage")
            else:
                result["error_payload"] = payload
    except Exception as exc:
        result["ok"] = False
        result["exception"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        result["duration_s"] = time.perf_counter() - started
    return result


async def run() -> int:
    args = parse_args()
    rows = load_rollouts(args.rollouts, args.limit)
    requests = list(iter_repeated(rows, args.repeat))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    url = chat_completions_url(args.base_url)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)

    print(f"replaying {len(requests)} requests from {args.rollouts}")
    print(f"endpoint={url} model={args.model} concurrency={args.concurrency} output={args.output}")

    failures = 0
    nonfinite = 0
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client, args.output.open("w") as out:

        async def guarded(repeat_idx: int, row: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await replay_one(client, url, args.model, args.api_key_var, repeat_idx, row)

        tasks = [asyncio.create_task(guarded(repeat_idx, row)) for repeat_idx, row in requests]
        for idx, task in enumerate(asyncio.as_completed(tasks), start=1):
            result = await task
            if not result.get("ok"):
                failures += 1
            if result.get("nonfinite_paths"):
                nonfinite += 1
            out.write(json.dumps(result, allow_nan=False) + "\n")
            out.flush()
            if idx == 1 or idx % 16 == 0 or idx == len(tasks):
                print(f"completed={idx}/{len(tasks)} failures={failures} nonfinite_json={nonfinite}")

    print(f"done failures={failures} nonfinite_json={nonfinite} output={args.output}")
    return 1 if failures or nonfinite else 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
