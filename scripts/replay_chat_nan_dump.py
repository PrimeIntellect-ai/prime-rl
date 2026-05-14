#!/usr/bin/env python3
"""Replay dumped /v1/chat/completions requests from a PrimeRL NaN capture.

Exact setting this was written for:
  model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  workload: daniel/swe@0.3.4, MITO chat completions, LoRA enabled
  context: seq_len=max_model_len=65536, max_completion_tokens=16384
  failure: vLLM JSON serialization rejects a ChatCompletionResponse containing NaN

This script is standalone: point it at one dump JSON or a directory of dump
JSON files and at any OpenAI-compatible vLLM /v1 endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


def _json_default(value: Any) -> str:
    return repr(value)


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.json"))


def _request_entry(body: Any, *, source: str, request_id: str | None, ordinal: int) -> dict[str, Any]:
    return {
        "body": body,
        "source": source,
        "request_id": request_id,
        "ordinal": ordinal,
    }


def collect_requests(path: Path, mode: str, repeat: int) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for dump_path in _dump_paths(path):
        dump = _load_json(dump_path)
        if mode == "ring":
            for idx, snapshot in enumerate(dump.get("recent_requests") or []):
                body = snapshot.get("body")
                if body is not None:
                    requests.append(
                        _request_entry(
                            body,
                            source=f"{dump_path}#recent_requests[{idx}]",
                            request_id=snapshot.get("request_id"),
                            ordinal=len(requests),
                        )
                    )
        else:
            snapshot = dump.get("request") or {}
            body = snapshot.get("body")
            if body is not None:
                requests.append(
                    _request_entry(
                        body,
                        source=f"{dump_path}#request",
                        request_id=snapshot.get("request_id") or dump.get("request_id"),
                        ordinal=len(requests),
                    )
                )

    if mode == "single" and requests:
        requests = [requests[0]]
    if repeat > 1:
        base = list(requests)
        requests = []
        for _ in range(repeat):
            for request in base:
                copied = dict(request)
                copied["ordinal"] = len(requests)
                requests.append(copied)
    return requests


async def _post_one(
    client: httpx.AsyncClient,
    url: str,
    request: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    started_at = time.time()
    async with semaphore:
        try:
            response = await client.post(url, json=request["body"])
            text = response.text
            try:
                parsed = response.json()
            except Exception:
                parsed = None
            return {
                "ordinal": request["ordinal"],
                "request_id": request.get("request_id"),
                "source": request["source"],
                "status_code": response.status_code,
                "elapsed_s": time.time() - started_at,
                "contains_nan_text": "nan" in text.lower(),
                "text_prefix": text[:1000],
                "json": parsed,
            }
        except Exception as exc:
            return {
                "ordinal": request["ordinal"],
                "request_id": request.get("request_id"),
                "source": request["source"],
                "status_code": None,
                "elapsed_s": time.time() - started_at,
                "exception": repr(exc),
            }


async def replay(args: argparse.Namespace) -> None:
    requests = collect_requests(args.input, args.mode, args.repeat)
    if not requests:
        raise SystemExit(f"No request bodies found in {args.input}")

    headers: dict[str, str] = {}
    api_key = args.api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = _chat_url(args.base_url)
    timeout = httpx.Timeout(args.timeout_s, connect=30.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "event": "replay_start",
                "url": url,
                "input": str(args.input),
                "mode": args.mode,
                "requests": len(requests),
                "concurrency": args.concurrency,
                "repeat": args.repeat,
                "output": str(output) if output else None,
            },
            sort_keys=True,
        )
    )
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        tasks = [_post_one(client, url, request, semaphore) for request in requests]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            line = json.dumps(result, default=_json_default, sort_keys=True)
            print(line, flush=True)
            if output is not None:
                with output.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Dump JSON file or directory of dump JSON files.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL. /v1 is optional.",
    )
    parser.add_argument("--api-key", default=None, help="Bearer token. Defaults to VLLM_API_KEY or OPENAI_API_KEY.")
    parser.add_argument(
        "--mode",
        choices=("single", "failed", "ring"),
        default="single",
        help="single: first failing request; failed: all dump requests; ring: recent request ring from each dump.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat collected request list this many times.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests.")
    parser.add_argument("--timeout-s", type=float, default=900.0, help="Per-request timeout.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSONL result path.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(replay(parse_args()))
