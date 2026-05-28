#!/usr/bin/env python3
"""Replay Prime-RL chat replay diagnostic dumps.

Modes:
  chat    - replay the exact raw /v1/chat/completions request.
  ring    - replay recent raw chat requests from the API-server ring.
  engine  - replay the rendered prompt token IDs and sampling params through
            /inference/v1/generate, bypassing chat rendering.
  cohort  - replay active engine requests captured at dump time concurrently.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import httpx


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.json"))


def _base(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _generate_url(base_url: str) -> str:
    return f"{_base(base_url)}/inference/v1/generate"


def _find_nonfinite(value: Any, path: str = "$", *, limit: int = 8) -> tuple[int, list[str]]:
    if isinstance(value, float):
        if not math.isfinite(value):
            return 1, [path]
        return 0, []
    if isinstance(value, dict):
        total = 0
        paths: list[str] = []
        for key, child in value.items():
            count, child_paths = _find_nonfinite(child, f"{path}.{key}", limit=limit)
            total += count
            if len(paths) < limit:
                paths.extend(child_paths[: limit - len(paths)])
        return total, paths
    if isinstance(value, list):
        total = 0
        paths = []
        for idx, child in enumerate(value):
            count, child_paths = _find_nonfinite(child, f"{path}[{idx}]", limit=limit)
            total += count
            if len(paths) < limit:
                paths.extend(child_paths[: limit - len(paths)])
        return total, paths
    return 0, []


def _response_summary(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    nonfinite_count, nonfinite_paths = _find_nonfinite(parsed)
    choices = parsed.get("choices") or []
    first_choice = choices[0] if choices else {}
    usage = parsed.get("usage") or {}
    return {
        "nonfinite_count": nonfinite_count,
        "first_nonfinite_paths": nonfinite_paths,
        "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _request_body_from_snapshot(snapshot: dict[str, Any]) -> Any | None:
    body = snapshot.get("body") or {}
    body_json = body.get("json")
    if isinstance(body_json, dict) and "json_error" not in body_json:
        return body_json
    raw_text = body.get("raw_text")
    if isinstance(raw_text, str):
        return json.loads(raw_text)
    return None


def _engine_body_from_generation(generation: dict[str, Any]) -> Any | None:
    body = generation.get("generate_replay_body")
    if not isinstance(body, dict):
        return None
    if not body.get("token_ids"):
        return None
    return body


def _entry(body: Any, *, url: str, source: str, started_at: float | None = None) -> dict[str, Any]:
    return {
        "body": body,
        "url": url,
        "source": source,
        "started_at": started_at,
    }


def collect_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for dump_path in _dump_paths(args.input):
        dump = _load_json(dump_path)
        if args.mode == "chat":
            body = _request_body_from_snapshot(dump.get("request") or {})
            if body is not None:
                entries.append(_entry(body, url=_chat_url(args.base_url), source=f"{dump_path}#request"))
        elif args.mode == "ring":
            for idx, snapshot in enumerate(dump.get("recent_requests") or []):
                body = _request_body_from_snapshot(snapshot)
                if body is not None:
                    entries.append(
                        _entry(
                            body,
                            url=_chat_url(args.base_url),
                            source=f"{dump_path}#recent_requests[{idx}]",
                            started_at=snapshot.get("created_at"),
                        )
                    )
        elif args.mode == "engine":
            body = _engine_body_from_generation(dump.get("generation") or {})
            if body is not None:
                entries.append(_entry(body, url=_generate_url(args.base_url), source=f"{dump_path}#generation"))
        elif args.mode == "cohort":
            generations = dump.get("active_generations_at_dump") or []
            if args.include_recent_generations:
                generations = generations + (dump.get("recent_generations") or [])
            for idx, generation in enumerate(generations):
                body = _engine_body_from_generation(generation)
                if body is not None:
                    entries.append(
                        _entry(
                            body,
                            url=_generate_url(args.base_url),
                            source=f"{dump_path}#generation_cohort[{idx}]",
                            started_at=generation.get("created_at"),
                        )
                    )
        else:
            raise AssertionError(args.mode)

    if args.limit is not None:
        entries = entries[: args.limit]
    if args.repeat > 1:
        base = list(entries)
        entries = []
        for _ in range(args.repeat):
            entries.extend(dict(entry) for entry in base)
    return entries


async def _post_one(
    client: httpx.AsyncClient,
    entry: dict[str, Any],
    semaphore: asyncio.Semaphore,
    *,
    delay_s: float,
    ordinal: int,
    summary_only: bool,
) -> dict[str, Any]:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    started = time.time()
    async with semaphore:
        try:
            response = await client.post(entry["url"], json=entry["body"])
            text = response.text
            try:
                parsed = response.json()
            except Exception:
                parsed = None
            return {
                "ordinal": ordinal,
                "source": entry["source"],
                "url": entry["url"],
                "status_code": response.status_code,
                "elapsed_s": time.time() - started,
                "contains_nan_text": "nan" in text.lower(),
                "response_summary": _response_summary(parsed),
                "text_prefix": text[:1000] if response.status_code >= 400 else None,
                **({} if summary_only else {"json": parsed}),
            }
        except Exception as exc:
            return {
                "ordinal": ordinal,
                "source": entry["source"],
                "url": entry["url"],
                "status_code": None,
                "elapsed_s": time.time() - started,
                "exception": repr(exc),
            }


async def replay(args: argparse.Namespace) -> None:
    entries = collect_entries(args)
    if not entries:
        raise SystemExit(f"No replayable entries found in {args.input} for mode={args.mode}")

    headers: dict[str, str] = {}
    api_key = args.api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    started_values = [float(entry["started_at"]) for entry in entries if entry.get("started_at") is not None]
    first_started_at = min(started_values) if started_values else None
    timeout = httpx.Timeout(args.timeout_s, connect=30.0)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        tasks = []
        for ordinal, entry in enumerate(entries):
            delay_s = 0.0
            if args.preserve_timing and entry.get("started_at") is not None and first_started_at is not None:
                delay_s = max(0.0, float(entry["started_at"]) - float(first_started_at))
            tasks.append(
                asyncio.create_task(
                    _post_one(
                        client,
                        entry,
                        semaphore,
                        delay_s=delay_s,
                        ordinal=ordinal,
                        summary_only=args.summary_only,
                    )
                )
            )
        for result in await asyncio.gather(*tasks):
            print(json.dumps(result, indent=None if args.jsonl else 2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Dump JSON file or directory of dumps")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible server base URL")
    parser.add_argument("--mode", choices=["chat", "ring", "engine", "cohort"], default="engine")
    parser.add_argument("--api-key")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--preserve-timing", action="store_true")
    parser.add_argument("--include-recent-generations", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--jsonl", action="store_true")
    return parser.parse_args()


def main() -> None:
    asyncio.run(replay(parse_args()))


if __name__ == "__main__":
    main()
