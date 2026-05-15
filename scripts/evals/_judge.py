"""Claude LLM-judge wrapper for evals that need semantic scoring.

Judge is frozen at `claude-sonnet-4-6` per configs/evals/rung6_suite.toml and
§10a of the SFT plan doc. Any change = explicit doc amendment.

Uses the Anthropic SDK directly rather than going through openai-compat so we
keep the exact model ID + provider explicit (matches the precommitment).
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic, AsyncAnthropic

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_CONCURRENCY = 16


@dataclass(frozen=True)
class JudgeCall:
    prompt: str
    system: str | None
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class JudgeResponse:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


_INT_RE = re.compile(r"-?\d+")


def _parse_int(text: str, *, lo: int, hi: int) -> int | None:
    for match in _INT_RE.finditer(text):
        value = int(match.group())
        if lo <= value <= hi:
            return value
    return None


class Judge:
    """Wrapper around the Anthropic client with a frozen model id.

    Exposes:
      - `call(call)` / `score_int(call, lo, hi)`: synchronous single-call
      - `call_batch(calls, max_concurrency)` / `score_int_batch(calls, lo, hi)`:
        parallel via AsyncAnthropic + asyncio.Semaphore
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        self.model = model
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set; cannot run LLM-judge evals. Set it in your environment or pass api_key=."
            )
        self.client = Anthropic(api_key=key, max_retries=max_retries)
        self.async_client = AsyncAnthropic(api_key=key, max_retries=max_retries)

    def call(self, call: JudgeCall) -> JudgeResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": call.max_tokens,
            "temperature": call.temperature,
            "messages": [{"role": "user", "content": call.prompt}],
        }
        if call.system is not None:
            kwargs["system"] = call.system
        response = self.client.messages.create(**kwargs)
        text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return JudgeResponse(
            text="".join(text_parts),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )

    def score_int(self, call: JudgeCall, *, lo: int, hi: int) -> int | None:
        return _parse_int(self.call(call).text, lo=lo, hi=hi)

    async def _call_one_async(self, call: JudgeCall, sem: asyncio.Semaphore) -> JudgeResponse | Exception:
        async with sem:
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": call.max_tokens,
                    "temperature": call.temperature,
                    "messages": [{"role": "user", "content": call.prompt}],
                }
                if call.system is not None:
                    kwargs["system"] = call.system
                resp = await self.async_client.messages.create(**kwargs)
                text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
                return JudgeResponse(
                    text="".join(text_parts),
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    stop_reason=resp.stop_reason,
                )
            except Exception as exc:
                return exc

    def call_batch(
        self,
        calls: list[JudgeCall],
        *,
        max_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    ) -> list[JudgeResponse | None]:
        """Fire `calls` concurrently via AsyncAnthropic. Failures → None at that slot."""
        if not calls:
            return []
        sem = asyncio.Semaphore(max_concurrency)

        async def _run():
            tasks = [self._call_one_async(c, sem) for c in calls]
            return await asyncio.gather(*tasks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("call_batch called from inside a running event loop")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        results = loop.run_until_complete(_run())
        out: list[JudgeResponse | None] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"[judge] call {i} failed: {r}", flush=True)
                out.append(None)
            else:
                out.append(r)
        return out

    def score_int_batch(
        self,
        calls: list[JudgeCall],
        *,
        lo: int,
        hi: int,
        max_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    ) -> list[int | None]:
        responses = self.call_batch(calls, max_concurrency=max_concurrency)
        return [_parse_int(r.text, lo=lo, hi=hi) if r is not None else None for r in responses]
