"""Low-overhead Verifiers v1 chat harness for native GlyphBench episodes.

The generic ``null`` harness is deliberately portable: it launches a standalone
Python program for every segment.  That startup cost dominates GlyphBench's
small, fast environment steps at RL-scale concurrency.  This harness implements
the in-process transport explicitly supported by Verifiers while keeping its
interception boundary intact.  Every model call still reaches the rollout's
authenticated endpoint, so tracing, sampling overrides, token rendering, prefix
caching, limits, and training data collection remain owned by Verifiers/Prime-RL.
"""

from __future__ import annotations

import httpx
import verifiers.v1 as vf
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.types import Messages, SystemMessage, UserMessage


class GlyphBenchHarnessConfig(vf.HarnessConfig):
    """Configuration for GlyphBench's tool-free intercepted chat loop."""


class GlyphBenchHarness(vf.Harness[GlyphBenchHarnessConfig]):
    """Run tool-free chat turns in-process instead of spawning Python programs."""

    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_RESUME = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def session(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
        tool_interception_url: str | None = None,
    ) -> vf.HarnessSession:
        if mcp_urls:
            raise ValueError("GlyphBenchHarness is tool-free and does not support MCP")
        return GlyphBenchHarnessSession(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            tool_interception_url,
        )

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
    ) -> ProgramResult:
        """Support one-shot use while preserving the intercepted HTTP boundary."""

        if mcp_urls:
            raise ValueError("GlyphBenchHarness is tool-free and does not support MCP")
        async with _client() as client:
            messages = _conversation(self, trace, data, None)
            await _request(client, endpoint, secret, ctx.model, messages)
        return ProgramResult(0, "", "")


class GlyphBenchHarnessSession(vf.HarnessSession):
    """Retain one lightweight HTTP connection for a native game episode."""

    def __init__(
        self,
        harness: GlyphBenchHarness,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: vf.TaskData,
        tool_interception_url: str | None = None,
    ) -> None:
        super().__init__(
            harness,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            tool_interception_url,
        )
        self._client = _client()

    async def _run(self, messages: Messages | None) -> ProgramResult:
        conversation = _conversation(self.harness, self.trace, self.data, messages)
        await _request(
            self._client,
            self.endpoint,
            self.secret,
            self.ctx.model,
            conversation,
        )
        return ProgramResult(0, "", "")

    async def close(self) -> None:
        if self._closed:
            return
        await self._client.aclose()
        await super().close()


def _client() -> httpx.AsyncClient:
    # One request can be active per rollout. Retaining its connection across
    # native turns avoids both process startup and TCP churn.
    return httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=5.0),
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
    )


def _conversation(
    harness: GlyphBenchHarness,
    trace: vf.Trace,
    data: vf.TaskData,
    messages: Messages | None,
) -> Messages:
    branch = list(trace.branches[-1].messages) if trace.branches else []
    if messages is not None:
        if branch:
            return [*branch, *messages]
        system_prompt, _ = harness.resolve_prompt(data)
        return [
            *(
                [SystemMessage(content=system_prompt)]
                if system_prompt is not None
                else []
            ),
            *messages,
        ]

    system_prompt, prompt = harness.resolve_prompt(data)
    if isinstance(prompt, str):
        initial: Messages = [UserMessage(content=prompt)]
    elif prompt is None:
        initial = []
    else:
        initial = list(prompt)
    if system_prompt is not None:
        initial.insert(0, SystemMessage(content=system_prompt))
    return initial


async def _request(
    client: httpx.AsyncClient,
    endpoint: str,
    secret: str,
    model: str,
    messages: Messages,
) -> None:
    response = await client.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {secret}"},
        json={
            "model": model,
            "messages": [message_to_wire(message) for message in messages],
        },
    )
    response.raise_for_status()


__all__ = [
    "GlyphBenchHarness",
    "GlyphBenchHarnessConfig",
    "GlyphBenchHarnessSession",
]
