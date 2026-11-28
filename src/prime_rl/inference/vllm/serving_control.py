"""Admission and health of one API server process, for operations that must change what its engine serves without
dropping requests (``/swap-checkpoint``).

vLLM's own ``wait`` and ``abort`` pauses reject the requests that arrive while they last, and a data-parallel pause
cannot be cancelled before it completes. Holding requests here instead lets an operation stop new work from reaching
the engine, wait for or abort the requests in flight, and give up again at any point. A held request is not rejected:
it enters the engine when admission reopens.

Every count here is of this process's requests, so they cover an engine only when it has one API server (as every
rank of an external-LB deployment does).
"""

import asyncio

from vllm.engine.protocol import EngineClient


class Admission:
    """Admits new requests into the engine, or holds them while closed."""

    def __init__(self, client: EngineClient):
        self.client = client
        self._open = asyncio.Event()
        self._open.set()
        self.held = 0
        """Requests waiting for admission to reopen."""
        self._entering = 0
        """Requests admitted that the engine client has not registered yet."""
        add_request = client.add_request

        async def admitted_add_request(*args, **kwargs):
            if not self._open.is_set():
                self.held += 1
                try:
                    await self._open.wait()
                finally:
                    self.held -= 1
            self._entering += 1
            try:
                return await add_request(*args, **kwargs)
            finally:
                self._entering -= 1

        # Every generation path (chat, completions, /inference/v1/generate, pooling) adds its request through this
        client.add_request = admitted_add_request

    @property
    def is_open(self) -> bool:
        return self._open.is_set()

    def in_flight(self) -> int:
        """Requests this process's engine client runs or is about to: admitted and not finished."""
        return self._entering + self.client.output_processor.get_num_unfinished_requests()

    async def close(self) -> int:
        """Hold new requests; returns the requests in flight once those already admitted are registered."""
        self._open.clear()
        while self._entering:
            await asyncio.sleep(0.001)
        return self.in_flight()

    def open(self) -> None:
        self._open.set()

    async def abort(self) -> int:
        """Abort every request in flight; each finishes with the ``abort`` finish reason. Returns how many."""
        request_ids = list(self.client.output_processor.request_states)
        if request_ids:
            await self.client.abort(request_ids, internal=True)
        return len(request_ids)


class Health:
    """Why this server must not receive requests (``/health`` answers 503 with it), or None while it may."""

    def __init__(self):
        self.unhealthy_reason: str | None = None

    def set(self, healthy: bool, reason: str | None = None) -> None:
        self.unhealthy_reason = None if healthy else (reason or "marked unhealthy")


_admission: Admission | None = None
health = Health()
"""This process's health override."""


def install_admission(client: EngineClient) -> Admission:
    """Route this process's new requests through an admission gate (once, at app setup)."""
    global _admission
    _admission = Admission(client)
    return _admission


def admission() -> Admission:
    if _admission is None:
        raise RuntimeError("This API server holds no admission gate")
    return _admission
