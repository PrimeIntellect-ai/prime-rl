"""Background async runner for fire-and-forget HTTP work.

Owns an event loop running on a background thread plus an httpx.AsyncClient,
so callers can submit coroutines without blocking the main thread.
Fork-safe via ``os.register_at_fork``.
"""

import asyncio
import os
from threading import Thread
from typing import Awaitable, Callable

import httpx

from prime_rl.utils.logger import get_logger


class BackgroundAsync:
    """Owns a background event loop + httpx client for fire-and-forget work."""

    def __init__(self, *, client_timeout: float = 30.0):
        self.logger = get_logger()
        self._client_timeout = client_timeout
        self._closed = False
        self._init_runtime()
        os.register_at_fork(after_in_child=self._reinit_after_fork)

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    def submit(self, coro_factory: Callable[[], Awaitable[None]]) -> None:
        """Schedule a coroutine on the background loop. Fire-and-forget.

        Takes a factory rather than a coroutine so the coroutine is created
        on the background thread (avoids "coroutine was never awaited" warnings
        if the loop is in the middle of restarting).
        """
        if self._closed:
            return
        future = asyncio.run_coroutine_threadsafe(coro_factory(), self._loop)
        self._pending_futures.append(future)
        # Drop completed futures to avoid unbounded growth
        self._pending_futures = [f for f in self._pending_futures if not f.done()]

    def close(self, *, flush_timeout: float = 15.0) -> None:
        if self._closed or not hasattr(self, "_loop"):
            return
        self._closed = True
        self._flush(timeout=flush_timeout)

        async def _close_client() -> None:
            await self._client.aclose()

        try:
            future = asyncio.run_coroutine_threadsafe(_close_client(), self._loop)
            future.result(timeout=5.0)
        except Exception as e:
            self.logger.debug("Error closing background async client: %s", e)

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def __del__(self) -> None:
        self.close()

    # -- internals --

    def _init_runtime(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._client = httpx.AsyncClient(timeout=self._client_timeout)
        self._pending_futures: list[asyncio.Future] = []

    def _reinit_after_fork(self) -> None:
        # Background thread doesn't survive fork — recreate everything.
        self._closed = False
        self._init_runtime()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _flush(self, timeout: float) -> None:
        if not self._pending_futures:
            return
        self.logger.debug("Flushing %d pending background request(s)", len(self._pending_futures))
        for future in self._pending_futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                self.logger.debug("Pending background request completed with error: %s", e)
        self._pending_futures.clear()
