"""Reproduce a burst of router session releases permanently filling the orchestrator's httpx pool.

Copies the release path from `prime_rl/orchestrator/clients.py` (`old`: the shared 4-connection
admin client; `new`: the dedicated client behind a semaphore) and runs it against a local
`/finish_session` server through the dispatcher's create_task + shield + CancelledError pattern.
Each burst is followed by single releases against a fast healthy server, then a fresh client.

uv run --no-project --with httpx==0.28.1 --with httpcore==1.0.9 --with anyio==4.13.0 \
    --with tenacity==9.1.4 --with starlette --with uvicorn \
    python tools/repro_session_release_pool_leak.py old
"""

import argparse
import asyncio
import multiprocessing
import random
import socket
import time
from collections import Counter

import httpx
from httpx import AsyncClient
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

SESSION_RELEASE_CONCURRENCY = 32
SESSION_RELEASE_QUEUE_TIMEOUT_S = 30.0


def serve(port: int) -> None:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    state = {"delay": 0.0}

    async def finish_session(request):
        await asyncio.sleep(state["delay"])
        return JSONResponse({"ok": True})

    async def set_delay(request):
        state["delay"] = float(request.query_params["delay"])
        return JSONResponse(state)

    app = Starlette(routes=[Route("/finish_session", finish_session, methods=["POST"]), Route("/delay", set_delay)])
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", backlog=8192)


def _is_retryable_admin_error(exception: BaseException) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code >= 500
    return isinstance(exception, (httpx.TimeoutException, httpx.TransportError))


async def _admin_post(client: AsyncClient, path: str, *, timeout_s: float, **kwargs) -> None:
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_admin_error),
        stop=stop_after_delay(2 * timeout_s) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    ):
        with attempt:
            response = await client.post(
                path,
                timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=10.0),
                **kwargs,
            )
            response.raise_for_status()


class Releaser:
    def __init__(self, base_url: str, variant: str, undersized_pool: bool):
        self.base_url = base_url
        self.variant = variant
        self.undersized_pool = undersized_pool
        self.client = self.make_client()
        self.slots = asyncio.Semaphore(SESSION_RELEASE_CONCURRENCY)
        self.replacements = 0
        self.outcomes: Counter[str] = Counter()

    def make_client(self) -> AsyncClient:
        if self.variant == "old" or self.undersized_pool:
            limits = httpx.Limits(max_connections=4, max_keepalive_connections=1)
        else:
            limits = httpx.Limits(
                max_connections=SESSION_RELEASE_CONCURRENCY, max_keepalive_connections=SESSION_RELEASE_CONCURRENCY
            )
        return AsyncClient(base_url=self.base_url, limits=limits, timeout=httpx.Timeout(None))

    async def finish_session_old(self, session_id: str) -> Exception | None:
        try:
            await _admin_post(self.client, "/finish_session", timeout_s=5.0, params={"session_id": session_id})
        except Exception as error:
            return error
        return None

    async def finish_session_new(self, session_id: str) -> Exception | None:
        try:
            await asyncio.wait_for(self.slots.acquire(), SESSION_RELEASE_QUEUE_TIMEOUT_S)
        except TimeoutError as error:
            return error
        client = self.client
        try:
            await _admin_post(client, "/finish_session", timeout_s=5.0, params={"session_id": session_id})
        except httpx.PoolTimeout as error:
            if client is self.client:
                self.client = self.make_client()
                self.replacements += 1
                await client.aclose()
            return error
        except Exception as error:
            return error
        finally:
            self.slots.release()
        return None

    async def finish_sessions(self, session_ids: list[str]) -> None:
        finish_session = self.finish_session_old if self.variant == "old" else self.finish_session_new
        errors = await asyncio.gather(*(finish_session(session_id) for session_id in session_ids))
        self.outcomes.update("ok" if error is None else type(error).__name__ for error in errors)

    def pool_summary(self) -> str:
        pool = self.client._transport._pool
        infos = [connection.info() for connection in pool.connections]
        states = Counter(info.split(", ")[2] if ", " in info else info for info in infos)
        return f"connections {dict(states)}, orphaned queued requests {len(pool._requests)}"


async def run_episode(releaser: Releaser, episode: int) -> None:
    try:
        await asyncio.sleep(random.uniform(0, 0.5))
    finally:
        cleanup = asyncio.create_task(releaser.finish_sessions([f"trace-{episode}", f"trace-{episode}:agent"]))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise


async def probe(releaser: Releaser, label: str, n: int = 3) -> None:
    before = releaser.outcomes.copy()
    for k in range(n):
        await releaser.finish_sessions([f"probe-{label}-{k}"])
        await asyncio.sleep(0.2)
    delta = releaser.outcomes - before
    print(f"  {label}: {delta['ok']}/{n} single releases succeeded")


async def main(args: argparse.Namespace, base_url: str) -> None:
    async def set_server_delay(delay: float) -> None:
        async with AsyncClient(base_url=base_url) as control:
            await control.get("/delay", params={"delay": delay})

    releaser = Releaser(base_url, args.variant, args.undersized_pool)
    await set_server_delay(0.0)
    await probe(releaser, "before any burst")
    for burst in range(args.bursts):
        await set_server_delay(args.server_delay)
        releaser.outcomes.clear()
        start = time.monotonic()
        tasks = [asyncio.create_task(run_episode(releaser, i)) for i in range(args.episodes)]
        await asyncio.sleep(0.25)
        for task in random.sample(tasks, int(args.cancel_fraction * args.episodes)):
            task.cancel()
        await asyncio.sleep(3)
        for task in random.sample(tasks, int(args.cancel_fraction * args.episodes)):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        total = sum(releaser.outcomes.values())
        print(
            f"burst {burst}: {releaser.outcomes['ok']}/{total} releases delivered in {time.monotonic() - start:.1f}s, "
            f"failures {dict(releaser.outcomes - Counter(ok=releaser.outcomes['ok']))}"
        )
        await set_server_delay(0.0)
        await asyncio.sleep(2)
        await probe(releaser, "after burst, same client")
        print(f"  pool: {releaser.pool_summary()}")
    releaser.client = releaser.make_client()
    await probe(releaser, "fresh client")
    if args.variant == "new":
        print(f"client replacements: {releaser.replacements}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", choices=["old", "new"])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--bursts", type=int, default=3)
    parser.add_argument("--server-delay", type=float, default=2.0, help="seconds per release during a burst")
    parser.add_argument("--cancel-fraction", type=float, default=0.0, help="episodes cancelled, twice, per burst")
    parser.add_argument(
        "--undersized-pool", action="store_true", help="new: 4 connections for 32 slots, to exercise replacement"
    )
    args = parser.parse_args()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = multiprocessing.Process(target=serve, args=(port,), daemon=True)
    server.start()
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{base_url}/delay", params={"delay": 0})
            break
        except httpx.TransportError:
            time.sleep(0.1)
    try:
        asyncio.run(main(args, base_url))
    finally:
        server.terminate()
