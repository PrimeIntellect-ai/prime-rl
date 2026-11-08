"""Env worker pool: a ROUTER broker over N worker processes.

A lone `EnvWorker` runs every rollout as an `asyncio.Task` on one event loop, so
CPU-bound work (renderer tokenization, scoring) competes for that loop; the pool
spreads that work across processes.

A broker binds the client-facing ROUTER, starts one worker process and scales up to
`max_workers` on demand — each an ordinary `EnvWorker` bound to its own ipc address —
load-balancing requests to the least-busy worker over a `DEALER` per worker. The
worker's `client_id` (its reply identity) is the broker's DEALER identity; the broker
holds the real client identity in `pending` and routes the reply back by `request_id`.
`health` is answered inline (no worker needed); everything else goes to a worker.

Scaling is elastic but upscale-only: a new worker is spawned when in-flight requests
reach 90% of current capacity (`workers * multiplex`). Workers are spawned `spawn`-style
(own env, own loop) and monitor a death pipe so an orphaned worker self-exits if the
broker dies. TODO: downscale idle workers, per-worker restart-on-death, stats/lag
monitors (rollout errors are returned as data, not crashes, so worker death is rare).
"""

import asyncio
import contextlib
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
from collections.abc import Callable

import msgpack
import zmq
import zmq.asyncio
from verifiers.v1.utils.loaders import resolve_env_config

from prime_rl.configs.env_worker_pool import ElasticPoolConfig, PoolConfig, StaticPoolConfig
from prime_rl.env.types import CancelResponse, HealthResponse
from prime_rl.env.worker import LATE_CANCEL_SECONDS, EnvWorker
from prime_rl.utils.logger import get_logger
from prime_rl.utils.process import set_proc_title

_HEALTH = msgpack.packb(HealthResponse().model_dump(mode="json"), use_bin_type=True)
_CANCEL_MISS = msgpack.packb(CancelResponse(cancelled=False).model_dump(mode="json"), use_bin_type=True)


def _arm_teardown(death_pipe=None) -> None:
    """Arm a broker or worker process for clean
    teardown: it inherits no signal handlers, so by default SIGTERM kills it abruptly, skipping
    asyncio.run()'s serving() cleanup and orphaning host tunnels (and sandboxes).

    - SIGTERM -> KeyboardInterrupt so the event loop runs its finallys (run_env_worker_pool swallows it);
    - with `death_pipe`, a worker self-SIGTERMs when the broker dies (pipe EOF, even on
      SIGKILL) so the worker is not orphaned."""

    def on_sigterm(*_) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    if death_pipe is None:
        return

    def _watch() -> None:
        with contextlib.suppress(Exception):
            death_pipe.recv()
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_watch, daemon=True).start()


class EnvWorkerPool:
    """One source's pool replica: a ROUTER broker with least-busy worker dispatch.

    An elastic pool starts with a single worker and spawns another whenever in-flight
    requests reach 90% of current capacity (`workers * multiplex`), up to `max_workers`.
    Upscale-only for now — workers are never reclaimed. A static pool pre-spawns all
    `num_workers` upfront."""

    def __init__(
        self,
        pool: ElasticPoolConfig | StaticPoolConfig,
        address: str,
        config_data: dict,
        max_concurrent: int | None,
        worker_setup: Callable[[], None] | None,
    ) -> None:
        self.pool = pool
        self.elastic = isinstance(pool, ElasticPoolConfig)
        self.max_workers = pool.max_workers if self.elastic else pool.num_workers
        self.config_data = config_data
        self.max_concurrent = max_concurrent
        self.worker_setup = worker_setup
        self._ipc_dir = tempfile.mkdtemp(prefix="prl-env-pool-", dir="/tmp")
        self.workers: list[dict] = []
        self._mpctx = mp.get_context("spawn")
        self._poller: zmq.asyncio.Poller | None = None

        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(address)
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)

    def _worker_path(self, i: int) -> str:
        return f"{self._ipc_dir}/{i}"

    def _spawn_worker(self) -> None:
        i = len(self.workers)  # upscale-only, so the next index is the current count
        address = f"ipc://{self._worker_path(i)}"
        parent_conn, child_conn = self._mpctx.Pipe()
        proc = self._mpctx.Process(
            target=run_env_worker,
            kwargs=dict(
                address=address,
                config_data=self.config_data,
                max_concurrent=self.max_concurrent,
                death_pipe=child_conn,
                worker_setup=self.worker_setup,
            ),
            daemon=False,
        )
        proc.start()
        child_conn.close()  # parent keeps the write end (its close signals death)
        dealer = self.ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(address)  # connect before bind is fine — ZMQ queues
        self.workers.append(
            {
                "process": proc,
                "dealer": dealer,
                "pipe": parent_conn,
                "active": 0,
                "index": i,
            }
        )
        if self._poller is not None:
            self._poller.register(dealer, zmq.POLLIN)

    def _maybe_scale_up(self, in_flight: int) -> None:
        """Spawn one more worker when in-flight rollout slots reach 90% of capacity.

        A new worker starts at `active=0`, so least-busy dispatch funnels the backlog to
        it as it comes online (a few seconds to load the env) — fine, since we only scale
        up once already saturated. `max_workers=None` scales without a cap."""
        if self.max_workers is not None and len(self.workers) >= self.max_workers:
            return
        if in_flight >= 0.9 * len(self.workers) * self.pool.multiplex:
            self._spawn_worker()
            get_logger().info(
                f"Grew env worker pool to {len(self.workers)} worker(s) max_workers={self._cap_str} in_flight={in_flight}"
            )

    @property
    def _cap_str(self) -> str:
        return "inf" if self.max_workers is None else str(self.max_workers)

    async def run(self) -> None:
        self._poller = zmq.asyncio.Poller()
        self._poller.register(self.frontend, zmq.POLLIN)
        # Elastic: start with one and scale up on demand. Static: pre-spawn the lot.
        for _ in range(1 if self.elastic else self.max_workers):
            self._spawn_worker()
        # request_id -> {client_id, worker}
        pending: dict[bytes, dict] = {}
        # request_id -> (replied_at, worker), oldest first: a late cancel still reaches
        # the worker that may hold the run's kept arrays (`LATE_CANCEL_SECONDS`).
        replied: dict[bytes, tuple[float, dict]] = {}
        sizing = (
            f"max_workers={self._cap_str} multiplex={self.pool.multiplex}"
            if self.elastic
            else f"num_workers={self.max_workers}"
        )
        get_logger().info(
            f"Started env worker pool address={self.address} pool={self.pool.type} workers={len(self.workers)} {sizing}"
        )
        try:
            in_flight = 0
            while True:
                events = dict(await self._poller.poll())
                if self.frontend in events:
                    frames = await self.frontend.recv_multipart()
                    # Drop malformed requests without skipping ready worker replies.
                    if len(frames) != 4:
                        get_logger().warning(f"Dropped malformed request frames={len(frames)} expected=4")
                    else:
                        client_id, request_id, method, payload = frames
                        if method == b"health":
                            await self.frontend.send_multipart([client_id, request_id, b"reply", _HEALTH])
                        elif method == b"cancel":
                            # Route the cancel to the worker holding the target run,
                            # or to the one that replied to it within
                            # `LATE_CANCEL_SECONDS` (it may hold the run's kept
                            # arrays); any other target (or an unparseable payload
                            # — one bad frame must not tear down the broker) is
                            # answered inline
                            try:
                                body = msgpack.unpackb(payload, raw=False)
                                target = str(body.get("request_id", "")).encode()
                            except Exception:  # noqa: BLE001 - one bad frame must not kill the broker
                                target = b""
                            target_entry = pending.get(target)
                            late = replied.get(target)
                            if target_entry is not None:
                                worker = target_entry["worker"]
                            elif late is not None and time.monotonic() - late[0] < LATE_CANCEL_SECONDS:
                                worker = late[1]
                            else:
                                worker = None
                            if worker is None:
                                await self.frontend.send_multipart([client_id, request_id, b"reply", _CANCEL_MISS])
                            else:
                                worker["active"] += 1
                                pending[request_id] = {
                                    "client_id": client_id,
                                    "worker": worker,
                                }
                                in_flight += 1
                                await worker["dealer"].send_multipart([request_id, method, payload])
                        else:
                            worker = min(self.workers, key=lambda w: w["active"])
                            worker["active"] += 1
                            pending[request_id] = {
                                "client_id": client_id,
                                "worker": worker,
                            }
                            in_flight += 1
                            # forward without client_id — the DEALER identity is the worker's
                            # `client_id`; we hold the real one in `pending`.
                            await worker["dealer"].send_multipart([request_id, method, payload])
                            if self.elastic:
                                self._maybe_scale_up(in_flight)
                for w in self.workers:
                    if w["dealer"] in events:
                        request_id, kind, data = await w["dealer"].recv_multipart(copy=False)
                        # Copy only the routing key; relay the Frames unchanged. A
                        # `delta` is one of many for its request — only the `reply`
                        # closes the request out.
                        if kind.bytes == b"reply":
                            entry = pending.pop(request_id.bytes, None)
                            if entry is None:
                                continue
                            entry["worker"]["active"] -= 1
                            in_flight -= 1
                            now = time.monotonic()
                            replied[request_id.bytes] = (now, entry["worker"])
                            for old, (replied_at, _) in list(replied.items()):
                                if now - replied_at < LATE_CANCEL_SECONDS:
                                    break
                                del replied[old]
                        else:
                            entry = pending.get(request_id.bytes)
                            if entry is None:
                                continue
                        try:
                            await self.frontend.send_multipart([entry["client_id"], request_id, kind, data], copy=False)
                        except zmq.ZMQError:
                            # The client cannot take this episode. Free it at the worker;
                            # the empty request id leaves the cancel reply untracked.
                            with contextlib.suppress(UnicodeDecodeError):
                                await w["dealer"].send_multipart(
                                    [
                                        b"",
                                        b"cancel",
                                        msgpack.packb(
                                            {"request_id": request_id.bytes.decode()},
                                            use_bin_type=True,
                                        ),
                                    ]
                                )
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        for w in self.workers:
            with contextlib.suppress(Exception):
                w["pipe"].close()
            with contextlib.suppress(Exception):
                w["process"].terminate()
        for w in self.workers:
            with contextlib.suppress(Exception):
                w["process"].join(timeout=10)
            if w["process"].is_alive():
                with contextlib.suppress(Exception):
                    w["process"].kill()
            with contextlib.suppress(Exception):
                w["dealer"].close()
            with contextlib.suppress(OSError):
                os.unlink(self._worker_path(w["index"]))
        with contextlib.suppress(OSError):
            os.rmdir(self._ipc_dir)
        self.frontend.close()
        self.ctx.term()
        get_logger().info(f"Stopped env worker pool address={self.address}")


def run_env_worker(
    *,
    address: str,
    config_data: dict,
    max_concurrent: int | None,
    death_pipe,
    worker_setup: Callable[[], None] | None,
) -> None:
    """A pool worker process: one `EnvWorker` on its ipc address. It self-terminates if
    the broker dies — see `_arm_teardown`."""
    set_proc_title("EnvWorker")
    _arm_teardown(death_pipe)
    if worker_setup is not None:
        worker_setup()
    try:
        asyncio.run(EnvWorker(resolve_env_config(config_data), address, max_concurrent).run())
    except KeyboardInterrupt:
        pass
    except Exception:
        get_logger().opt(exception=True).error("Failed to run env worker")
        raise


def run_env_worker_pool(
    *,
    pool: PoolConfig,
    address: str,
    config_data: dict,
    max_concurrent: int | None,
    address_queue=None,
    worker_setup: Callable[[], None] | None = None,
) -> None:
    """Run one source's pool replica: a ZMQ broker over env worker processes. Reports
    the bound address on `address_queue` (for a spawner that passed an OS-assigned `:0`).

    `config_data` is the env config as plain data (`model_dump(mode="json")`), since a
    dynamically-narrowed config type can't be pickled into spawned workers.

    `worker_setup` (a picklable callable) runs in this process and every spawned worker —
    without it a spawned worker inherits no log handlers and its logs are silently dropped."""
    # Graceful SIGTERM (run asyncio teardown). The re-raised KeyboardInterrupt is swallowed
    # below for a clean exit (no spurious traceback).
    set_proc_title("EnvBroker")
    _arm_teardown()
    if worker_setup is not None:
        worker_setup()
    try:
        broker = EnvWorkerPool(pool, address, config_data, max_concurrent, worker_setup)
        if address_queue is not None:
            address_queue.put(broker.address)
        asyncio.run(broker.run())
    except KeyboardInterrupt:
        pass
    except Exception:
        get_logger().opt(exception=True).error("Failed to run env worker pool")
        raise
