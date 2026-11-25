from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING, Protocol

import msgpack
import zmq
import zmq.asyncio
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.delta import DeltaStreamer
from verifiers.v1.episode import Episode
from verifiers.v1.task import Task
from verifiers.v1.utils.loaders import load_environment

from prime_rl.env.algorithm import ScoreClient
from prime_rl.env.encoding import pack
from prime_rl.env.heads import episode_head
from prime_rl.env.types import (
    BaseResponse,
    CancelRequest,
    CancelResponse,
    HealthResponse,
    RunRequest,
    RunResponse,
    SampleToScore,
    ScoreRequest,
)
from prime_rl.monitors.file.traces.index import index_record
from prime_rl.orchestrator.algo.base import iter_trainable_traces
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.configs.algorithm import ActionLossType
    from prime_rl.orchestrator.algo import Algorithm


class EpisodeHook(Protocol):
    """Takes finished train episodes' training samples (per-token columns, routing rows,
    sampling masks, top-k heads) before the episodes leave the worker."""

    def keep(self, episode: Episode, action_loss_type: ActionLossType) -> None:
        """Keep the episode's samples, their action tokens routed into `action_loss_type`;
        may leave references to them in `trace.info`. Runs off the event loop, before the
        episode's final flush."""

    def discard(self, episode: Episode) -> None:
        """Free what `keep` kept for an episode that was not delivered."""


_episode_hook: EpisodeHook | None = None
_algorithm_worker: str | None = None

LATE_CANCEL_SECONDS = 10.0
"""How long a delivered episode stays discardable: a client whose caller cancels the run
after the reply left drops the reply and sends a cancel right away."""

_SHUTDOWN_DRAIN_SECONDS = 5.0
"""Leave time for shared-resource teardown before the pool's 10-second kill deadline."""


def set_algorithm_worker(address: str | None) -> None:
    """The address of the replica's algorithm worker, which scores the samples the episode
    hook keeps (`prime_rl.env.algorithm`)."""
    global _algorithm_worker
    _algorithm_worker = address


def set_episode_hook(hook: EpisodeHook | None) -> None:
    """Install a hook that keeps finished train episodes' training samples. Streamed nodes
    never carry their training arrays, so the hook is the only way they leave the worker. Every episode it
    keeps is either delivered in its run's reply or handed back to `discard`, and so is
    a delivered one whose client cancels the run within `LATE_CANCEL_SECONDS`."""
    global _episode_hook
    _episode_hook = hook


def progress_delta(delta: dict) -> dict | None:
    """What a trace delta tells about the trace's progress: its phase, its model calls'
    usage and sub-agent sessions, its errors; None if nothing."""
    progress: dict = {"trace": delta["trace"]}
    if delta.get("discard"):
        progress["discard"] = True
    if "open" in delta:
        progress["open"] = {}
    if calls := delta.get("calls"):
        progress["calls"] = [
            {
                "usage": call.get("usage"),
                **({"acp": {"agent_session_id": acp.get("agent_session_id")}} if (acp := call.get("acp")) else {}),
            }
            for call in calls
        ]
    if changed := {key: value for key, value in (delta.get("set") or {}).items() if key in ("errors", "timing")}:
        progress["set"] = changed
    return progress if len(progress) > 1 else None


class _Keeping:
    """One episode's `keep`, run off the event loop. A failed or abandoned `keep` hands
    the episode to `discard` exactly once, and only after `keep` has returned: in the
    thread when `keep` finishes last, else in `abandon`."""

    def __init__(self, hook: EpisodeHook, episode: Episode) -> None:
        self.hook = hook
        self.episode = episode
        self._lock = threading.Lock()
        self._kept = False
        self._abandoned = False

    def run(self, action_loss_type: ActionLossType) -> None:
        try:
            self.hook.keep(self.episode, action_loss_type)
        except BaseException:
            self.hook.discard(self.episode)
            raise
        with self._lock:
            self._kept = True
            discard = self._abandoned
        if discard:
            self.hook.discard(self.episode)

    def abandon(self) -> None:
        with self._lock:
            self._abandoned = True
            discard = self._kept
        if discard:
            self.hook.discard(self.episode)


class EnvWorker:
    """Execute episodes through one Verifiers env and stream their traces over ZMQ."""

    def __init__(
        self,
        config: EnvConfig,
        address: str,
        max_concurrent: int | None,
    ) -> None:
        self.address = address
        self.taskset_id = config.taskset.id
        self.env = load_environment(config)
        self.task_cls = type(self.env.taskset).task_type()
        self.data_cls = self.task_cls.data_type()
        # A dispatched task is its client-side model_dump(): a field excluded from
        # serialization would vanish on the wire and rebuild silently defaulted, so
        # refuse to serve such a taskset.
        excluded = [name for name, field in self.data_cls.model_fields.items() if field.exclude]
        if excluded:
            raise ValueError(
                f"{self.data_cls.__name__} excludes {excluded} from serialization — "
                "a served task must survive the wire whole (drop exclude=True)"
            )
        # This worker's episode bound (`serve.max_concurrent`), spanning requests.
        self._gate = asyncio.Semaphore(max_concurrent) if max_concurrent else None
        # In-flight run tasks by wire request_id, so a `cancel` can abort them
        self._running: dict[str, asyncio.Task] = {}
        # Episodes the hook kept, by run request, until their reply is sent.
        self._kept: dict[bytes, tuple[EpisodeHook, Episode]] = {}
        # Train sources' algorithms by config, for their rollout-local work on episodes.
        self._algorithms: dict[str, Algorithm] = {}
        self._scorer = ScoreClient(_algorithm_worker) if _algorithm_worker is not None else None
        self._delivered: dict[str, tuple[float, EpisodeHook, Episode]] = {}
        """Recently delivered kept episodes, oldest first, for a cancel that arrives late."""
        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.address)
        # Resolve the concrete endpoint — when bound to an OS-assigned port
        # (address ending in `:0`), this is how we learn the actual port.
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)

    def _build_task(self, task_data: dict) -> Task:
        """Rebuild a request's task from its wire data: validate into the taskset's
        declared `TaskData` type and wrap it in the declared `Task` with the config's
        task subtree — the same construction the taskset's own `load()` performs. The
        client owns the taskset; this worker never `load()`s data, so pool workers
        don't each pull the dataset."""
        data = self.data_cls.model_validate(task_data)
        return self.task_cls(data, self.env.config.taskset.task)

    def _algorithm(self, req: RunRequest) -> Algorithm:
        """The train source's algorithm, built once per config."""
        from pydantic import TypeAdapter

        from prime_rl.configs.algorithm import AlgoConfig
        from prime_rl.orchestrator.algo import build_algorithm

        key = json.dumps(req.train.algo, sort_keys=True)
        if key not in self._algorithms:
            config = TypeAdapter(AlgoConfig).validate_python(req.train.algo)
            self._algorithms[key] = build_algorithm(config, clients=None)
        return self._algorithms[key]

    async def _score(self, req: RunRequest, algorithm: Algorithm, episode: Episode) -> None:
        """Have the replica's algorithm worker score the samples kept for the episode."""
        if self._scorer is None:
            raise RuntimeError(
                f"{type(algorithm).__name__} scores samples in the replica's algorithm worker: "
                "serve the env with prime-rl's `env-worker-pool` entrypoint"
            )
        samples = [
            SampleToScore(key=reference["id"], context=algorithm.sample_context(episode, trace))
            for _, trace in iter_trainable_traces([episode])
            for reference in trace.info[PAYLOAD_INFO_KEY]["branches"].values()
        ]
        if samples:
            request = ScoreRequest(train=req.train, client=req.client, model=req.model, samples=samples)
            await self._scorer.score(request)

    async def _run(self, req: RunRequest, client_id: bytes, request_id: bytes) -> RunResponse:
        # No client is built or cached here — each rollout constructs its own from
        # `req.client` and closes it, so a training run's changing model leaves nothing behind.
        ctx = ModelContext(client=req.client, model=req.model, sampling=req.sampling)
        (slot,) = self.env.slots(self._build_task(req.task_data))

        async def send_delta(delta: dict) -> None:
            if not req.live and (delta := progress_delta(delta)) is None:
                return
            await self.frontend.send_multipart([client_id, request_id, b"delta", pack(delta)], copy=False)

        # The gate spans requests: it bounds this worker's episodes in flight. The
        # streamer ships each trace as it changes; the reply below carries only its head.
        hook = _episode_hook
        async with DeltaStreamer(lambda: slot.traces, send_delta) as streamer:
            episode = await self.env.run_slot(slot, ctx, self._gate, on_trace=streamer.watch)
            if hook is not None and req.train is not None:
                algorithm = self._algorithm(req)
                # Rollout-local work on the trace graph (observation weights) annotates the
                # nodes the hook keeps.
                await algorithm.finalize_episode(episode)
                # Before the final flush, so what the hook writes to the traces streams.
                keeping = _Keeping(hook, episode)
                try:
                    # The thread runs to the end even when this await is cancelled.
                    await asyncio.shield(asyncio.to_thread(keeping.run, algorithm.action_loss_type))
                except asyncio.CancelledError:
                    keeping.abandon()
                    raise
                # From here, a failed or cancelled run discards what the hook kept.
                self._kept[request_id] = (hook, episode)
                if algorithm.scores_samples:
                    await self._score(req, algorithm, episode)
        if req.run is not None:
            episode.record_run(req.run)
            for trace in episode.traces:
                trace.info["kind"] = req.run.work.type
                trace.info["dispatch"] = {"step": req.run.work.step, "time": trace.timing.start}

        def reply() -> RunResponse:
            response = RunResponse(head=episode_head(episode).model_dump(mode="json"))
            if req.record is not None:
                response.record = episode.to_json(float_decimals=req.record.float_decimals)
                response.index = index_record(episode)
            return response

        # A long episode's head and record take a while to build: off the loop.
        return await asyncio.to_thread(reply)

    async def _handle(self, client_id: bytes, request_id: bytes, method: bytes, payload: bytes) -> None:
        try:
            route = method.decode()
            raw = msgpack.unpackb(payload, raw=False)
            if route == "health":
                response: BaseResponse = HealthResponse()
            elif route == "run":
                # Registered in the dispatch loop, before this task first runs
                try:
                    response = await self._run(RunRequest.model_validate(raw), client_id, request_id)
                finally:
                    self._running.pop(request_id.decode(), None)
            elif route == "cancel":
                target = CancelRequest.model_validate(raw).request_id
                running = self._running.get(target)
                if running is not None:
                    running.cancel()
                    get_logger().info(f"Cancelled run on client request request_id={target}")
                # The client dropped the reply, or will: nothing will take the arrays.
                self._expire_delivered()
                kept = self._kept.pop(target.encode(), None)
                if kept is None and (late := self._delivered.pop(target, None)):
                    kept = late[1:]
                if kept is not None:
                    hook, episode = kept
                    hook.discard(episode)
                response = CancelResponse(cancelled=running is not None)
            else:
                response = BaseResponse(success=False, error=f"unknown method {route!r}")
        except asyncio.CancelledError:
            # An aborted run still replies so broker/client accounting stays
            # exact; the requesting client has already given up on the result
            response = BaseResponse(success=False, error="Cancelled: rollout aborted")
        except Exception as e:  # a failed request is data, not a crash — report and keep serving
            get_logger().opt(exception=True).warning(
                f"Failed request method={method.decode(errors='replace')} - {type(e).__name__}: {e}"
            )
            response = BaseResponse(success=False, error=f"{type(e).__name__}: {e}")
        try:
            data = pack(response.model_dump(mode="python"))
        except Exception as e:
            # Encoding failures must also reply so clients don't wait indefinitely.
            get_logger().opt(exception=True).warning(f"Failed to encode response - {type(e).__name__}: {e}")
            response = BaseResponse(success=False, error=f"{type(e).__name__}: {e}")
            data = msgpack.packb(response.model_dump(), use_bin_type=True)
        delivered = False
        try:
            await self.frontend.send_multipart([client_id, request_id, b"reply", data], copy=False)
            delivered = response.success
        except zmq.ZMQError as e:
            get_logger().warning(f"Failed to send response - {type(e).__name__}: {e}")
        finally:
            # Also on a cancel (shutdown) while the reply is sending.
            kept = self._kept.pop(request_id, None)
            if kept is not None:
                hook, episode = kept
                if delivered:
                    self._remember_delivered(request_id.decode(), hook, episode)
                else:
                    hook.discard(episode)

    def _remember_delivered(self, request_id: str, hook: EpisodeHook, episode: Episode) -> None:
        self._delivered[request_id] = (time.monotonic(), hook, episode)
        asyncio.get_running_loop().call_later(LATE_CANCEL_SECONDS, self._expire_delivered)

    def _expire_delivered(self) -> None:
        now = time.monotonic()
        for request_id, (delivered_at, _, _) in list(self._delivered.items()):
            if now - delivered_at < LATE_CANCEL_SECONDS:
                break
            del self._delivered[request_id]

    async def run(self) -> None:
        get_logger().info(f"Started env worker taskset={self.taskset_id} address={self.address}")
        poller = zmq.asyncio.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        tasks: set[asyncio.Task] = set()
        # Shared servers and the interception live across requests in this worker.
        async with self.env.serving():
            try:
                while True:
                    events = dict(await poller.poll(timeout=100))
                    if self.frontend not in events:
                        continue
                    frames = await self.frontend.recv_multipart()
                    if len(frames) != 4:
                        get_logger().warning(f"Dropped malformed request frames={len(frames)} expected=4")
                        continue
                    client_id, request_id, method, payload = frames
                    task = asyncio.create_task(self._handle(client_id, request_id, method, payload))
                    if method == b"run":
                        # Register at dispatch, not handler start: a cancel
                        # processed ahead of the run's handler task must still
                        # find its target (ZMQ preserves per-client frame
                        # order, so the run always arrives first)
                        self._running[request_id.decode()] = task
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            finally:
                pending = tuple(tasks)
                try:
                    for task in pending:
                        # asyncio.run may already be cancelling these on SIGTERM.
                        if not task.cancelling():
                            task.cancel()
                    # Request finalizers still need the socket and serving resources.
                    if pending:
                        done, unfinished = await asyncio.wait(pending, timeout=_SHUTDOWN_DRAIN_SECONDS)
                        await asyncio.gather(*done, return_exceptions=True)
                        if unfinished:
                            get_logger().warning(f"Timed out draining env worker requests pending={len(unfinished)}")
                finally:
                    self.frontend.close()
                    self.ctx.term()
                get_logger().info(f"Stopped env worker taskset={self.taskset_id} address={self.address}")
