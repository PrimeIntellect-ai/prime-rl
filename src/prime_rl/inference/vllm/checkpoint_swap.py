"""Swap the checkpoint a deployment serves (``POST /swap-checkpoint``): install a ModelExpress version, such as one a
checkpoint publisher offers, into every engine of a DP group without restarting it.

A swap resolves its target on ModelExpress and stages it on every worker while the engines keep serving, so a missing
or unreadable version fails before anything changes. It then stops admitting requests and handles the ones in flight
(``mode``: fail if there are any, drain them, or abort them), pauses the group, installs the version, drops the prefix
cache (its blocks hold KV of the old weights), labels the tokens sampled from then on with the version's step and
resumes. Requests that arrive meanwhile wait at the API servers and then run on the new weights.

The first version of a mesh (one checkpoint publisher, or one trainer) that a group installs stages while paused: that
stage records how the mesh's tensors map onto the engine's, on the live model. Its time counts toward the pause.

One swap runs at a time per group. A request for the version being installed waits for that swap's result; one for
another version gets 409. A failed install may leave the weights partly written, so the group stays paused, reports
unhealthy and retries the install until it succeeds or ``timeout`` passes. It never resumes on mixed weights. A new
swap request, to any version, takes over from a failed one.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Literal

import grpc
from modelexpress_rl import ModelExpressControlClient, WeightVersion, WeightVersionState
from pydantic import BaseModel, Field, model_validator
from vllm.logger import init_logger

from prime_rl.inference.vllm.weight_pull import Engines, mesh_key
from prime_rl.transports.weights.checkpoint_versions import version_integer_tables

logger = init_logger("vllm.inference.vllm.checkpoint_swap")

IDLE_POLL_S = 0.05
ABORT_TIMEOUT_S = 60.0
MAX_REPAIR_INTERVAL_S = 30.0


@dataclass(frozen=True)
class SwapConfig:
    """Where a group finds its versions (the ModelExpress server and model name), its workers' staging settings, and
    how long a group retries a failed install."""

    host: str
    port: int
    model_name: str
    staging_buffer_bytes: int | None
    staging_buffers_count: int
    timeout: int

    @property
    def server_url(self) -> str:
        return f"{self.host}:{self.port}"


class SwapRequest(BaseModel):
    step: int | None = None
    """The step to serve: the newest READY version with this version number."""

    version_uid: str | None = None
    """The version to serve, instead of a step."""

    mode: Literal["fail", "drain", "abort"] = "fail"
    """Requests in flight: refuse the swap (409), stop admitting and wait for them (up to ``drain_timeout_s``), or
    abort them (they finish with the ``abort`` finish reason)."""

    drain_timeout_s: float = Field(300.0, gt=0)

    draft_dir: str | None = None
    """A speculative-decoding draft to load with the version, in the pause (a directory or a Hub id in the local
    cache, matching the served draft's architecture). Without one the draft keeps its weights; a draft that shares
    the target's embedding or LM head reads the new ones."""

    @model_validator(mode="after")
    def one_target(self):
        if (self.step is None) == (self.version_uid is None):
            raise ValueError("Give either step or version_uid")
        return self


class SwapFailed(Exception):
    """A swap that did not happen, or did not finish, with the HTTP status to answer it with."""

    def __init__(self, status: int, error: str, **details):
        super().__init__(error)
        self.status = status
        self.body = {"error": error, **details}


class CheckpointSwapper:
    """Moves one DP group (``engines``) onto the versions it is asked for, one swap at a time."""

    def __init__(self, engines: Engines, config: SwapConfig):
        self.engines = engines
        self.config = config
        self.session_id = uuid.uuid4().hex
        self._control: ModelExpressControlClient | None = None
        self._swapping: tuple[str, asyncio.Task] | None = None
        self._repair: asyncio.Task | None = None
        self._stop_repair = asyncio.Event()
        self.broken: str | None = None
        """Why the group is paused and unhealthy after a failed install, None while it serves."""
        self.phase: str | None = None
        """Where the running swap is: staging (while serving), draining (holding new requests) or paused."""

    @property
    def control(self) -> ModelExpressControlClient:
        if self._control is None:
            self._control = ModelExpressControlClient.connect(server_url=self.config.server_url)
        return self._control

    async def prepare(self) -> None:
        """Connect the group's workers to ModelExpress ahead of the first swap, so that they build what a first stage
        needs while they serve. Best effort: every swap connects them again."""
        try:
            await self.engines.ready()
            await self.engines.init(self._init_payload())
        except Exception as error:
            logger.warning(f"Connecting the engines to ModelExpress ahead of a swap failed: {error!r}")

    def status(self) -> dict:
        return {
            "swapping": self._swapping[0] if self._swapping is not None else None,
            "phase": self.phase,
            "broken": self.broken,
            "repairing": self._repair is not None and not self._repair.done(),
            "model_name": self.config.model_name,
        }

    async def swap(self, request: SwapRequest) -> dict:
        try:
            target = await self.resolve(request)
        except grpc.RpcError as error:
            raise SwapFailed(502, f"ModelExpress at {self.config.server_url} failed: {error.details()}") from error
        if self._swapping is not None:
            uid, task = self._swapping
            if uid != target.version_id:
                raise SwapFailed(409, f"A swap to {uid} is in progress", swapping=uid)
        else:
            task = asyncio.create_task(self._swap(target, request))
            self._swapping = (target.version_id, task)
            task.add_done_callback(self._swap_done)
        # The swap goes on if this request is cancelled (its client disconnected): it must not stop half-way
        return await asyncio.shield(task)

    def _swap_done(self, task: asyncio.Task) -> None:
        if self._swapping is not None and self._swapping[1] is task:
            self._swapping = None
        self.phase = None
        if not task.cancelled() and task.exception() is not None and not isinstance(task.exception(), SwapFailed):
            logger.error("Checkpoint swap failed", exc_info=task.exception())

    async def resolve(self, request: SwapRequest) -> WeightVersion:
        """The READY version a request names: by uid, or the newest of a step."""
        if request.version_uid is not None:
            try:
                version = await asyncio.to_thread(self.control.get_weight_version, request.version_uid)
            except grpc.RpcError as error:
                if error.code() is grpc.StatusCode.NOT_FOUND:
                    raise SwapFailed(404, f"No version {request.version_uid} on ModelExpress") from error
                raise
            if version.model_name != self.config.model_name:
                raise SwapFailed(
                    404, f"Version {version.version_id} is of {version.model_name}, not {self.config.model_name}"
                )
        else:
            versions = await asyncio.to_thread(self.control.list_weight_versions, model_name=self.config.model_name)
            of_step = [version for version in versions if version.version_number == request.step]
            ready = [version for version in of_step if version.state is WeightVersionState.READY]
            if not ready:
                states = sorted({version.state.value for version in of_step})
                raise SwapFailed(
                    404,
                    f"No READY version of step {request.step} of {self.config.model_name} on ModelExpress"
                    + (f" (only {', '.join(states)})" if states else ""),
                )
            version = max(ready, key=lambda version: version.created_at_unix_ms)
        if version.state is not WeightVersionState.READY:
            raise SwapFailed(409, f"Version {version.version_id} is {version.state.value}, not READY")
        if version.version_number is None:
            raise SwapFailed(409, f"Version {version.version_id} has no step to label the tokens it samples with")
        return version

    async def _swap(self, target: WeightVersion, request: SwapRequest) -> dict:
        if self._repair is not None:
            await self._stop_repairing()
        uid, step, mesh = target.version_id, target.version_number, mesh_key(target)
        # After a failed install the group is paused, unhealthy and holds new requests already
        paused = self.broken is not None
        timings = {"stage": 0.0, "drain": 0.0, "pause": 0.0, "apply": 0.0}
        start = time.perf_counter()
        await self.engines.init(self._init_payload())
        if not paused and request.draft_dir is None:
            served = await self.engines.served()
            if served == (step, uid):
                return {"step": step, "version_uid": uid, "unchanged": True}
        await self._check_integer_tables(uid)

        live = not paused and await self.engines.can_stage_live(mesh)
        if live:
            self.phase = "staging"
            mark = time.perf_counter()
            await self._stage_live(uid)
            timings["stage"] = time.perf_counter() - mark

        in_flight = aborted = 0
        if not paused:
            mark = time.perf_counter()
            try:
                in_flight = await self.engines.hold()
                self.phase = "draining"
                if in_flight and request.mode == "fail":
                    raise SwapFailed(
                        409, f"{in_flight} requests in flight (mode 'fail'); nothing changed", in_flight=in_flight
                    )
                if in_flight and request.mode == "abort":
                    aborted = await self.engines.abort()
                if in_flight:
                    timeout = request.drain_timeout_s if request.mode == "drain" else ABORT_TIMEOUT_S
                    await self._wait_idle(timeout)
            except BaseException:
                await self.engines.release()
                if live:
                    await self.engines.discard()
                raise
            timings["drain"] = time.perf_counter() - mark
            try:
                await self.engines.pause()
            except BaseException as error:
                await self._abandon(live)
                raise SwapFailed(500, f"Pausing the engines failed: {error!r}") from error
        paused_at = time.perf_counter()
        self.phase = "paused"

        if not live:
            mark = time.perf_counter()
            try:
                await self.engines.stage(uid, live=False)
            except Exception as error:
                await self.engines.discard()
                if paused:
                    # The weights are still partly written from the install that failed before
                    reason = f"Staging {uid} (step {step}) failed while the group is broken: {error!r}"
                    await self._break(target, reason, request.draft_dir)
                    raise SwapFailed(
                        500, f"{reason}; the group stays paused and unhealthy and retries", version_uid=uid
                    ) from error
                # A stage leaves the live weights as they are, also a failed one
                await self._abandon(False)
                raise SwapFailed(
                    502, f"Staging {uid} failed; the engines resume on the previous weights: {error}", version_uid=uid
                ) from error
            timings["stage_paused"] = time.perf_counter() - mark
        try:
            mark = time.perf_counter()
            await self.engines.apply(uid, mesh, step)
            timings["apply"] = time.perf_counter() - mark
            if request.draft_dir is not None:
                mark = time.perf_counter()
                await self.engines.update_draft(request.draft_dir)
                timings["draft"] = time.perf_counter() - mark
            # Its blocks hold KV of the old weights. Nothing runs: no request holds a block.
            await self.engines.reset_prefix_cache()
        except Exception as error:
            reason = f"Installing {uid} (step {step}) failed: {error!r}"
            await self._break(target, reason, request.draft_dir)
            raise SwapFailed(
                500, f"{reason}; the group stays paused and unhealthy and retries the install", version_uid=uid
            ) from error
        await self._resume()
        timings["pause"] = time.perf_counter() - paused_at
        timings["total"] = time.perf_counter() - start
        result = {
            "step": step,
            "version_uid": uid,
            "unchanged": False,
            "cold": not live,
            "in_flight": in_flight,
            "aborted": aborted,
            "draft_dir": request.draft_dir,
            "timings": timings,
        }
        logger.info(f"Swapped to step {step} ({uid}): {result}")
        return result

    def _init_payload(self) -> dict:
        return {
            "host": self.config.host,
            "port": self.config.port,
            "timeout": self.config.timeout,
            "session_id": self.session_id,
            "staging_buffer_bytes": self.config.staging_buffer_bytes,
            "staging_buffers_count": self.config.staging_buffers_count,
            "model_name": self.config.model_name,
        }

    async def _check_integer_tables(self, uid: str) -> None:
        """Versions leave integer tables out (ModelExpress publishes floating-point tensors only), so the engines keep
        the ones they loaded. Refuse a checkpoint whose tables differ."""
        expected = version_integer_tables(uid)
        if expected is None:
            return
        actual = await self.engines.integer_tables()
        if actual != expected:
            raise SwapFailed(
                409,
                f"Version {uid} has other integer tables (digest {expected}) than the engines loaded "
                f"({actual or 'they disagree'}); ModelExpress cannot install integer tensors, so serving this "
                "checkpoint needs a restart on it",
            )

    async def _stage_live(self, uid: str) -> None:
        try:
            await self.engines.stage(uid, live=True)
        except Exception as error:
            await self.engines.discard()
            raise SwapFailed(
                502, f"Staging {uid} failed; the engines keep serving the previous weights: {error}", version_uid=uid
            ) from error

    async def _wait_idle(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while (in_flight := await self.engines.in_flight()) > 0:
            if time.monotonic() >= deadline:
                raise SwapFailed(
                    409, f"{in_flight} requests still in flight after {timeout}s; nothing changed", in_flight=in_flight
                )
            await asyncio.sleep(IDLE_POLL_S)

    async def _abandon(self, staged: bool) -> None:
        """Serve the previous weights again: resume, admit and drop what was staged."""
        if staged:
            await self.engines.discard()
        await self.engines.resume()
        await self.engines.release()

    async def _resume(self) -> None:
        await self.engines.resume()
        await self.engines.release()
        if self.broken is not None:
            await self.engines.set_health(True)
            self.broken = None

    async def _break(self, target: WeightVersion, reason: str, draft_dir: str | None) -> None:
        """Keep the group paused with mixed weights out of service, and retry the install."""
        logger.error(f"{reason}; the group stays paused and unhealthy")
        self.broken = reason
        try:
            await self.engines.set_health(False, reason)
        except Exception:
            logger.exception("Reporting the group unhealthy failed")
        self._stop_repair.clear()
        self._repair = asyncio.create_task(self._repair_install(target, draft_dir))

    async def _repair_install(self, target: WeightVersion, draft_dir: str | None) -> None:
        """Retry installing ``target`` from a fresh stage until it succeeds, ``timeout`` passes or a swap takes over."""
        uid, step, mesh = target.version_id, target.version_number, mesh_key(target)
        deadline = time.monotonic() + self.config.timeout
        interval = 1.0
        while True:
            try:
                await asyncio.wait_for(self._stop_repair.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                await self.engines.discard()
                await self.engines.stage(uid, live=False)
                await self.engines.apply(uid, mesh, step)
                if draft_dir is not None:
                    await self.engines.update_draft(draft_dir)
                await self.engines.reset_prefix_cache()
                await self._resume()
                logger.info(f"Installed {uid} (step {step}) on retry; the group serves again")
                return
            except Exception:
                logger.exception(f"Retrying the install of {uid} failed")
            if time.monotonic() >= deadline:
                logger.error(
                    f"No install of {uid} succeeded within {self.config.timeout}s; the group stays paused and "
                    "unhealthy until a swap succeeds"
                )
                return
            interval = min(2 * interval, MAX_REPAIR_INTERVAL_S)

    async def _stop_repairing(self) -> None:
        """Let a running repair finish its attempt and stop."""
        assert self._repair is not None
        self._stop_repair.set()
        await self._repair
        self._repair = None
