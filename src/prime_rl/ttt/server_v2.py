"""TTT service v2 server: rank-0 HTTP over the FSDP/MultiLoRA engine.

Rank 0 runs the FastAPI app (same surface as v1) plus a collector loop that drains queued
update jobs into batches; all ranks sit in a work loop fed by a rank-0 gloo broadcast and
execute the same collective path. Job failures are per-job: the batch runs for the healthy
jobs, each failure is reported to its own caller.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prime_rl.configs.ttt import TTTServiceConfig
from prime_rl.ttt.admin import (
    AdapterLoadError,
    AdapterUnloadError,
    load_adapter_into_replicas,
    unload_adapter_from_replicas,
)
from prime_rl.ttt.identity import validate_adapter_name, validate_rollout_id
from prime_rl.ttt.server import (
    ReleaseRequest,
    UpdateRequest,
    UpdateResponse,
)
from prime_rl.utils.logger import get_logger

# HTTP caller wait on the work loop before a 503 — a wedged loop must not pin threads forever.
_RESULT_WAIT_SECONDS = 3600.0

# Idle-heartbeat period (see run_server_v2.heartbeat); must stay well below gloo's 24h op timeout.
_HEARTBEAT_SECONDS = 3600.0


class InitBroadcasterRequest(BaseModel):
    host: str


class UpdateBaseWeightsRequest(BaseModel):
    step: int
    weight_dir: str | None = None


@dataclass
class _BaseWeightsAck:
    done: threading.Event = field(default_factory=threading.Event)
    error: str | None = None


@dataclass
class _Pending:
    """One queued update job + the event/result plumbing back to its HTTP caller."""

    job: object  # UpdateJob
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: str | None = None
    error_status: int = 409  # validation errors; anything unexpected maps to 500


@dataclass
class _ReleaseAck:
    """Release work-order ack: `had_slot` comes from the work loop's authoritative
    `trainer.release` result, not a racy pre-read of `trainer.slots`."""

    done: threading.Event = field(default_factory=threading.Event)
    had_slot: bool = False


@dataclass
class _KeyedLockLease:
    lock: asyncio.Lock
    key: str
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.lock.release()


async def _acquire_rollout_lease(locks: dict[str, asyncio.Lock], key: str) -> _KeyedLockLease:
    """Acquire the stable per-rollout lock retained for this service lifetime."""
    lock = locks.setdefault(key, asyncio.Lock())
    await lock.acquire()
    return _KeyedLockLease(lock, key)


def _defer_lease_release(app: FastAPI, lease: _KeyedLockLease, completion: asyncio.Task) -> None:
    """Keep a rollout lease after its request times out or is cancelled."""

    async def release_after_completion() -> None:
        try:
            await completion
        except BaseException as e:
            # A detached release transaction may fail after its HTTP caller has gone
            # away. Consume the exception (release is idempotent, so an explicit retry
            # still converges) instead of emitting an unhandled-task warning.
            get_logger().warning(f"TTT detached rollout operation failed for {lease.key}: {e}")
        finally:
            await lease.release()

    task = asyncio.create_task(release_after_completion())
    app.state.deferred_rollout_releases.add(task)
    task.add_done_callback(app.state.deferred_rollout_releases.discard)


async def _wait_with_lease(app: FastAPI, lease: _KeyedLockLease, done: threading.Event) -> None:
    """Wait up to the HTTP result deadline, transferring ownership on interruption."""
    completion = asyncio.create_task(asyncio.to_thread(done.wait))
    try:
        await asyncio.wait_for(asyncio.shield(completion), _RESULT_WAIT_SECONDS)
    except BaseException:
        # ``shield`` keeps the thread waiter alive. The background task holds the lease
        # until the queued/running work order really completes, so a retry or release
        # cannot overtake an orphaned update after client cancellation or a 503 timeout.
        _defer_lease_release(app, lease, completion)
        raise


def _work_loop(trainer, work_queue: Queue, world, ctrl_pg=None) -> None:
    """The per-rank execution loop (runs in the main thread on every rank). Rank 0 feeds it
    from the HTTP side via `work_queue`; other ranks receive orders by broadcast."""
    import torch.distributed as dist

    from prime_rl.ttt.trainer_v2 import UpdateJob  # noqa: F401 (broadcast payload type)

    logger = get_logger()
    while True:
        if world.is_master:
            order = work_queue.get()
            # ``stop`` is a one-item order; tolerate it so peers aren't stranded mid-receive.
            payload = [order[0], order[1] if len(order) > 1 else None]
        else:
            payload = [None, None]
        if dist.is_initialized() and world.world_size > 1:
            # Control plane rides the watchdog-free gloo group (see run_server_v2).
            dist.broadcast_object_list(payload, src=0, group=ctrl_pg)
        kind, data = payload
        if kind == "stop":
            return
        if kind == "abort":
            # A dead rank-0 HTTP thread must not leave workers parked in the broadcast forever.
            raise RuntimeError(f"TTT v2 HTTP server failed: {data}")
        if kind == "noop":
            continue  # idle heartbeat: keeps the gloo broadcast fed (see run_server_v2)
        if kind == "init_receiver":
            try:
                trainer.setup_weight_receiver()
            except Exception as exc:
                logger.exception("ttt v2: broadcaster initialization failed")
                if world.is_master:
                    order[2].error = f"{type(exc).__name__}: {exc}"
                    order[2].done.set()
                continue
            if world.is_master:
                order[2].done.set()
            continue
        if kind == "recv_weights":
            # data = (step, exact filesystem weight dir or None). For NCCL a failed
            # receive can desynchronize ranks, so preserve fail-fast behavior. Filesystem
            # failures are coherent and can be returned without killing the service.
            try:
                trainer.receive_base_weights(*data)
            except Exception as exc:
                if trainer.config.weight_broadcast.type == "nccl":
                    logger.exception("ttt v2: base weight receive failed, aborting rank")
                    os._exit(1)
                logger.exception("ttt v2: filesystem base weight update failed")
                if world.is_master:
                    order[2].error = f"{type(exc).__name__}: {exc}"
                    order[2].done.set()
                continue
            if world.is_master:
                order[2].done.set()
            continue
        if kind == "release":
            try:
                released = trainer.release(*data)
            except Exception:
                logger.exception("ttt v2: release failed after broadcast, aborting rank")
                os._exit(1)
            if world.is_master:
                order[2].had_slot = released
                order[2].done.set()
            continue
        # kind == "update": data = list[UpdateJob]; on rank 0, order[2] = list[_Pending]
        jobs = data
        try:
            results = trainer.update_batch(jobs)
        except Exception:
            # All request/sequence validation is isolated before slot claims. Anything
            # escaping update_batch — including ValueError — is post-preparation and may
            # have mutated a slot or stranded peers in a collective, so every exception is
            # fatal to this torchrun worker.
            logger.exception("ttt v2: post-preparation batch failure, aborting rank")
            os._exit(1)
        if world.is_master:
            for pending in order[2]:
                pending.result = results.get(pending.job.rollout_id)
                if pending.result is None:
                    pending.error = "job produced no result"
                    pending.error_status = 500
                elif "error" in pending.result:
                    # Per-job failure isolated inside update_batch: a ValueError is a real
                    # validation rejection (the job's own 409); anything else — including
                    # the non-finite loss/checkpoint rejections — is unexpected-execution
                    # class and surfaces as 500 (the hook retries neither status).
                    pending.error = pending.result["error"]
                    if not pending.error.startswith("ValueError:"):
                        pending.error_status = 500
                    pending.result = None
                pending.done.set()


def _validate_and_split(trainer, pendings: list["_Pending"]) -> list["_Pending"]:
    """Validate and materialize sequences on rank 0 before broadcasting.

    Workers must consume the exact same prepared token lists. Running tokenizer/chat-
    template code independently on every rank risks a rank-local exception or divergent
    rendering, followed by mismatched packed-forward collectives. Slot claims remain in
    ``update_batch`` and therefore still happen identically on every rank.
    """
    valid: list[_Pending] = []
    for pending in pendings:
        try:
            cached = trainer.validate_job(pending.job)
            if cached is None:
                trainer.prepare_job(pending.job)
        except ValueError as e:
            pending.error = str(e)
            pending.done.set()
            continue
        except Exception as e:
            # Preparation has not been broadcast and mutates no GPU/slot state, so an
            # unexpected tokenizer/template failure is safe to isolate as this job's 500.
            pending.error = f"{type(e).__name__}: {e}"
            pending.error_status = 500
            pending.done.set()
            continue
        valid.append(pending)
    return valid


def build_app_v2(config: TTTServiceConfig, trainer, work_queue: Queue) -> FastAPI:
    """The rank-0 app: enqueue jobs, wait for the work loop's results, drive engine adapter
    loads. `trainer` is touched only for pure validation/sequence preparation here — GPU
    execution and slot mutation happen in the work loop on all ranks."""
    from prime_rl.ttt.trainer_v2 import UpdateJob

    batch_queue: Queue[_Pending] = Queue()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(timeout=config.admin_timeout_seconds)
        # Per-rollout locks order /update and /release across enqueue→result→adapter load
        # (a concurrent release could otherwise unload+free the slot between train and engine
        # load) and serialize duplicate-replay retries — the duplicate check MUST live in the
        # work loop, not as an HTTP pre-read, or it races an orphaned pending's completion.
        app.state.rollout_locks = {}
        app.state.deferred_rollout_releases = set()
        # One transaction per policy step. Exact retries await/reuse it instead of
        # enqueueing a second one-shot collective receive.
        app.state.base_weight_updates = {}
        app.state.broadcaster_init = None
        collector = threading.Thread(
            target=_collector_loop,
            args=(config, trainer, batch_queue, work_queue),
            daemon=True,
        )
        collector.start()
        try:
            yield
        finally:
            await app.state.http.aclose()

    app = FastAPI(lifespan=lifespan)

    async def load_adapter(adapter_name: str, ckpt_path: str) -> None:
        await load_adapter_into_replicas(
            app.state.http,
            config.inference_admin_urls,
            adapter_name,
            ckpt_path,
            config.admin_timeout_seconds,
        )

    async def unload_adapter(adapter_name: str) -> None:
        await unload_adapter_from_replicas(
            app.state.http,
            config.inference_admin_urls,
            adapter_name,
            config.admin_timeout_seconds,
        )

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "adapters": len(trainer.slots),
            "free_slots": len(trainer.free_idxs),
            "base_version": getattr(trainer, "base_version", 0),
        }

    async def _wait_for_base_ack(ack: _BaseWeightsAck) -> None:
        deadline = time.monotonic() + _RESULT_WAIT_SECONDS
        while not ack.done.is_set():
            if time.monotonic() >= deadline:
                raise HTTPException(status_code=503, detail="TTT work loop did not answer in time")
            await asyncio.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if ack.error is not None:
            raise HTTPException(status_code=500, detail=ack.error)

    @app.post("/init_broadcaster")
    async def init_broadcaster(request: InitBroadcasterRequest) -> dict:
        wb = config.weight_broadcast
        if wb is None or wb.type != "nccl":
            raise HTTPException(status_code=409, detail="TTT service is not configured for NCCL weight broadcast")
        if request.host != wb.host:
            raise HTTPException(status_code=409, detail=f"NCCL host mismatch: expected {wb.host}, got {request.host}")
        ack = app.state.broadcaster_init
        if ack is None:
            ack = _BaseWeightsAck()
            app.state.broadcaster_init = ack
            work_queue.put(("init_receiver", None, ack))
        await _wait_for_base_ack(ack)
        return {"status": "ok"}

    @app.post("/update_base_weights")
    async def update_base_weights(request: UpdateBaseWeightsRequest) -> dict:
        """Apply one policy broadcast exactly once; retries share the original ack."""
        wb = config.weight_broadcast
        if wb is None:
            raise HTTPException(status_code=409, detail="TTT service does not follow policy weight broadcasts")
        if request.step < trainer.base_version:
            raise HTTPException(
                status_code=409,
                detail=f"stale base-weight update {request.step}; current version is {trainer.base_version}",
            )
        if request.step == trainer.base_version:
            return {"status": "ok", "base_version": request.step}
        if wb.type == "nccl" and trainer.weight_receiver is None:
            raise HTTPException(status_code=409, detail="NCCL broadcaster has not been initialized")

        ack = app.state.base_weight_updates.get(request.step)
        if ack is None:
            ack = _BaseWeightsAck()
            app.state.base_weight_updates[request.step] = ack
            work_queue.put(("recv_weights", (request.step, request.weight_dir), ack))
        await _wait_for_base_ack(ack)
        return {"status": "ok", "base_version": request.step}

    @app.post("/update")
    async def update(request: UpdateRequest) -> UpdateResponse:
        try:
            # rollout_id is a filesystem path component; adapter_name only needs to be
            # present — the trusted in-repo hook derives it as {prefix}-{rollout_id}.
            validate_rollout_id(request.rollout_id)
            validate_adapter_name(request.adapter_name)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        job = UpdateJob(
            rollout_id=request.rollout_id,
            adapter_name=request.adapter_name,
            token_ids=request.token_ids,
            loss_mask=request.loss_mask,
            seq_no=request.seq_no,
            qa_pairs=request.qa_pairs,
            train_rollout=request.train_rollout,
            system_prompt=request.system_prompt,
            tools=request.tools,
        )
        lease = await _acquire_rollout_lease(app.state.rollout_locks, request.rollout_id)
        lease_transferred = False
        try:
            pending = _Pending(job=job)
            batch_queue.put(pending)
            try:
                await _wait_with_lease(app, lease, pending.done)
            except TimeoutError:
                lease_transferred = True
                raise HTTPException(status_code=503, detail="TTT work loop did not answer in time")
            except BaseException:
                lease_transferred = True
                raise
            if pending.error is not None:
                # 409: deterministic per-job validation errors; 500: unexpected-execution class.
                raise HTTPException(status_code=pending.error_status, detail=pending.error)
            if pending.result is None:
                raise RuntimeError("TTT update produced no response")
            response = UpdateResponse.model_validate(pending.result)
            try:
                # Mark loaded only after every replica load; reconcile a partial/ambiguous
                # load by unloading everywhere before the rollout lock is released.
                try:
                    await load_adapter(request.adapter_name, response.ckpt_path)
                    trainer.mark_loaded(request.rollout_id, request.adapter_name, response.version)
                except BaseException:
                    try:
                        await unload_adapter(request.adapter_name)
                    except Exception as cleanup_error:
                        get_logger().warning(
                            f"TTT adapter-load cleanup failed for {request.adapter_name}: {cleanup_error}"
                        )
                    raise
            except (AdapterLoadError, httpx.HTTPError, TimeoutError) as e:
                raise HTTPException(status_code=502, detail=f"adapter load failed: {e}") from e
            return response
        finally:
            if not lease_transferred:
                await lease.release()

    @app.post("/release")
    async def release(request: ReleaseRequest) -> dict:
        try:
            validate_rollout_id(request.rollout_id)
            validate_adapter_name(request.adapter_name)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        lease = await _acquire_rollout_lease(app.state.rollout_locks, request.rollout_id)
        lease_transferred = False
        try:
            try:
                trainer.validate_release(request.rollout_id, request.adapter_name)
            except ValueError as e:
                raise HTTPException(status_code=409, detail=str(e)) from e

            async def release_and_unload() -> _ReleaseAck:
                ack = _ReleaseAck()
                work_queue.put(("release", (request.rollout_id, request.adapter_name), ack))
                await asyncio.to_thread(ack.done.wait)
                # Unload UNCONDITIONALLY: a retry after a lost response finds the slot
                # already dropped, but the first attempt's engine unload may never have
                # run — gating on had_slot would leak the adapter in vLLM until restart.
                await unload_adapter(request.adapter_name)
                return ack

            transaction = asyncio.create_task(release_and_unload())
            try:
                ack = await asyncio.wait_for(asyncio.shield(transaction), _RESULT_WAIT_SECONDS)
            except TimeoutError:
                lease_transferred = True
                _defer_lease_release(app, lease, transaction)
                raise HTTPException(status_code=503, detail="TTT work loop did not answer in time")
            except asyncio.CancelledError:
                lease_transferred = True
                _defer_lease_release(app, lease, transaction)
                raise
            return {"released": ack.had_slot}  # authoritative (see _ReleaseAck)
        except AdapterUnloadError as e:
            raise HTTPException(status_code=502, detail=f"adapter unload incomplete: {e}") from e
        finally:
            if not lease_transferred:
                await lease.release()

    return app


def _collector_loop(config, trainer, batch_queue: Queue, work_queue: Queue) -> None:
    """Rank 0: drain queued jobs into batches. Waits `max_batch_wait_seconds` after the
    first arrival to let concurrent rollouts' updates coalesce, validates each job, then
    hands the batch to the work loop."""
    wait = config.engine.max_batch_wait_seconds
    while True:
        first: _Pending = batch_queue.get()
        pendings = [first]
        deadline = threading.Event()
        deadline.wait(wait)  # simple sleep; arrivals buffer in the queue meanwhile
        while True:
            try:
                pendings.append(batch_queue.get_nowait())
            except Empty:
                break
        # Keep the FIRST pending per rollout_id; a same-rollout follow-up (seq_no k, k+1
        # in one drain) would trip strict seq_no validation — re-queue it for the next batch.
        seen: set[str] = set()
        deduped: list[_Pending] = []
        for pending in pendings:
            if pending.job.rollout_id in seen:
                batch_queue.put(pending)
            else:
                seen.add(pending.job.rollout_id)
                deduped.append(pending)
        pendings = deduped
        valid = _validate_and_split(trainer, pendings)
        if valid:
            work_queue.put(("update", [p.job for p in valid], valid))


def _monitor_http_server(server, http_thread: threading.Thread, work_queue: Queue, startup_timeout: float) -> None:
    """Turn uvicorn startup/exit failures into a work order seen by every rank."""
    deadline = time.monotonic() + startup_timeout
    while http_thread.is_alive() and not server.started:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            work_queue.put(("abort", f"did not report ready within {startup_timeout:g} seconds"))
            return
        # Joining with a short timeout detects early thread exit without a busy spin.
        http_thread.join(timeout=min(0.05, remaining))

    if not server.started:
        work_queue.put(("abort", "exited before startup completed (bind/configuration/startup failure)"))
        return

    # A serving TTT process has no normal independent HTTP-thread exit.  Observe it for
    # the process lifetime so an exception or unexpected uvicorn shutdown also wakes the
    # distributed work loop instead of leaving non-master ranks parked forever.
    http_thread.join()
    work_queue.put(("abort", "exited unexpectedly after startup"))


def run_server_v2(config: TTTServiceConfig) -> None:
    """Entry under torchrun: build the engine on every rank; rank 0 serves HTTP in a
    daemon thread and feeds the shared work loop; all ranks execute it."""
    import torch
    import uvicorn

    from prime_rl.trainer.world import get_world
    from prime_rl.ttt.trainer_v2 import TTTTrainerV2

    world = get_world()
    torch.cuda.set_device(world.local_rank)
    # Init unconditionally (not just world_size > 1): the trainer stack's MultiRunManager
    # needs the c10d default store even on a single rank — without this a 1-GPU service
    # (e.g. the tiny-model smoke) crashes at startup.
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", device_id=torch.device("cuda", world.local_rank))
    ctrl_pg = None
    if torch.distributed.is_initialized() and world.world_size > 1:
        # Control plane on a dedicated gloo group: non-master ranks park in the broadcast
        # for hours and NCCL's ~10-min watchdog would abort them; training stays on NCCL.
        ctrl_pg = torch.distributed.new_group(backend="gloo", timeout=datetime.timedelta(hours=24))
    trainer = TTTTrainerV2(config)
    trainer.ctrl_pg = ctrl_pg  # checkpoint barrier rides the gloo group too
    work_queue: Queue = Queue()

    if world.is_master:
        app = build_app_v2(config, trainer, work_queue)
        server = uvicorn.Server(
            uvicorn.Config(app, host=config.host, port=config.port, log_level=config.log.level.lower())
        )
        http_thread = threading.Thread(target=server.run, daemon=True)
        http_thread.start()
        threading.Thread(
            target=_monitor_http_server,
            args=(server, http_thread, work_queue, config.startup_timeout_seconds),
            daemon=True,
        ).start()

        def heartbeat() -> None:
            # Gloo enforces its 24h timeout per op — a fully idle service would kill
            # ranks 1..N; a periodic no-op order keeps the broadcast fed.
            while True:
                time.sleep(_HEARTBEAT_SECONDS)
                work_queue.put(("noop", None))

        threading.Thread(target=heartbeat, daemon=True).start()
        get_logger().info(f"TTT v2 starting on {config.host}:{config.port} ({world})")

    _work_loop(trainer, work_queue, world, ctrl_pg)
