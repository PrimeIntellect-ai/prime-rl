"""TTT service v2 server: rank-0 HTTP over the FSDP/MultiLoRA engine.

Topology (under torchrun, 1+ nodes):

- **Rank 0** runs the FastAPI app (same surface as v1: `/update`, `/release`, `/health`)
  plus a collector loop that drains queued update jobs into batches
  (`engine.max_batch_wait_seconds` window, `max_tokens_per_forward` cap handled by the
  trainer's packer).
- **All ranks** sit in a work loop: rank 0 broadcasts a work order
  (`("update", jobs) | ("release", rollout_id) | ("abort", reason) | ("stop",)`) via
  `broadcast_object_list`; every rank executes the same collective path
  (forward/backward/step, checkpoint barrier); rank 0 writes checkpoints, loads adapters
  into the inference engines, and answers the HTTP callers.

Per-rollout ordering is inherited from the rollout side (the hook blocks per update), and
same-rollout jobs never coexist in one batch for the same reason. Job failures (validation,
out-of-order seq_no) are per-job: the batch runs for the healthy jobs and the failed ones
report their error to their callers.
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

# How long an HTTP caller waits on the work loop before giving up with a 503. Generous:
# an update normally completes within one batch window + forward, but a wedged work loop
# must not pin request threads forever.
_RESULT_WAIT_SECONDS = 3600.0

# Idle-heartbeat period for the control-plane broadcast (see run_server_v2.heartbeat):
# must be comfortably below the gloo group's 24h op timeout.
_HEARTBEAT_SECONDS = 3600.0


def validate_update_response(result: dict | None) -> UpdateResponse:
    """Validate the work loop's payload before touching inference replicas."""
    if result is None:
        raise RuntimeError("TTT update produced no response")
    return UpdateResponse.model_validate(result)


async def complete_adapter_load(load, mark_loaded, unload, adapter_name: str) -> None:
    """Load all replicas, recording success only after the load completes.

    Reconcile a partial/ambiguous load by unloading the name everywhere before the
    rollout lock is released. The original load/commit error remains authoritative.
    """
    try:
        await load()
        await mark_loaded()
    except BaseException:
        try:
            await unload()
        except Exception as cleanup_error:
            get_logger().warning(f"TTT adapter-load cleanup failed for {adapter_name}: {cleanup_error}")
        raise


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
            # ``stop`` is intentionally a one-item control order. Keep the transport
            # tolerant of that documented shape so rank 0 broadcasts the stop instead of
            # raising IndexError and leaving peer ranks blocked in their receive.
            payload = [order[0], order[1] if len(order) > 1 else None]
        else:
            payload = [None, None]
        if dist.is_initialized() and world.world_size > 1:
            # Control plane goes over a dedicated gloo group: non-master ranks block here
            # for arbitrarily long idle gaps, and NCCL's ~10-min watchdog would abort them;
            # gloo blocks CPU-side without a watchdog. Training collectives stay on NCCL.
            dist.broadcast_object_list(payload, src=0, group=ctrl_pg)
        kind, data = payload
        if kind == "stop":
            return
        if kind == "abort":
            # Rank 0's HTTP thread is the only way work enters this service.  If it
            # cannot bind/start, or exits later, broadcast the failure so workers do not
            # remain parked forever in the next control-plane collective.
            raise RuntimeError(f"TTT v2 HTTP server failed: {data}")
        if kind == "noop":
            continue  # idle heartbeat: keeps the gloo broadcast fed (see run_server_v2)
        if kind == "release":
            try:
                released = trainer.release(*data)
            except Exception:
                logger.exception("ttt v2: release failed after broadcast, aborting rank")
                os._exit(1)
            if world.is_master:
                order[2].had_slot = released.released
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
        # Per-rollout locks order /update and /release: an update holds its rollout's lock
        # across enqueue→result→adapter load so a concurrent release can't unload+free the
        # slot between train and engine load (orphaned adapter in vLLM otherwise).
        # They also serialize duplicate-replay retries: an orphaned 503 pending may still
        # complete in the work loop WHILE a retry waits on the lock, so the duplicate check
        # lives inside the work loop (which sees the post-completion slot version) and must
        # NOT be a pre-read here in the HTTP layer — a pre-read would race that completion.
        app.state.rollout_locks = {}
        app.state.deferred_rollout_releases = set()
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
        return {"status": "ok", "adapters": len(trainer.slots), "free_slots": len(trainer.free_idxs)}

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
                # 409: deterministic per-job validation errors; 500: unexpected-execution
                # class (non-finite loss/checkpoint rejections and other isolated faults).
                raise HTTPException(status_code=pending.error_status, detail=pending.error)
            response = validate_update_response(pending.result)

            async def mark_loaded() -> None:
                trainer.mark_loaded(request.rollout_id, request.adapter_name, response.version)

            try:
                await complete_adapter_load(
                    lambda: load_adapter(request.adapter_name, response.ckpt_path),
                    mark_loaded,
                    lambda: unload_adapter(request.adapter_name),
                    request.adapter_name,
                )
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
            # had_slot comes from the work order's result — a pre-read of trainer.slots
            # could race the work loop's own mutation of the registry.
            return {"released": ack.had_slot}
        except AdapterUnloadError as e:
            raise HTTPException(status_code=502, detail=f"adapter unload incomplete: {e}") from e
        finally:
            if not lease_transferred:
                await lease.release()

    return app


def _dedup_pendings(pendings: list["_Pending"]) -> tuple[list["_Pending"], list["_Pending"]]:
    """Keep the FIRST pending per rollout_id; later duplicates are deferred to the next
    batch. Legitimate back-to-back updates (seq_no k, k+1) can land in one drain — running
    both in one packed batch would trip the strict seq_no validation on the second."""
    seen: set[str] = set()
    batch: list[_Pending] = []
    deferred: list[_Pending] = []
    for pending in pendings:
        if pending.job.rollout_id in seen:
            deferred.append(pending)
        else:
            seen.add(pending.job.rollout_id)
            batch.append(pending)
    return batch, deferred


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
        pendings, deferred = _dedup_pendings(pendings)
        for pending in deferred:
            batch_queue.put(pending)  # re-queue: next batch, after this rollout's first job
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
    if torch.distributed.is_available() and not torch.distributed.is_initialized() and world.world_size > 1:
        torch.distributed.init_process_group(backend="nccl", device_id=torch.device("cuda", world.local_rank))
    ctrl_pg = None
    if torch.distributed.is_initialized() and world.world_size > 1:
        # Dedicated gloo group for the control-plane broadcast: idle services park
        # non-master ranks in the broadcast for hours; NCCL's ~10-min watchdog would abort
        # them, gloo just blocks CPU-side. Training collectives keep the default NCCL group.
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
            # Idle heartbeat: non-master ranks park inside the gloo broadcast between work
            # orders, and gloo enforces the group timeout per op — a fully idle service
            # (e.g. launched before the RL run starts) would have ranks 1..N time out and
            # die after 24h, taking the WHOLE service down. A periodic no-op order keeps
            # the broadcast fed; it costs one tiny gloo broadcast per interval.
            while True:
                time.sleep(_HEARTBEAT_SECONDS)
                work_queue.put(("noop", None))

        threading.Thread(target=heartbeat, daemon=True).start()
        get_logger().info(f"TTT v2 starting on {config.host}:{config.port} ({world})")

    _work_loop(trainer, work_queue, world, ctrl_pg)
