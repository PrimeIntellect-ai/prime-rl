"""TTT service v2 server: rank-0 HTTP over the FSDP/MultiLoRA engine.

Topology (under torchrun, 1+ nodes):

- **Rank 0** runs the FastAPI app (same surface as v1: `/update`, `/release`, `/health`)
  plus a collector loop that drains queued update jobs into batches
  (`engine.max_batch_wait_seconds` window, `max_tokens_per_forward` cap handled by the
  trainer's packer).
- **All ranks** sit in a work loop: rank 0 broadcasts a work order
  (`("update", jobs) | ("release", rollout_id) | ("stop",)`) via
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
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue

import httpx
from fastapi import FastAPI, HTTPException

from prime_rl.configs.ttt import TTTServiceConfig
from prime_rl.ttt.server import ReleaseRequest, UpdateRequest, UpdateResponse
from prime_rl.utils.logger import get_logger

# How long an HTTP caller waits on the work loop before giving up with a 503. Generous:
# an update normally completes within one batch window + forward, but a wedged work loop
# must not pin request threads forever.
_RESULT_WAIT_SECONDS = 3600.0

# Idle-heartbeat period for the control-plane broadcast (see run_server_v2.heartbeat):
# must be comfortably below the gloo group's 24h op timeout.
_HEARTBEAT_SECONDS = 3600.0


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


def _work_loop(trainer, work_queue: Queue, world, ctrl_pg=None) -> None:
    """The per-rank execution loop (runs in the main thread on every rank). Rank 0 feeds it
    from the HTTP side via `work_queue`; other ranks receive orders by broadcast."""
    import torch.distributed as dist

    from prime_rl.ttt.trainer_v2 import UpdateJob  # noqa: F401 (broadcast payload type)

    logger = get_logger()
    while True:
        if world.is_master:
            order = work_queue.get()
            payload = [order[0], order[1]]
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
        if kind == "noop":
            continue  # idle heartbeat: keeps the gloo broadcast fed (see run_server_v2)
        if kind == "release":
            released = trainer.release(data)
            if world.is_master:
                order[2].had_slot = released is not None
                order[2].done.set()
            continue
        # kind == "update": data = list[UpdateJob]; on rank 0, order[2] = list[_Pending]
        jobs = data
        try:
            results = trainer.update_batch(jobs)
        except ValueError as e:
            # Deterministic validation error raised identically on every rank (jobs arrive
            # in the same order via broadcast): safe to report per-batch and keep going.
            logger.exception("ttt v2: batch failed validation")
            if world.is_master:
                for pending in order[2]:
                    pending.error = f"{type(e).__name__}: {e}"
                    pending.done.set()
            continue
        except Exception:
            # A rank-local fault mid-collective (OOM, NCCL, driver) would leave the peer
            # ranks hung inside the collective. Crash this rank: torchrun tears the whole
            # group down loudly instead of wedging silently — fail-loud stance.
            logger.exception("ttt v2: non-deterministic batch failure, aborting rank")
            os._exit(1)
        if world.is_master:
            for pending in order[2]:
                pending.result = results.get(pending.job.rollout_id)
                if pending.result is None:
                    pending.error = "job produced no result"
                    pending.error_status = 500
                elif "error" in pending.result:
                    # Per-job failure isolated inside update_batch: a ValueError is a real
                    # validation rejection (the job's own 409); anything else is a genuinely
                    # unexpected fault — surface it as 500 so it's distinguishable in logs
                    # (the hook retries neither status).
                    pending.error = pending.result["error"]
                    if not pending.error.startswith("ValueError:"):
                        pending.error_status = 500
                    pending.result = None
                pending.done.set()


def _validate_and_split(trainer, pendings: list["_Pending"]) -> list["_Pending"]:
    """Per-job validation on rank 0 BEFORE broadcasting, so a malformed job 409s its own
    caller without poisoning the batch. PURE (`validate_job`) — slot claims happen inside
    `update_batch`, identically on every rank."""
    valid: list[_Pending] = []
    for pending in pendings:
        try:
            trainer.validate_job(pending.job)
        except ValueError as e:
            pending.error = str(e)
            pending.done.set()
            continue
        valid.append(pending)
    return valid


def build_app_v2(config: TTTServiceConfig, trainer, work_queue: Queue) -> FastAPI:
    """The rank-0 app: enqueue jobs, wait for the work loop's results, drive engine adapter
    loads. `trainer` is only touched for validation here — execution happens in the work
    loop on all ranks."""
    from prime_rl.ttt.trainer_v2 import UpdateJob

    logger = get_logger()
    batch_queue: Queue[_Pending] = Queue()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(timeout=120.0)
        # Per-rollout locks order /update and /release: an update holds its rollout's lock
        # across enqueue→result→adapter load so a concurrent release can't unload+free the
        # slot between train and engine load (orphaned adapter in vLLM otherwise).
        # They also serialize duplicate-replay retries: an orphaned 503 pending may still
        # complete in the work loop WHILE a retry waits on the lock, so the duplicate check
        # lives inside the work loop (which sees the post-completion slot version) and must
        # NOT be a pre-read here in the HTTP layer — a pre-read would race that completion.
        app.state.rollout_locks = defaultdict(asyncio.Lock)
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
        for url in config.inference_admin_urls:
            response = await app.state.http.post(
                f"{url.rstrip('/')}/load_lora_adapter",
                json={"lora_name": adapter_name, "lora_path": ckpt_path},
            )
            response.raise_for_status()

    async def unload_adapter(adapter_name: str) -> None:
        for url in config.inference_admin_urls:
            try:
                response = await app.state.http.post(
                    f"{url.rstrip('/')}/v1/unload_lora_adapter",
                    json={"lora_name": adapter_name},
                )
                if response.status_code // 100 != 2:
                    # Best-effort: a not-loaded 4xx is expected on release retries — warn,
                    # never raise (the slot is freed regardless).
                    logger.warning(
                        f"ttt: unload of {adapter_name} on {url} returned {response.status_code}: {response.text[:200]}"
                    )
            except httpx.HTTPError:
                logger.warning(f"ttt: unload of {adapter_name} failed on {url}")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "adapters": len(trainer.slots), "free_slots": len(trainer.free_idxs)}

    @app.post("/update")
    async def update(request: UpdateRequest) -> UpdateResponse:
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
        async with app.state.rollout_locks[request.rollout_id]:
            pending = _Pending(job=job)
            batch_queue.put(pending)
            finished = await asyncio.to_thread(pending.done.wait, _RESULT_WAIT_SECONDS)
            if not finished:
                raise HTTPException(status_code=503, detail="TTT work loop did not answer in time")
            if pending.error is not None:
                # 409: deterministic per-job validation errors; 500: unexpected (defensive —
                # non-deterministic failures crash the rank, so this should be unreachable).
                raise HTTPException(status_code=pending.error_status, detail=pending.error)
            try:
                await load_adapter(request.adapter_name, pending.result["ckpt_path"])
            except httpx.HTTPError as e:
                raise HTTPException(status_code=502, detail=f"adapter load failed: {e}") from e
            return UpdateResponse(**pending.result)

    @app.post("/release")
    async def release(request: ReleaseRequest) -> dict:
        async with app.state.rollout_locks[request.rollout_id]:
            ack = _ReleaseAck()
            work_queue.put(("release", request.rollout_id, ack))
            finished = await asyncio.to_thread(ack.done.wait, _RESULT_WAIT_SECONDS)
            if not finished:
                raise HTTPException(status_code=503, detail="TTT work loop did not answer in time")
            # Unload UNCONDITIONALLY: a client retry after a lost response finds
            # had_slot=False (the first attempt already freed the slot) but the engine
            # unload may never have run — gating it on had_slot would leak the adapter in
            # vLLM until restart, permanently eating one of max_loras. unload_adapter is
            # idempotent (a not-loaded name is caught and warn-logged).
            await unload_adapter(request.adapter_name)
            # had_slot comes from the work order's result — a pre-read of trainer.slots
            # could race the work loop's own mutation of the registry.
            return {"released": ack.had_slot}

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
        get_logger().info(f"TTT v2 serving on {config.host}:{config.port} ({world})")

    _work_loop(trainer, work_queue, world, ctrl_pg)
