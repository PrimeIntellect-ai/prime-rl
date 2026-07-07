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
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from queue import Empty, Queue

import httpx
from fastapi import FastAPI, HTTPException

from prime_rl.configs.ttt import TTTServiceConfig
from prime_rl.ttt.server import ReleaseRequest, UpdateRequest, UpdateResponse
from prime_rl.utils.logger import get_logger


@dataclass
class _Pending:
    """One queued update job + the event/result plumbing back to its HTTP caller."""

    job: object  # UpdateJob
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: str | None = None


def _work_loop(trainer, work_queue: Queue, world) -> None:
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
            dist.broadcast_object_list(payload, src=0)
        kind, data = payload
        if kind == "stop":
            return
        if kind == "release":
            trainer.release(data)
            if world.is_master:
                order[2].set()  # release ack event
            continue
        # kind == "update": data = list[UpdateJob]; on rank 0, order[2] = list[_Pending]
        jobs = data
        try:
            results = trainer.update_batch(jobs)
        except Exception as e:
            # A collective failure (OOM, NCCL) poisons the whole batch; per-job validation
            # errors are handled inside update_batch/prepare via individual exclusion below.
            logger.exception("ttt v2: batch failed")
            if world.is_master:
                for pending in order[2]:
                    pending.error = f"{type(e).__name__}: {e}"
                    pending.done.set()
            continue
        if world.is_master:
            for pending in order[2]:
                pending.result = results.get(pending.job.rollout_id)
                if pending.result is None:
                    pending.error = "job produced no result"
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
                await app.state.http.post(
                    f"{url.rstrip('/')}/v1/unload_lora_adapter",
                    json={"lora_name": adapter_name},
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
        pending = _Pending(job=job)
        batch_queue.put(pending)
        await asyncio.to_thread(pending.done.wait)
        if pending.error is not None:
            raise HTTPException(status_code=409, detail=pending.error)
        try:
            await load_adapter(request.adapter_name, pending.result["ckpt_path"])
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"adapter load failed: {e}") from e
        return UpdateResponse(**pending.result)

    @app.post("/release")
    async def release(request: ReleaseRequest) -> dict:
        had_slot = request.rollout_id in trainer.slots
        ack = threading.Event()
        work_queue.put(("release", request.rollout_id, ack))
        await asyncio.to_thread(ack.wait)
        if had_slot:
            await unload_adapter(request.adapter_name)
        return {"released": had_slot}

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
    trainer = TTTTrainerV2(config)
    work_queue: Queue = Queue()

    if world.is_master:
        app = build_app_v2(config, trainer, work_queue)
        server = uvicorn.Server(
            uvicorn.Config(app, host=config.host, port=config.port, log_level=config.log.level.lower())
        )
        http_thread = threading.Thread(target=server.run, daemon=True)
        http_thread.start()
        get_logger().info(f"TTT v2 serving on {config.host}:{config.port} ({world})")

    _work_loop(trainer, work_queue, world)
