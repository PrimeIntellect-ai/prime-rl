from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prime_rl.utils.logger import get_logger, setup_logger

if TYPE_CHECKING:
    from prime_rl.ttt.lora_engine import HookedLoRAEngine


class PrepareTurnRequest(BaseModel):
    session_id: str
    turn_idx: int
    model: str
    prompt_ids: list[int]
    new_token_ids: list[int]
    new_token_replay_mask: list[bool] | None = None


class CompleteTurnRequest(BaseModel):
    session_id: str
    turn_idx: int
    model: str
    completion_ids: list[int]
    completion_logprobs: list[float] = []
    prepare_version: int | None = None
    adapter_name: str | None = None


class UpdateBaseWeightsRequest(BaseModel):
    weight_dir: str | None = None
    step: int = 0


class FinishSessionRequest(BaseModel):
    session_id: str


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def create_app(engine: HookedLoRAEngine, session_offload: Literal["none", "cpu_after_request"]) -> FastAPI:
    app = FastAPI(title="Prime-RL TTT learner")
    locks: dict[str, asyncio.Lock] = {}
    engine_lock = asyncio.Lock()
    vllm_admin_lock = asyncio.Lock()
    vllm_admin_pause_state_lock = asyncio.Lock()
    vllm_admin_paused = False
    logger = get_logger()

    def elapsed(start: float) -> float:
        return time.perf_counter() - start

    def lock_for(session_id: str) -> asyncio.Lock:
        lock = locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            locks[session_id] = lock
        return lock

    async def activate_session(session_id: str):
        async with engine_lock:
            try:
                session = engine.get_or_create_session(session_id)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if session_offload == "cpu_after_request":
            session.to_device(engine.device)
        return session

    def offload_session(session) -> None:
        if session_offload == "cpu_after_request":
            session.to_cpu()
            if engine.device.type == "cuda":
                torch.cuda.empty_cache()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "base_step": engine.base_step, "sessions": len(engine.sessions)}

    @app.post("/update_base_weights")
    async def update_base_weights(request: UpdateBaseWeightsRequest) -> dict[str, Any]:
        start = time.perf_counter()
        weight_dir = Path(request.weight_dir) if request.weight_dir else None
        async with engine_lock:
            await asyncio.to_thread(engine.load_base_weights, weight_dir, request.step)
        logger.info(
            f"TTT timing update_base_weights step={request.step} weight_dir={weight_dir} "
            f"elapsed={elapsed(start):.3f}s sessions={len(engine.sessions)}"
        )
        return {"status": "ok", "base_step": engine.base_step, "sessions": len(engine.sessions)}

    @app.post("/pause_vllm_admin")
    async def pause_vllm_admin() -> dict[str, Any]:
        nonlocal vllm_admin_paused
        start = time.perf_counter()
        async with vllm_admin_pause_state_lock:
            if not vllm_admin_paused:
                await vllm_admin_lock.acquire()
                vllm_admin_paused = True
        logger.info(f"TTT timing pause_vllm_admin elapsed={elapsed(start):.3f}s")
        return {"status": "ok", "paused": True}

    @app.post("/resume_vllm_admin")
    async def resume_vllm_admin() -> dict[str, Any]:
        nonlocal vllm_admin_paused
        start = time.perf_counter()
        async with vllm_admin_pause_state_lock:
            if vllm_admin_paused:
                vllm_admin_paused = False
                vllm_admin_lock.release()
        logger.info(f"TTT timing resume_vllm_admin elapsed={elapsed(start):.3f}s")
        return {"status": "ok", "paused": False}

    @app.post("/start_session")
    async def start_session(request: FinishSessionRequest) -> dict[str, Any]:
        session = await activate_session(request.session_id)
        offload_session(session)
        return {"status": "ok", "session_id": request.session_id}

    @app.post("/prepare_turn")
    async def prepare_turn(request: PrepareTurnRequest) -> dict[str, Any]:
        request_start = time.perf_counter()
        async with lock_for(request.session_id):
            activate_start = time.perf_counter()
            session = await activate_session(request.session_id)
            activate_elapsed = elapsed(activate_start)
            try:
                train_start = time.perf_counter()
                async with engine_lock:
                    train_stats = await engine.append_and_train_with_replay_spans(
                        session,
                        request.new_token_ids,
                        request.new_token_replay_mask,
                        request.turn_idx,
                    )
                    meta = session.latest_adapter
                    if session.version > 0 and meta is None:
                        adapter_name = (
                            f"ttt-{request.session_id[:12]}-t{request.turn_idx}-v{session.version}-b{engine.base_step}"
                        )
                        meta = await engine.materialize(
                            session,
                            name=adapter_name,
                            load_into_vllm=False,
                            turn_idx=request.turn_idx,
                        )
                if meta is not None:
                    async with vllm_admin_lock:
                        await engine.ensure_vllm_loaded(meta)
                train_elapsed = elapsed(train_start)
                meta = dict(meta or {})
                logger.info(
                    f"TTT timing prepare_turn session={request.session_id} turn={request.turn_idx} "
                    f"new_tokens={len(request.new_token_ids)} prompt_tokens={len(request.prompt_ids)} "
                    f"trained_chunks={train_stats['trained_chunks']} "
                    f"trained_tokens={train_stats['trained_token_count']} "
                    f"pending_tokens={train_stats['pending_token_count']} version={session.version} "
                    f"adapter={meta.get('adapter_name')} activate={activate_elapsed:.3f}s "
                    f"train_materialize={train_elapsed:.3f}s total={elapsed(request_start):.3f}s "
                    f"loss={train_stats['loss']:.6f}"
                )
                return {
                    **meta,
                    "prompt_replay_spans": train_stats.get("prompt_replay_spans", []),
                    "status": "ok",
                    "version": session.version,
                    "trained_token_count": train_stats["trained_token_count"],
                    "trained_chunks": train_stats["trained_chunks"],
                    "pending_token_count": train_stats["pending_token_count"],
                    "loss": train_stats["loss"],
                }
            finally:
                offload_start = time.perf_counter()
                offload_session(session)
                logger.info(
                    f"TTT timing prepare_turn_offload session={request.session_id} "
                    f"turn={request.turn_idx} elapsed={elapsed(offload_start):.3f}s"
                )

    @app.post("/complete_turn")
    async def complete_turn(request: CompleteTurnRequest) -> dict[str, Any]:
        request_start = time.perf_counter()
        async with lock_for(request.session_id):
            activate_start = time.perf_counter()
            session = await activate_session(request.session_id)
            activate_elapsed = elapsed(activate_start)
            try:
                train_start = time.perf_counter()
                async with engine_lock:
                    train_stats = await asyncio.to_thread(engine.append_and_train, session, request.completion_ids)
                train_elapsed = elapsed(train_start)
                unload_elapsed = 0.0
                if request.adapter_name:
                    unload_start = time.perf_counter()
                    async with vllm_admin_lock:
                        await engine.unload_vllm_adapter(request.adapter_name)
                    engine.mark_vllm_unloaded(session, request.adapter_name)
                    unload_elapsed = elapsed(unload_start)
                logger.info(
                    f"TTT timing complete_turn session={request.session_id} turn={request.turn_idx} "
                    f"completion_tokens={len(request.completion_ids)} adapter={request.adapter_name} "
                    f"trained_chunks={train_stats['trained_chunks']} "
                    f"trained_tokens={train_stats['trained_token_count']} "
                    f"pending_tokens={train_stats['pending_token_count']} version={session.version} "
                    f"activate={activate_elapsed:.3f}s train={train_elapsed:.3f}s "
                    f"unload={unload_elapsed:.3f}s total={elapsed(request_start):.3f}s "
                    f"loss={train_stats['loss']:.6f}"
                )
                return {
                    "status": "ok",
                    "base_step": engine.base_step,
                    "version": session.version,
                    "trained_token_count": train_stats["trained_token_count"],
                    "trained_chunks": train_stats["trained_chunks"],
                    "pending_token_count": train_stats["pending_token_count"],
                    "loss": train_stats["loss"],
                }
            finally:
                offload_start = time.perf_counter()
                offload_session(session)
                logger.info(
                    f"TTT timing complete_turn_offload session={request.session_id} "
                    f"turn={request.turn_idx} elapsed={elapsed(offload_start):.3f}s"
                )

    @app.post("/finish_session")
    async def finish_session(request: FinishSessionRequest) -> dict[str, Any]:
        start = time.perf_counter()
        lock = lock_for(request.session_id)
        async with lock:
            async with engine_lock:
                session = engine.sessions.pop(request.session_id, None)
            async with vllm_admin_lock:
                await engine.unload_session_loaded_adapters(session)
        logger.info(
            f"TTT timing finish_session session={request.session_id} "
            f"had_session={session is not None} elapsed={elapsed(start):.3f}s"
        )
        return {
            "status": "ok",
            "session_id": request.session_id,
        }

    @app.post("/abort_session")
    async def abort_session(request: FinishSessionRequest) -> dict[str, Any]:
        start = time.perf_counter()
        lock = lock_for(request.session_id)
        async with lock:
            async with engine_lock:
                session = engine.sessions.pop(request.session_id, None)
            async with vllm_admin_lock:
                await engine.unload_session_loaded_adapters(session)
        logger.info(
            f"TTT timing abort_session session={request.session_id} "
            f"had_session={session is not None} elapsed={elapsed(start):.3f}s"
        )
        return {"status": "ok", "session_id": request.session_id}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--target-modules", nargs="+", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps-per-update", type=int, default=1)
    parser.add_argument("--update-every-tokens", type=int, default=1024)
    parser.add_argument("--max-concurrent-sessions", type=int, default=64)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--vllm-admin-base-url", action="append", default=[])
    parser.add_argument("--no-load-adapters-into-vllm", action="store_true")
    parser.add_argument("--session-offload", default="cpu_after_request", choices=["none", "cpu_after_request"])
    parser.add_argument("--no-unload-vllm-adapters", action="store_true")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)
    logger = get_logger()
    logger.info("Importing TTT LoRA engine")
    from prime_rl.ttt.lora_engine import HookedLoRAEngine

    logger.info("Imported TTT LoRA engine")
    logger.info(
        f"Starting TTT learner model={args.model_name} device={args.device} dtype={args.dtype} "
        f"adapter_dir={args.adapter_dir}"
    )
    engine = HookedLoRAEngine(
        model_name=args.model_name,
        adapter_dir=Path(args.adapter_dir),
        target_modules=args.target_modules,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        steps_per_update=args.steps_per_update,
        update_every_tokens=args.update_every_tokens,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        dtype=_dtype(args.dtype),
        vllm_admin_base_urls=args.vllm_admin_base_url,
        max_concurrent_sessions=args.max_concurrent_sessions,
        load_adapters_into_vllm=not args.no_load_adapters_into_vllm,
        unload_vllm_adapters=not args.no_unload_vllm_adapters,
    )
    logger.info(f"TTT learner engine initialized; serving on {args.host}:{args.port}")
    uvicorn.run(create_app(engine, session_offload=args.session_offload), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
