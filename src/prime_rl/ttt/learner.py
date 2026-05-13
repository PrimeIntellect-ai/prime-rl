from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Literal

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from prime_rl.ttt.lora_engine import HookedLoRAEngine
from prime_rl.utils.logger import setup_logger


class PrepareTurnRequest(BaseModel):
    session_id: str
    turn_idx: int
    model: str
    prompt_ids: list[int]
    new_prompt_ids: list[int]
    token_role: Literal["completion_initial_prompt", "prompt_environment"]


class CompleteTurnRequest(BaseModel):
    session_id: str
    turn_idx: int
    model: str
    completion_ids: list[int]
    completion_logprobs: list[float] = []
    prepare_version: int | None = None


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

    def lock_for(session_id: str) -> asyncio.Lock:
        lock = locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            locks[session_id] = lock
        return lock

    def activate_session(session_id: str):
        session = engine.get_or_create_session(session_id)
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
        weight_dir = Path(request.weight_dir) if request.weight_dir else None
        await asyncio.to_thread(engine.load_base_weights, weight_dir, request.step)
        return {"status": "ok", "base_step": engine.base_step, "sessions": len(engine.sessions)}

    @app.post("/start_session")
    async def start_session(request: FinishSessionRequest) -> dict[str, Any]:
        session = engine.get_or_create_session(request.session_id)
        offload_session(session)
        return {"status": "ok", "session_id": request.session_id}

    @app.post("/prepare_turn")
    async def prepare_turn(request: PrepareTurnRequest) -> dict[str, Any]:
        async with lock_for(request.session_id):
            session = activate_session(request.session_id)
            try:
                kind: Literal["prompt", "completion"] = (
                    "completion" if request.token_role == "completion_initial_prompt" else "prompt"
                )
                token_ids = request.new_prompt_ids
                loss = await asyncio.to_thread(engine.train_tokens, session, kind, token_ids)
                adapter_name = (
                    f"ttt-{request.session_id[:12]}-t{request.turn_idx}"
                    f"-p{session.prompt_version}-c{session.completion_version}-b{engine.base_step}"
                )
                meta = await engine.materialize(
                    session,
                    name=adapter_name,
                    kinds=("prompt", "completion"),
                    load_into_vllm=True,
                    adapter_kind="combined",
                    turn_idx=request.turn_idx,
                )
                return {
                    **meta,
                    "status": "ok",
                    "version": request.turn_idx,
                    "token_role": request.token_role,
                    "trained_lora": kind,
                    "trained_token_count": len(token_ids),
                    "loss": loss,
                }
            finally:
                offload_session(session)

    @app.post("/complete_turn")
    async def complete_turn(request: CompleteTurnRequest) -> dict[str, Any]:
        async with lock_for(request.session_id):
            session = activate_session(request.session_id)
            try:
                loss = await asyncio.to_thread(engine.train_tokens, session, "completion", request.completion_ids)
                prompt_adapter_name = (
                    f"ttt-{request.session_id[:12]}-final-p{session.prompt_version}-b{engine.base_step}"
                )
                final_prompt = await engine.materialize(
                    session,
                    name=prompt_adapter_name,
                    kinds=("prompt",),
                    load_into_vllm=False,
                    adapter_kind="final_prompt",
                    turn_idx=request.turn_idx,
                )
                session.final_prompt_adapter = final_prompt
                combined = session.combined_adapters()
                if combined:
                    await engine.unload_vllm_adapter(str(combined[-1]["adapter_name"]))
                return {
                    "status": "ok",
                    "base_step": engine.base_step,
                    "completion_version": session.completion_version,
                    "trained_token_count": len(request.completion_ids),
                    "loss": loss,
                    "final_prompt_adapter": final_prompt,
                }
            finally:
                offload_session(session)

    @app.post("/finish_session")
    async def finish_session(request: FinishSessionRequest) -> dict[str, Any]:
        session = engine.sessions.pop(request.session_id, None)
        await engine.unload_session_combined_adapters(session)
        locks.pop(request.session_id, None)
        return {
            "status": "ok",
            "session_id": request.session_id,
            "final_prompt_adapter": session.final_prompt_adapter if session else None,
        }

    @app.post("/abort_session")
    async def abort_session(request: FinishSessionRequest) -> dict[str, Any]:
        session = engine.sessions.pop(request.session_id, None)
        await engine.unload_session_combined_adapters(session)
        locks.pop(request.session_id, None)
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
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        dtype=_dtype(args.dtype),
        vllm_admin_base_urls=args.vllm_admin_base_url,
        load_adapters_into_vllm=not args.no_load_adapters_into_vllm,
        unload_vllm_adapters=not args.no_unload_vllm_adapters,
    )
    uvicorn.run(create_app(engine, session_offload=args.session_offload), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
