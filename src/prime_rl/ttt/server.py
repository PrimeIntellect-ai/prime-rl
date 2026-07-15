"""The TTT service's HTTP surface (v1): FastAPI over `TTTTrainer`.

`POST /update` runs one blocking TTT update then (re)loads the adapter on every inference
server; `POST /release` drops the rollout's state and unloads the adapter; `GET /health`
is readiness. Training is serialized (the adapters share one PEFT wrapper).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prime_rl.configs.ttt import TTTServiceConfig
from prime_rl.ttt.identity import validate_adapter_name, validate_rollout_id
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.ttt.trainer import TTTTrainer


class UpdateRequest(BaseModel):
    rollout_id: str
    adapter_name: str
    token_ids: list[int]
    loss_mask: list[bool]
    seq_no: int
    qa_pairs: list[dict] | None = None
    """Cartridges-style Q&A pairs (`{type, question, answer}` text), rendered and trained
    by the service with the base model's chat template, loss on the answers."""
    train_rollout: bool = True
    """Whether the raw branch sequence itself joins the training set (False = Q&A only)."""
    system_prompt: str | None = None
    """The rollout's system prompt: each standalone Q&A rendering is conditioned on it
    (loss-masked) so pairs are learned in the same frame the rollout ran under."""
    tools: list[dict] | None = None
    """The rollout's tool schemas (OpenAI wire format), passed to the chat template's
    `tools=` so tool lessons are learned next to the tool descriptions (loss-masked)."""


class UpdateResponse(BaseModel):
    version: int
    loss: float
    ckpt_path: str
    num_loss_tokens: int


class ReleaseRequest(BaseModel):
    rollout_id: str
    adapter_name: str


def build_app(config: TTTServiceConfig, trainer: "TTTTrainer | None" = None) -> FastAPI:
    """The service app. `trainer` is injectable for tests (a fake, or a CPU trainer on a
    tiny model); None loads the real one (torch/transformers/peft) at startup."""
    logger = get_logger()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if trainer is None:
            from prime_rl.ttt.trainer import TTTTrainer

            app.state.trainer = TTTTrainer(config)
        else:
            app.state.trainer = trainer
        app.state.semaphore = asyncio.Semaphore(config.max_concurrent_updates)
        app.state.train_lock = asyncio.Lock()  # one update in the trainer at a time
        app.state.http = httpx.AsyncClient(timeout=config.admin_timeout_seconds)
        try:
            yield
        finally:
            await app.state.http.aclose()

    app = FastAPI(lifespan=lifespan)

    async def load_adapter(adapter_name: str, ckpt_path: str) -> None:
        """(Re)load the adapter on every inference server. prime-rl's `/load_lora_adapter`
        wrapper forces an in-place reload for a same-name adapter, so version k+1 replaces
        k under the same name (the rollout side salts the prefix cache per version)."""
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
        return {"status": "ok", "adapters": len(app.state.trainer.adapters)}

    @app.post("/update")
    async def update(request: UpdateRequest) -> UpdateResponse:
        try:
            validate_rollout_id(request.rollout_id)
            validate_adapter_name(request.adapter_name)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        async with app.state.semaphore:
            async with app.state.train_lock:
                try:
                    # The update is GPU-bound; a worker thread keeps the event loop serving
                    # health checks and other rollouts' queuing.
                    result = await asyncio.to_thread(
                        app.state.trainer.update,
                        request.rollout_id,
                        request.adapter_name,
                        request.token_ids,
                        request.loss_mask,
                        request.seq_no,
                        request.qa_pairs,
                        request.train_rollout,
                        request.system_prompt,
                        request.tools,
                    )
                except ValueError as e:
                    raise HTTPException(status_code=409, detail=str(e)) from e
                # Load inside the train_lock — same orphaned-adapter race as v2's
                # per-rollout locks (see server_v2.build_app_v2 lifespan).
                try:
                    await load_adapter(request.adapter_name, result["ckpt_path"])
                except httpx.HTTPError as e:
                    raise HTTPException(status_code=502, detail=f"adapter load failed: {e}") from e
        return UpdateResponse(**result)

    @app.post("/release")
    async def release(request: ReleaseRequest) -> dict:
        try:
            validate_rollout_id(request.rollout_id)
            validate_adapter_name(request.adapter_name)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        async with app.state.train_lock:
            state = app.state.trainer.release(request.rollout_id)
        # Unload UNCONDITIONALLY: a retry after a lost response finds no state, but the
        # first attempt's engine unload may never have run (unload is idempotent).
        await unload_adapter(request.adapter_name)
        return {"released": state is not None}

    return app


def run_server(config: TTTServiceConfig) -> None:
    import uvicorn

    uvicorn.run(
        build_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log.level.lower(),
    )
