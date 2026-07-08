"""The TTT service's HTTP surface.

FastAPI app over the `TTTTrainer`:

- `POST /update` — one blocking TTT update for a rollout (gradient step(s) + checkpoint),
  then a `/load_lora_adapter` (in-place reload, prime-rl's wrapper) on every inference
  server. The rollout side blocks on this call, so the response is the ack.
- `POST /release` — drop the rollout's training state and unload its adapter from the
  engines; checkpoints stay on disk (unless `keep_checkpoints=false`).
- `GET /health` — readiness (the model is loaded when the app is serving).

Training is serialized through an executor (`max_concurrent_updates` bounds distinct
rollouts queuing; the trainer itself runs one update at a time — the adapters share one
PEFT wrapper). Per-rollout ordering is guaranteed by the rollout side blocking per update.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prime_rl.configs.ttt import TTTServiceConfig
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
        app.state.http = httpx.AsyncClient(timeout=120.0)
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
                await app.state.http.post(
                    f"{url.rstrip('/')}/v1/unload_lora_adapter",
                    json={"lora_name": adapter_name},
                )
            except httpx.HTTPError:
                logger.warning(f"ttt: unload of {adapter_name} failed on {url}")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "adapters": len(app.state.trainer.adapters)}

    @app.post("/update")
    async def update(request: UpdateRequest) -> UpdateResponse:
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
                # load_adapter stays inside the train_lock so a concurrent /release can't
                # unload+pop between train and load (orphaned adapter in vLLM). This
                # serializes engine loads — acceptable: max_concurrent_updates defaults to 1.
                try:
                    await load_adapter(request.adapter_name, result["ckpt_path"])
                except httpx.HTTPError as e:
                    raise HTTPException(status_code=502, detail=f"adapter load failed: {e}") from e
        return UpdateResponse(**result)

    @app.post("/release")
    async def release(request: ReleaseRequest) -> dict:
        async with app.state.train_lock:
            state = app.state.trainer.release(request.rollout_id)
        if state is not None and state.version > 0:
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
