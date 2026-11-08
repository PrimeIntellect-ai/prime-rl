"""The eval's control API: read and set its episode concurrency while it runs."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from prime_rl.orchestrator.concurrency import ConcurrencyController


class Concurrency(BaseModel):
    target: int | None
    """The manual cap, or None while the adaptive controller sizes it."""

    cap: int
    """The in-flight episode cap in effect."""

    inflight: int
    """Episodes in flight. Above ``cap`` after a cut until enough of them finish."""


class ConcurrencyRequest(BaseModel):
    target: int | None = Field(ge=1)
    """Pin the cap here, or None to hand it back to the adaptive controller."""


def create_app(concurrency: ConcurrencyController) -> FastAPI:
    app = FastAPI(title="prime-rl eval control")

    @app.get("/v1/concurrency")
    async def get_concurrency() -> Concurrency:
        return _to_concurrency(concurrency)

    @app.put("/v1/concurrency")
    async def set_concurrency(request: ConcurrencyRequest) -> Concurrency:
        concurrency.set_target(request.target)
        return _to_concurrency(concurrency)

    return app


def _to_concurrency(concurrency: ConcurrencyController) -> Concurrency:
    return Concurrency(target=concurrency.target, cap=concurrency.max_inflight, inflight=concurrency.get_inflight())
