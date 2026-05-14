"""Score Service: FastAPI server for CLIP cosine similarity scoring.

In-process service: per-GPU CLIP models are loaded in the FastAPI process
at startup and never migrated. GPU routing uses SemaphoreRouter with
max_concurrent=2 for back-pressure control.

Usage:
    python -m blendergym.services.score.server \\
        --port 8421 --gpu-pool 0,1,2,3,4,5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from pydantic import BaseModel

from ..base import BaseService
from ..semaphore_router import SemaphoreRouter
from .clip_scorer import CLIPScorer

logger = logging.getLogger(__name__)


class ScoreRequest(BaseModel):
    image_a: str
    image_b: str


class ScoreResponse(BaseModel):
    similarity: float
    gpu_id: int = -1
    duration_s: float = 0


class ScoreService(BaseService):
    """In-process service with per-GPU CLIP models."""

    def __init__(
        self,
        gpu_pool: list[int],
        clip_model: str,
        clip_pretrained: str,
    ):
        super().__init__("BlenderGym Score Service", gpu_pool)
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.app.add_api_route("/score", self.score, methods=["POST"])

    async def on_startup(self) -> None:
        self.router = SemaphoreRouter(self.gpu_pool, max_concurrent=2)
        self.scorer = CLIPScorer(
            self.gpu_pool, self.clip_model, self.clip_pretrained
        )

    async def on_shutdown(self) -> None:
        pass

    async def health_detail(self) -> dict:
        return {"gpus": len(self.gpu_pool), "model": self.clip_model}

    async def score(self, req: ScoreRequest) -> ScoreResponse:
        async with self.router.lease() as gpu_id:
            t0 = time.monotonic()
            sim = await asyncio.to_thread(
                self.scorer.score_on_gpu, req.image_a, req.image_b, gpu_id
            )
            return ScoreResponse(
                similarity=sim,
                gpu_id=gpu_id,
                duration_s=time.monotonic() - t0,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BlenderGym Score Service")
    BaseService.common_cli_args(parser, default_port=8421)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    svc = ScoreService(
        gpu_pool=BaseService.parse_gpu_pool(args.gpu_pool),
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    svc.run(args.port)


if __name__ == "__main__":
    main()
