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
        super().__init__("BlenderGym Score Service", gpu_pool, service_id="score")
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self.scorer: CLIPScorer | None = None
        self._scorer_lock = asyncio.Lock()
        self.app.add_api_route("/score", self.score, methods=["POST"])

    async def on_startup(self) -> None:
        self.router = SemaphoreRouter(self.gpu_pool, max_concurrent=2)

    async def _ensure_scorer(self) -> CLIPScorer:
        if self.scorer is not None:
            return self.scorer
        async with self._scorer_lock:
            if self.scorer is None:
                logger.info("Loading CLIP model (first request)...")
                self.scorer = await asyncio.to_thread(
                    CLIPScorer, self.gpu_pool, self.clip_model, self.clip_pretrained
                )
                logger.info("CLIP model loaded")
            return self.scorer

    async def on_shutdown(self) -> None:
        pass

    async def health_detail(self) -> dict:
        return {
            "gpus": len(self.gpu_pool),
            "model": self.clip_model,
            "scorer_loaded": self.scorer is not None,
        }

    async def score(self, req: ScoreRequest) -> ScoreResponse:
        scorer = await self._ensure_scorer()
        async with self.router.lease() as gpu_id:
            t0 = time.monotonic()
            sim = await asyncio.to_thread(
                scorer.score_on_gpu, req.image_a, req.image_b, gpu_id
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
    parser.add_argument("--log-config", default=None, help="Path to JSON log config")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    svc = ScoreService(
        gpu_pool=BaseService.parse_gpu_pool(args.gpu_pool),
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    svc.run(args.port, log_config=args.log_config)


if __name__ == "__main__":
    main()
