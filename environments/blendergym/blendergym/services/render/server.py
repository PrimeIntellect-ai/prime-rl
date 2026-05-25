"""Render Service: FastAPI server managing PersistentBlender worker pools.

Subprocess-isolated service: each GPU gets a BlenderPool with one or more
long-lived Blender processes. GPU scheduling uses SemaphoreRouter with
lease-based resource management.

Usage:
    python -m blendergym.services.render.server \\
        --port 8420 --blender-bin /path/to/blender --gpu-pool 0,1,2,3,4,5
"""

from __future__ import annotations

import argparse
import logging
import os
import asyncio
from pathlib import Path

from ..base import BaseService
from ..semaphore_router import SemaphoreRouter
from .persistent_blender import BlenderPool, RenderRequest, RenderResponse

logger = logging.getLogger(__name__)


class RenderService(BaseService):
    """Subprocess-isolated service managing PersistentBlender worker pools."""

    def __init__(
        self,
        gpu_pool: list[int],
        blender_bin: Path,
        pool_size: int,
        cycles_cfg: dict[str, str | int],
    ):
        super().__init__("BlenderGym Render Service", gpu_pool, service_id="render")
        self.blender_bin = blender_bin
        self.pool_size = pool_size
        self.cycles_cfg = cycles_cfg
        self.app.add_api_route("/render", self.render, methods=["POST"])

    async def on_startup(self) -> None:
        for k, v in self.cycles_cfg.items():
            os.environ[f"BLENDERGYM_{k.upper()}"] = str(v)
        self.router = SemaphoreRouter(self.gpu_pool, max_concurrent=self.pool_size)
        self.pools: dict[int, BlenderPool] = {
            g: BlenderPool(g, self.blender_bin, pool_size=self.pool_size)
            for g in self.gpu_pool
        }
        await asyncio.gather(*[pool.wait_ready(timeout=120) for pool in self.pools.values()])

    async def on_shutdown(self) -> None:
        for pool in self.pools.values():
            pool.shutdown()

    async def health_detail(self) -> dict:
        gpus = {
            str(g): {"alive": p.alive, "pool_size": p.pool_size}
            for g, p in self.pools.items()
        }
        return {"gpus": gpus, "_ok": all(s["alive"] for s in gpus.values())}

    async def render(self, req: RenderRequest) -> RenderResponse:
        async with self.router.lease() as gpu_id:
            return await self.pools[gpu_id].render(req)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BlenderGym Render Service")
    BaseService.common_cli_args(parser, default_port=8420)
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--pool-size", type=int, default=1)
    parser.add_argument("--cycles-resolution", type=int, default=256)
    parser.add_argument("--cycles-samples", type=int, default=8)
    parser.add_argument("--cycles-denoiser", default="OPENIMAGEDENOISE")
    parser.add_argument("--cycles-compute-device", default="OPTIX")
    parser.add_argument("--log-config", default=None, help="Path to JSON log config")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    svc = RenderService(
        gpu_pool=BaseService.parse_gpu_pool(args.gpu_pool),
        blender_bin=Path(args.blender_bin),
        pool_size=args.pool_size,
        cycles_cfg={
            "RENDER_RESOLUTION": args.cycles_resolution,
            "CYCLES_SAMPLES": args.cycles_samples,
            "CYCLES_DENOISER": args.cycles_denoiser,
            "CYCLES_COMPUTE_DEVICE": args.cycles_compute_device,
        },
    )
    svc.run(args.port, log_config=args.log_config)


if __name__ == "__main__":
    main()
