"""Engines that pull their weights (``weight_broadcast.pull``): the engine moves itself onto each policy version
the trainer publishes to the broadcast dir, instead of waiting for the orchestrator to pause it and push one.

While serving, it decodes the sparse delta of the newest version the trainer is still writing. Once a version's
broadcast finished, it pauses with the running requests kept, applies the newest finished version, labels the
tokens sampled from then on with it, and resumes. A version it skipped needs nothing extra: the worker replays the
deltas since the last anchor. A trainer that restarts from a checkpoint drops the versions beyond it and publishes
that one again, so the engine follows the newest finished version also back. The orchestrator only reads the
version the engine serves (``GET /weight_version``).

On several nodes every DP rank is its own server. The ranks of an expert-parallel DP group share every forward pass,
so they switch together: the group's first rank pulls for all of them, through their admin routes. Groups switch
independently, each as soon as it is ready, so a slow or dead group holds up no other. Under disaggregated P/D a
request can then be prefilled on one version and decoded on the next, as a running request already is across any
pause.
"""

import asyncio
import time
from pathlib import Path
from typing import Protocol

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

from prime_rl.utils.broadcasts import FINISHED_MARKER, marker_exists
from prime_rl.utils.pathing import get_all_ckpt_steps, get_step_path
from prime_rl.utils.sparse_weights import DELTA_DIR, MANIFEST, revalidate

logger = init_logger("vllm.inference.vllm.weight_pull")

POLL_INTERVAL_S = 0.1
RETRY_INTERVAL_S = 1.0


class Engines(Protocol):
    """The engines a puller moves onto each version, together."""

    async def ready(self) -> None:
        """Wait until every engine serves."""

    async def prefetch(self, weights_dir: Path) -> None:
        """Read and decode a sparse delta while the engines serve."""

    async def pause(self, step: int) -> None:
        """Pause the engines for the update to v{step}, keeping the running requests."""

    async def load(self, weights_dir: Path, step: int) -> None:
        """Move the paused engines onto v{step} and label the tokens they sample from then on with it."""

    async def resume(self) -> None: ...


class LocalEngine:
    """The engine of this server, reached in process."""

    def __init__(self, client: EngineClient):
        self.client = client

    async def ready(self) -> None:
        pass

    async def prefetch(self, weights_dir: Path) -> None:
        await self.client.collective_rpc("prefetch_weights_from_path", args=(weights_dir.as_posix(),))
        while not all(await self.client.collective_rpc("prefetch_weights_done")):
            await asyncio.sleep(0.05)

    async def pause(self, step: int) -> None:
        await self.client.pause_generation(mode="keep", clear_cache=False)

    async def load(self, weights_dir: Path, step: int) -> None:
        await self.client.collective_rpc("update_weights_from_path", args=(weights_dir.as_posix(),))
        await self.client.update_weight_version(str(step))

    async def resume(self) -> None:
        await self.client.resume_generation()


class EngineGroup:
    """The ranks of one DP group (``weight_broadcast.pull_group``), this server's own included, reached through their
    admin routes: all pause, all load the version, all resume."""

    def __init__(self, admin_urls: list[str], admin_token_var: str):
        from prime_rl.configs.shared import VLLMClientConfig
        from prime_rl.orchestrator.clients import AdminPlane

        # The engines' client config, as the orchestrator's: their API key, and the admin token to send
        config = VLLMClientConfig(base_url=admin_urls[0], admin_base_url=admin_urls, admin_token_var=admin_token_var)
        self.admin_plane = AdminPlane(config)

    async def ready(self) -> None:
        from prime_rl.orchestrator.clients import check_health

        await check_health(self.admin_plane.clients)

    async def prefetch(self, weights_dir: Path) -> None:
        await self.admin_plane.prefetch_weights(weights_dir)

    async def pause(self, step: int) -> None:
        await self.admin_plane.pause(step)

    async def load(self, weights_dir: Path, step: int) -> None:
        await self.admin_plane.transfer(weights_dir, step)

    async def resume(self) -> None:
        await self.admin_plane.resume()


class WeightPuller:
    def __init__(self, engines: Engines, broadcast_dir: Path):
        self.engines = engines
        self.broadcast_dir = broadcast_dir
        self.served = -1
        """The version the engines serve, -1 until they applied the trainer's first broadcast."""
        self.prefetched: int | None = None

    def step_dir(self, step: int) -> Path:
        return get_step_path(self.broadcast_dir, step)

    async def run(self) -> None:
        logger.info(f"Pulling policy weights from {self.broadcast_dir}")
        try:
            await self.engines.ready()
        except Exception:
            logger.exception("The engines to pull policy weights for never came up")
            raise
        while True:
            try:
                await self.poll()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Pulling policy weights failed; retrying")
                await asyncio.sleep(RETRY_INTERVAL_S)
            await asyncio.sleep(POLL_INTERVAL_S)

    def steps(self) -> list[int]:
        """The published steps, oldest first. Reopening the broadcast dir first makes an NFS client list a step
        another host just published, instead of answering from its cached listing."""
        revalidate(self.broadcast_dir)
        return get_all_ckpt_steps(self.broadcast_dir)

    def newest_finished(self) -> int | None:
        return next(
            (step for step in reversed(self.steps()) if marker_exists(self.step_dir(step) / FINISHED_MARKER)), None
        )

    async def poll(self) -> None:
        finished = self.newest_finished()
        if finished is not None and finished != self.served:
            await self.apply(finished)
            return
        newest = max(self.steps(), default=-1)
        if newest > self.served and newest != self.prefetched:
            if marker_exists(self.step_dir(newest) / DELTA_DIR / MANIFEST):
                await self.prefetch(newest)

    async def prefetch(self, step: int) -> None:
        """Read and decode v{step}'s sparse delta while the engines serve, so their pause only applies it."""
        self.prefetched = step
        await self.engines.prefetch(self.step_dir(step))

    async def apply(self, step: int) -> None:
        """Move the engines onto v{step}. The running requests stay in the engines across the pause and go on
        sampling with the new weights; new requests queue. A failed update leaves the weights unknown, so the engines
        stay paused and retry with the newest finished version until one applies."""
        start = time.perf_counter()
        await self.engines.pause(step)
        try:
            while True:
                try:
                    await self.engines.load(self.step_dir(step), step)
                    break
                except Exception:
                    logger.exception(f"Applying policy v{step} failed; retrying while paused")
                    await asyncio.sleep(RETRY_INTERVAL_S)
                    newest = self.newest_finished()
                    step = step if newest is None else newest
        finally:
            await self.engines.resume()
        self.served = step
        logger.info(f"Pulled policy v{step}, paused for {time.perf_counter() - start:.3f}s")
