"""Engines that pull their weights (``weight_broadcast.pull``): the engine moves itself onto each policy version
the trainer publishes to the broadcast dir, instead of waiting for the orchestrator to pause it and push one.

While serving, it decodes the sparse delta of the newest version the trainer is still writing. Once a version's
broadcast finished, it pauses with the running requests kept, applies the newest finished version, labels the
tokens sampled from then on with it, and resumes. A version it skipped needs nothing extra: the worker replays the
deltas since the last anchor. A trainer that restarts from a checkpoint drops the versions beyond it and publishes
that one again, so the engine follows the newest finished version also back. The orchestrator only reads the
version the engine serves (``GET /weight_version``).
"""

import asyncio
import time
from pathlib import Path

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

from prime_rl.utils.broadcasts import FINISHED_MARKER, marker_exists
from prime_rl.utils.pathing import get_all_ckpt_steps, get_step_path
from prime_rl.utils.sparse_weights import DELTA_DIR, MANIFEST

logger = init_logger("vllm.inference.vllm.weight_pull")

POLL_INTERVAL_S = 0.1
RETRY_INTERVAL_S = 1.0


class WeightPuller:
    def __init__(self, client: EngineClient, broadcast_dir: Path):
        self.client = client
        self.broadcast_dir = broadcast_dir
        self.served = -1
        """The version the engine serves, -1 until it applied the trainer's first broadcast."""
        self.prefetched: int | None = None

    def step_dir(self, step: int) -> Path:
        return get_step_path(self.broadcast_dir, step)

    async def run(self) -> None:
        logger.info(f"Pulling policy weights from {self.broadcast_dir}")
        while True:
            try:
                await self.poll()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Pulling policy weights failed; retrying")
                await asyncio.sleep(RETRY_INTERVAL_S)
            await asyncio.sleep(POLL_INTERVAL_S)

    def newest_finished(self) -> int | None:
        steps = reversed(get_all_ckpt_steps(self.broadcast_dir))
        return next((step for step in steps if marker_exists(self.step_dir(step) / FINISHED_MARKER)), None)

    async def poll(self) -> None:
        finished = self.newest_finished()
        if finished is not None and finished != self.served:
            await self.apply(finished)
            return
        newest = max(get_all_ckpt_steps(self.broadcast_dir), default=-1)
        if newest > self.served and newest != self.prefetched:
            if marker_exists(self.step_dir(newest) / DELTA_DIR / MANIFEST):
                await self.prefetch(newest)

    async def prefetch(self, step: int) -> None:
        """Read and decode v{step}'s sparse delta while the engine serves, so its pause only applies it."""
        self.prefetched = step
        await self.client.collective_rpc("prefetch_weights_from_path", args=(self.step_dir(step).as_posix(),))
        while not all(await self.client.collective_rpc("prefetch_weights_done")):
            await asyncio.sleep(0.05)

    async def apply(self, step: int) -> None:
        """Move the engine onto v{step}. The running requests stay in the engine across the pause and go on
        sampling with the new weights; new requests queue. A failed update leaves the weights unknown, so the engine
        stays paused and retries with the newest finished version until one applies."""
        start = time.perf_counter()
        await self.client.pause_generation(mode="keep", clear_cache=False)
        try:
            while True:
                try:
                    await self.client.collective_rpc("update_weights_from_path", args=(self.step_dir(step).as_posix(),))
                    break
                except Exception:
                    logger.exception(f"Applying policy v{step} failed; retrying while paused")
                    await asyncio.sleep(RETRY_INTERVAL_S)
                    newest = self.newest_finished()
                    step = step if newest is None else newest
            await self.client.update_weight_version(str(step))
        finally:
            await self.client.resume_generation()
        self.served = step
        logger.info(f"Pulled policy v{step}, paused for {time.perf_counter() - start:.3f}s")
