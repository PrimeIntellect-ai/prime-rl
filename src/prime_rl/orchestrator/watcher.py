"""WeightWatcher: polls the broadcast dir, advances ``Policy``, notifies observers.

Standalone async task. The watcher does three things:

1. Discovers the next checkpoint step from ``broadcasts/`` (or, equivalently,
   from the NCCL-broadcast in-memory path which writes a ``NCCL_READY`` marker).
2. Calls ``inference.update_weights(weights_path, lora_name, step)`` —
   the inference pool's pause/resume + LoRA / NCCL handshake lives there.
3. Mutates the shared ``Policy`` (version and, on LoRA, model_name) and walks
   the observer list in order so each observer (dispatcher, future plugins) can
   react synchronously (off-policy cancel, eval triggers, etc.).

The watcher always stays at least one step ahead of the trainer: the trainer
broadcasts step ``progress.step - 1``, we adopt anything fresher than what we
already loaded. The dispatcher's barrier (``policy.version`` vs the batcher's
step counter) keeps the in-flight lead bounded.
"""

from __future__ import annotations

import asyncio
import time

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.types import Policy, VersionObserver
from prime_rl.utils.async_utils import safe_cancel
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path, wait_for_path
from prime_rl.utils.utils import get_latest_ckpt_step


class WeightWatcher:
    """``await watcher.start()`` to drive the polling loop until ``stop()``."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        policy: Policy,
        inference: InferencePool,
        observers: list[VersionObserver],
        lora_name: str | None,
        ckpt_step: int = 0,
        poll_interval: float = 1.0,
    ) -> None:
        self.config = config
        self.policy = policy
        self.inference = inference
        self.observers = observers
        self.lora_name = lora_name
        self.ckpt_step = ckpt_step
        self.poll_interval = poll_interval
        self.logger = get_logger()

        # Latency metrics surfaced via ``metrics()`` for the IntervalLogger.
        self.last_update_weights_time: float = 0.0
        self.last_wait_for_ckpt_time: float = 0.0
        self.update_count: int = 0

        self.task: asyncio.Task | None = None
        self.update_lock = asyncio.Lock()
        self.stopped = asyncio.Event()

    async def start(self) -> None:
        """Main poll loop. Runs until ``stop()`` is called."""
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                next_step = self.compute_next_ckpt_step()
                if next_step > self.ckpt_step:
                    await self.apply_policy_update(next_step)
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    def compute_next_ckpt_step(self) -> int:
        """Next-checkpoint discovery: one step ahead of the trainer.

        The orchestrator always runs one step ahead of the trainer, so we
        must advance to at least ``policy.version + 1`` once the trainer
        broadcasts it. We additionally adopt anything fresher the trainer
        has already published (a fast trainer briefly running on-policy is
        fine).
        """
        broadcast_dir = get_broadcast_dir(self.config.output_dir)
        latest_ckpt_step = get_latest_ckpt_step(broadcast_dir) or 0
        return max(self.policy.version, latest_ckpt_step)

    async def apply_policy_update(self, next_step: int) -> None:
        async with self.update_lock:
            if next_step <= self.ckpt_step:
                # Another caller raced us — bail without re-applying.
                return

            broadcast_dir = get_broadcast_dir(self.config.output_dir)
            weights_path = get_step_path(broadcast_dir, next_step)
            stable_marker = weights_path / "STABLE"
            if not stable_marker.exists():
                self.logger.info(
                    f"Orchestrator paused: waiting for trainer to broadcast checkpoint {next_step}. "
                    "Training is progressing normally."
                )
                t0 = time.perf_counter()
                await wait_for_path(stable_marker)
                self.last_wait_for_ckpt_time = time.perf_counter() - t0
                self.logger.info(
                    f"Orchestrator resumed: checkpoint {next_step} ready (after {self.last_wait_for_ckpt_time:.2f}s)"
                )

            self.logger.debug(f"Updating weights to step {next_step}")
            t1 = time.perf_counter()
            await self.inference.update_weights(weights_path, lora_name=self.lora_name, step=next_step)
            self.last_update_weights_time = time.perf_counter() - t1
            self.update_count += 1
            self.logger.debug(f"Updated weights to step {next_step} in {self.last_update_weights_time:.2f}s")

            self.ckpt_step = next_step
            self.policy.version = next_step
            if self.lora_name is not None:
                self.inference.update_model_name(self.lora_name)
                self.policy.model_name = self.lora_name

            # Notify observers in registration order. Each gets the freshly
            # installed version so they can invalidate stale work synchronously
            # (the dispatcher uses this for off-policy cancellation + eval
            # triggers).
            for observer in self.observers:
                try:
                    await observer.on_new_version(next_step)
                except Exception as exc:
                    self.logger.warning(
                        f"Observer {type(observer).__name__}.on_new_version({next_step}) raised: {exc!r}"
                    )

    def metrics(self) -> dict[str, float]:
        return {
            "watcher/update_weights_time_s": self.last_update_weights_time,
            "watcher/wait_for_ckpt_time_s": self.last_wait_for_ckpt_time,
            "watcher/update_count": float(self.update_count),
        }
