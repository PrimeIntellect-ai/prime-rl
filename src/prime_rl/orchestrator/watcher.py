import asyncio
import time
from typing import Protocol

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.group import Policy
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path
from prime_rl.utils.utils import get_latest_ckpt_step


class VersionObserver(Protocol):
    async def on_new_version(self, step: int) -> None: ...


class WeightWatcher:
    """Polls a broadcast directory for new trainer checkpoints.

    Each tick jumps straight to the LATEST step on disk, skipping intermediates
    the trainer's cleanup may have pruned. After observers (admin) succeed, we
    mutate `policy.version` (and `policy.model_name` once the LoRA adapter is
    loaded for the first time). Groups read those fields at dispatch time.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        observers: list[VersionObserver],
        policy: Policy,
        lora_name: str | None = None,
        poll_interval: float = 0.5,
    ):
        self.broadcast_dir = get_broadcast_dir(config.output_dir)
        self.observers = observers
        self.policy = policy
        self.lora_name = lora_name
        self.poll_interval = poll_interval
        self.current_step = 0
        self.logger = get_logger()

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval)
            await self.tick()

    async def tick(self) -> None:
        if not self.broadcast_dir.exists():
            return
        latest = get_latest_ckpt_step(self.broadcast_dir)
        if latest is None or latest <= self.current_step:
            return
        if not get_step_path(self.broadcast_dir, latest).exists():
            return  # raced with trainer cleanup
        try:
            t0 = time.perf_counter()
            for obs in self.observers:
                await obs.on_new_version(latest)
            self.policy.version = latest
            # LoRA adapter exists on vLLM only after admin's first successful
            # /load_lora_adapter call. Until then `policy.model_name` is the
            # base model.
            if self.lora_name and self.policy.model_name != self.lora_name:
                self.logger.info(f"Switching rollouts to LoRA adapter '{self.lora_name}'")
                self.policy.model_name = self.lora_name
            self.logger.success(f"Weights updated to step {latest} in {time.perf_counter() - t0:.2f}s")
            self.current_step = latest
        except Exception as exc:
            self.logger.warning(f"Weight update for step {latest} failed: {exc}. Skipping.")
            self.current_step = latest
