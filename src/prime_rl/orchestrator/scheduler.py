from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import NamedTuple

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.vf_utils import run_group
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
)


class InflightRolloutInfo(NamedTuple):
    """Metadata for an in-flight group rollout request."""

    off_policy_steps: int
    client_config: vf.ClientConfig


class Scheduler:
    """
    Asynchronously manages scheduling of group rollout requests and policy
    updates. Keeps a constant number of groups in-flight (continuous batching)
    and updates the policy as soon as it becomes available.

    References:
    - AReal: https://arxiv.org/abs/2505.24298v1
    - PipelineRL: https://arxiv.org/abs/2509.19128v1
    """

    def __init__(
        self,
        env: vf.Environment,
        inference_pool: InferencePool,
        buffer: Buffer,
        config: OrchestratorConfig,
    ):
        self.logger = get_logger()
        if config.tasks_per_minute is not None:
            self.rate_limiter = AsyncLimiter(max_rate=config.tasks_per_minute, time_period=60)
        else:
            self.rate_limiter = None
        self.env = env
        self.buffer = buffer
        self.config = config
        self.inference_pool = inference_pool
        self.lora_name = config.model.lora.name if config.model.lora else None
        self.sampling_args: dict = {}
        self.model_name = config.model.name

        # Track in-flight requests: task -> info
        self.inflight_group_rollouts: dict[asyncio.Task, InflightRolloutInfo] = {}

        self.ckpt_step = 0
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()
        self.update_weights_time = 0
        self.cancelled_rollouts_count = 0

    def set_sampling_args(self, sampling_args: dict) -> None:
        """Update sampling args for future rollout requests."""
        self.sampling_args = sampling_args

    def cancel_all_inflight_rollouts(self):
        """Cancel all in-flight rollout requests.

        Used when weights are updated to discard stale rollouts generated with old weights.
        """
        count = len(self.inflight_group_rollouts)
        for future in list(self.inflight_group_rollouts.keys()):
            if not future.done():
                future.cancel()
        self.inflight_group_rollouts.clear()
        self.cancelled_rollouts_count += count

    async def _select_least_loaded_client(self) -> vf.ClientConfig:
        """Select the client with the fewest in-flight tasks."""
        clients = self.inference_pool.clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.inference_pool.clients
        inflight_by_url = Counter(info.client_config.api_base_url for info in self.inflight_group_rollouts.values())
        return min(clients, key=lambda c: inflight_by_url[c.api_base_url])

    async def schedule_group_rollout(self):
        """Asynchronously schedules a group rollout request."""
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        example = self.buffer.sample_examples(n=1)[0]
        client_config = await self._select_least_loaded_client()
        run_group_task = asyncio.create_task(
            run_group(
                env=self.env,
                client=client_config,
                example=example,
                model_name=self.model_name,
                rollouts_per_example=self.config.rollouts_per_example,
                sampling_args=self.sampling_args,
                max_retries=0,  # TODO: make configurable
            )
        )
        self.inflight_group_rollouts[run_group_task] = InflightRolloutInfo(0, client_config)

    async def update_policy_loop(self):
        """Continuously checks for new policy checkpoints."""
        while True:
            await self.update_policy()
            await asyncio.sleep(1)

    async def update_policy(self):
        """Updates the policy to the latest available checkpoint. Aborts rollout requests that are older than the max retention steps."""
        next_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.config.output_dir)) or 0

        if next_ckpt_step > self.ckpt_step:
            self.logger.debug(
                f"Got new policy with step {next_ckpt_step}. Updating weights and cancelling old rollout requests."
            )

            # Update weights on inference servers
            update_weights_start_time = time.perf_counter()
            weights_path = get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step)
            await self.inference_pool.update_weights(weights_path, lora_name=self.lora_name, step=next_ckpt_step)
            self.update_weights_time = time.perf_counter() - update_weights_start_time
            self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

            if self.lora_name is not None:
                self.model_name = self.lora_name
                self.inference_pool.update_model_name(self.model_name)

            self.checkpoint_ready.set()

            # Handle off-policy tracking - cancel old requests
            tasks_to_remove = []
            tasks_to_update = []

            for task, info in self.inflight_group_rollouts.items():
                if info.off_policy_steps > self.config.max_off_policy_steps:
                    if not task.done():
                        task.cancel()
                    tasks_to_remove.append((task, info.client_config))
                else:
                    tasks_to_update.append((task, info.off_policy_steps + 1, info.client_config))

            # Remove cancelled
            for task, _ in tasks_to_remove:
                self.inflight_group_rollouts.pop(task, None)
            self.cancelled_rollouts_count += len(tasks_to_remove)

            # Update off-policy steps for remaining
            for task, off_policy_steps, client_config in tasks_to_update:
                if task in self.inflight_group_rollouts:
                    self.inflight_group_rollouts[task] = InflightRolloutInfo(
                        off_policy_steps=off_policy_steps, client_config=client_config
                    )

            if len(tasks_to_remove) > 0:
                self.logger.warning(
                    f"Cancelled {len(tasks_to_remove)} old rollout requests (will refill naturally). Consider increasing max_off_policy_steps to avoid this."
                )

            self.ckpt_step = next_ckpt_step

    async def next_completed_group(self) -> list[vf.RolloutOutput]:
        """Wait for one group rollout to complete and return its rollouts.

        Returns an empty list if the group was cancelled or failed.
        """
        # Top up the inflight pool
        while len(self.inflight_group_rollouts) < self.config.max_inflight_rollouts:
            await self.schedule_group_rollout()

        # Wait for at least one future to complete
        done, _ = await asyncio.wait(
            self.inflight_group_rollouts.keys(),
            return_when=asyncio.FIRST_COMPLETED,
        )

        await self.checkpoint_ready.wait()

        for finished_task in done:
            if self.inflight_group_rollouts.pop(finished_task, None) is None:
                continue

            rollouts: list[vf.RolloutOutput] = []
            try:
                rollouts = finished_task.result()
                self.buffer.update(rollouts)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.warning(f"Rollout failed: {e}")

            # Schedule a replacement
            await self.schedule_group_rollout()
            return rollouts

        return []

    @property
    def max_off_policy_level(self) -> int:
        if not self.inflight_group_rollouts:
            return 0
        return max(info.off_policy_steps for info in self.inflight_group_rollouts.values())

    @property
    def mean_off_policy_level(self) -> float:
        if not self.inflight_group_rollouts:
            return 0
        steps = [info.off_policy_steps for info in self.inflight_group_rollouts.values()]
        return sum(steps) / len(steps)

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "time/update_weights": self.update_weights_time,
            "batch/off_policy_level/max": self.max_off_policy_level,
            "batch/off_policy_level/mean": self.mean_off_policy_level,
            "batch/cancelled_rollouts": self.cancelled_rollouts_count,
        }
        self.cancelled_rollouts_count = 0

        # Add inference pool metrics (e.g. elastic pool server counts)
        metrics.update(self.inference_pool.get_metrics())

        return metrics
