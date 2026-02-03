"""
Scheduler that runs environments in subprocesses.

Isolates event loop lag from environment execution.
"""

from __future__ import annotations

import asyncio
import time
from itertools import cycle
from pathlib import Path
from typing import NamedTuple

import verifiers as vf

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    wait_for_path,
)
from prime_rl.utils.vf import generate_group


class InflightRolloutInfo(NamedTuple):
    """Metadata for an in-flight group rollout request."""

    off_policy_steps: int
    client_config: vf.ClientConfig


class Scheduler:
    """Asynchronously schedules group rollout requests using subprocess workers.

    Runs environment execution in separate processes to isolate event loop lag
    from the main orchestrator process.

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
        oversampling_factor: float,
        max_async_level: int,
        max_off_policy_steps: int,
        strict_async_level: bool,
        lora_name: str | None = None,
        output_dir: Path | None = None,
    ):
        self.logger = get_logger()
        self.env = env
        self.buffer = buffer
        self.config = config
        self.batch_size = config.batch_size
        self.rollouts_per_example = config.rollouts_per_example
        self.seq_len = config.seq_len
        self.problems_per_batch = int(oversampling_factor * self.batch_size // self.rollouts_per_example)
        self.max_async_level = max_async_level
        self.max_off_policy_steps = max_off_policy_steps
        self.strict_async_level = strict_async_level
        self.lora_name = lora_name
        initial_temp = compute_temperature(step=0, sampling_config=config.sampling, max_steps=config.max_steps)
        self.sampling_args = get_sampling_args(config.sampling, temperature=initial_temp)
        self.model_name = self.config.model.name
        self.json_logging = config.log.json_logging

        # Inference pool - used for admin operations (adapter sync) and metrics
        self.inference_pool = inference_pool

        # Track in-flight requests: task -> info
        self.inflight_group_rollouts: dict[asyncio.Task, InflightRolloutInfo] = {}

        self.step, self.ckpt_step = 0, 0
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()
        self.update_weights_time, self.wait_for_ckpt_time = 0, 0
        self.update_policy_task = None
        self.cancelled_rollouts_count = 0
        self.last_batch_generation_time = 0.0

        self.client_config_cycle = cycle(inference_pool.clients)

    async def start(self):
        """Start the scheduler."""
        self.update_policy_task = asyncio.create_task(self.update_policy_loop())

    async def stop(self):
        """Stop the scheduler."""
        if self.update_policy_task is not None:
            self.update_policy_task.cancel()

        for task in list(self.inflight_group_rollouts.keys()):
            if not task.done():
                task.cancel()
        if self.inflight_group_rollouts:
            await asyncio.gather(*self.inflight_group_rollouts.keys(), return_exceptions=True)
        self.inflight_group_rollouts.clear()

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

    async def schedule_group_rollout(self):
        """Asynchronously schedules a group rollout request."""
        example = self.buffer.sample_examples(n=1)[0]
        client_config = next(self.client_config_cycle)
        run_group_task = asyncio.create_task(
            generate_group(
                example=example,
                client=client_config,
                env=self.env,
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
        latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.config.output_dir)) or 0
        async_away_ckpt_step = max(self.step - self.max_async_level, 0)
        next_ckpt_step = (
            async_away_ckpt_step if self.strict_async_level else max(async_away_ckpt_step, latest_ckpt_step)
        )

        if next_ckpt_step > self.ckpt_step:
            if next_ckpt_step == async_away_ckpt_step:
                self.logger.info(
                    f"Orchestrator paused: waiting for trainer process to complete checkpoint {next_ckpt_step} "
                    f"(>{self.max_async_level} step(s) ahead). Training is progressing normally."
                )
                self.checkpoint_ready.clear()
                wait_for_ckpt_start_time = time.perf_counter()
                await wait_for_path(get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
                self.wait_for_ckpt_time = time.perf_counter() - wait_for_ckpt_start_time
                self.logger.info(
                    f"Orchestrator resumed: checkpoint {next_ckpt_step} ready (after {self.wait_for_ckpt_time:.2f}s)"
                )

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

            self.checkpoint_ready.set()

            # Handle off-policy tracking - cancel old requests
            tasks_to_remove = []
            tasks_to_update = []

            for task, info in self.inflight_group_rollouts.items():
                if info.off_policy_steps > self.max_off_policy_steps:
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
                        off_policy_steps=off_policy_steps + 1, client_config=client_config
                    )

            if len(tasks_to_remove) > 0:
                self.logger.warning(
                    f"Cancelled {len(tasks_to_remove)} old rollout requests (will refill naturally). Consider increasing max_off_policy_steps to avoid this."
                )

            self.ckpt_step = next_ckpt_step

    async def generate_batch(self, step: int) -> list[dict]:
        """Generate a batch of rollouts using workers.

        Returns list of result dicts (not vf.State, since those stay in workers).
        """
        self.step = step
        batch_start_time = time.perf_counter()

        # Schedule initial tasks
        self.logger.debug("Starting to generate batch rollouts")
        while len(self.inflight_group_rollouts) < self.problems_per_batch:
            await self.schedule_group_rollout()

        batch_rollouts: list[dict] = []
        pbar = ProgressTracker(
            total=self.config.batch_size, desc="Generating rollouts (train)", json_logging=self.json_logging, step=step
        )

        while len(batch_rollouts) < self.config.batch_size:
            # Wait for at least one future to complete
            finished_tasks, _ = await asyncio.wait(
                self.inflight_group_rollouts.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            await self.checkpoint_ready.wait()

            for finished_task in finished_tasks:
                if len(batch_rollouts) >= self.config.batch_size:
                    batch_rollouts = batch_rollouts[: self.config.batch_size]
                    break

                # Safely pop the future from tracking
                if self.inflight_group_rollouts.pop(finished_task, None) is None:
                    continue

                try:
                    finished_rollouts: list[vf.RolloutOutput] = finished_task.result()

                    # Update buffer with results
                    self.buffer.update(finished_rollouts)
                    accepted_rollouts = self.buffer.sample_rollouts(n=self.config.rollouts_per_example)

                    batch_rollouts.extend(accepted_rollouts)
                    pbar.update(len(accepted_rollouts))

                except asyncio.CancelledError:
                    pass  # Request was cancelled, will be rescheduled
                except Exception as e:
                    self.logger.warning(f"Rollout failed: {e}")

                await self.schedule_group_rollout()

        pbar.close()
        self.last_batch_generation_time = time.perf_counter() - batch_start_time
        return batch_rollouts

    @property
    def max_off_policy_level(self) -> int:
        if not self.inflight_group_rollouts:
            return 0
        return max(info.off_policy_steps for info in self.inflight_group_rollouts.values())

    @property
    def min_off_policy_level(self) -> int:
        if not self.inflight_group_rollouts:
            return 0
        return min(info.off_policy_steps for info in self.inflight_group_rollouts.values())

    @property
    def mean_off_policy_level(self) -> float:
        if not self.inflight_group_rollouts:
            return 0
        steps = [info.off_policy_steps for info in self.inflight_group_rollouts.values()]
        return sum(steps) / len(steps)

    @property
    def async_level(self) -> int:
        return self.step - self.ckpt_step

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "batch/async_level": self.async_level,
            "batch/inflight_rollouts": len(self.inflight_group_rollouts),
            "batch/inflight_samples": len(self.inflight_group_rollouts) * self.rollouts_per_example,
            "batch/off_policy_level/max": self.max_off_policy_level,
            "batch/off_policy_level/mean": self.mean_off_policy_level,
            "batch/off_policy_level/min": self.min_off_policy_level,
            "batch/cancelled_rollouts": self.cancelled_rollouts_count,
        }
        self.cancelled_rollouts_count = 0

        # Add inference pool metrics (e.g. elastic pool server counts)
        metrics.update(self.inference_pool.get_metrics())

        return metrics
