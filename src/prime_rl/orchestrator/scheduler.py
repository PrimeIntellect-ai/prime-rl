from __future__ import annotations

import asyncio
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.orchestrator.vf_utils import get_seq_len, run_group
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    wait_for_path,
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
        max_inflight_rollouts: int,
        max_async_level: int,
        max_off_policy_steps: int,
        strict_async_level: bool,
        tasks_per_minute: int | None,
        lora_name: str | None = None,
        output_dir: Path | None = None,
    ):
        self.logger = get_logger()
        if tasks_per_minute is not None:
            self.rate_limiter = AsyncLimiter(max_rate=tasks_per_minute, time_period=60)
        else:
            self.rate_limiter = None
        self.env = env
        self.buffer = buffer
        self.config = config
        self.batch_size = config.batch_size
        self.token_batch_size = config.token_batch_size
        self.rollouts_per_example = config.rollouts_per_example
        self.seq_len = config.seq_len
        self.max_inflight_rollouts = max_inflight_rollouts
        self.max_inflight_group_rollouts = max(1, self.max_inflight_rollouts // self.rollouts_per_example)
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

    @property
    def uses_token_batching(self) -> bool:
        return self.token_batch_size is not None

    @property
    def batch_target(self) -> int:
        if self.uses_token_batching:
            assert self.token_batch_size is not None
            return self.token_batch_size
        assert self.batch_size is not None
        return self.batch_size

    def get_batch_progress_increment(self, rollouts: list[vf.RolloutOutput]) -> int:
        if self.uses_token_batching:
            return sum(get_seq_len(rollout) for rollout in rollouts)
        return len(rollouts)

    def finalize_batch_rollouts(self, rollouts: list[vf.RolloutOutput]) -> list[vf.RolloutOutput]:
        if self.batch_size is None:
            return rollouts
        return rollouts[: self.batch_size]

    def set_sampling_args(self, sampling_args: dict) -> None:
        """Update sampling args for future rollout requests."""
        self.sampling_args = sampling_args

    async def cancel_inflight_rollouts(self):
        """Cancel all in-flight rollout requests."""
        count = len(self.inflight_group_rollouts)
        for future in list(self.inflight_group_rollouts.keys()):
            await safe_cancel(future)
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
            await self.maybe_update_policy()
            await asyncio.sleep(1)

    async def maybe_update_policy(self):
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

            # Update weights on inference servers.
            update_weights_start_time = time.perf_counter()
            weights_path = get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step)
            await self.inference_pool.update_weights(weights_path, lora_name=self.lora_name, step=next_ckpt_step)
            self.update_weights_time = time.perf_counter() - update_weights_start_time
            self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

            # Signal to the trainer that this checkpoint has been consumed and older ones can be cleaned
            loaded_step_file = get_broadcast_dir(self.config.output_dir) / "LOADED_STEP"
            loaded_step_tmp_file = loaded_step_file.with_suffix(".tmp")
            loaded_step_tmp_file.write_text(str(next_ckpt_step))
            loaded_step_tmp_file.replace(loaded_step_file)

            if self.lora_name is not None:
                self.model_name = self.lora_name
                self.inference_pool.update_model_name(self.model_name)

            self.checkpoint_ready.set()

            # Cancel stale rollouts and bump off-policy step for the rest
            cancelled = []
            updated = {}
            for task, info in list(self.inflight_group_rollouts.items()):
                new_off_policy_steps = info.off_policy_steps + 1
                if new_off_policy_steps > self.max_off_policy_steps:
                    cancelled.append(task)
                else:
                    updated[task] = InflightRolloutInfo(new_off_policy_steps, info.client_config)
            self.inflight_group_rollouts = updated
            await safe_cancel_all(cancelled)
            self.cancelled_rollouts_count += len(cancelled)

            if len(cancelled) > 0:
                self.logger.warning(
                    f"Cancelled {len(cancelled)} group rollout requests because they would be generated by >{self.max_off_policy_steps} policy versions. These requests will refill naturally. To avoid cancellation, consider increasing max_off_policy_steps."
                )

            self.ckpt_step = next_ckpt_step

    async def generate_batch(self, step: int) -> list[vf.RolloutOutput]:
        """Continuously generates a batch of rollouts."""
        self.step = step

        # Cancel the previous update policy task to avoid concurrent updates
        if self.update_policy_task is not None:
            await safe_cancel(self.update_policy_task)

        # Manually check the async barrier before starting the step, the re-create the update policy loop
        # This ensures that we respect max_async_level, while still listening for policy updates mid-step
        await self.maybe_update_policy()
        self.update_policy_task = asyncio.create_task(self.update_policy_loop())

        batch_start_time = time.perf_counter()

        batch_rollouts: list[vf.RolloutOutput] = []
        batch_progress = 0
        pbar = ProgressTracker(
            total=self.batch_target, desc="Generating rollouts (train)", json_logging=self.json_logging, step=step
        )

        while batch_progress < self.batch_target:
            # Fill up
            while len(self.inflight_group_rollouts) < self.max_inflight_group_rollouts:
                await self.schedule_group_rollout()

            # Wait for at least one future to complete
            finished_tasks, _ = await asyncio.wait(
                self.inflight_group_rollouts.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            await self.checkpoint_ready.wait()

            for finished_task in finished_tasks:
                if batch_progress >= self.batch_target:
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
                    progress_increment = self.get_batch_progress_increment(accepted_rollouts)
                    batch_progress += progress_increment
                    pbar.update(progress_increment)

                except asyncio.CancelledError:
                    pass  # Request was cancelled, will be rescheduled
                except Exception as e:
                    self.logger.warning(f"Rollout failed: {e}")

        batch_rollouts = self.finalize_batch_rollouts(batch_rollouts)

        pbar.close()
        self.last_batch_generation_time = time.perf_counter() - batch_start_time
        return batch_rollouts

    async def stop(self) -> None:
        await self.cancel_inflight_rollouts()
        if self.update_policy_task is not None:
            await safe_cancel(self.update_policy_task)
            self.update_policy_task = None

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
