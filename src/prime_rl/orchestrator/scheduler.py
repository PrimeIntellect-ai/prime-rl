"""
Scheduler that runs environments in subprocesses.

Isolates event loop lag from environment execution.
"""

import asyncio
import time
from pathlib import Path

from tqdm import tqdm

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import BufferConfig, EvalEnvGroupConfig, TrainEnvGroupConfig
from prime_rl.orchestrator.env_worker_group import EnvWorkerGroup
from prime_rl.utils.client import (
    update_weights,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    wait_for_path,
)


class RolloutScheduler:
    """Schedules a batch of group rollout requests and await synchronously."""

    def __init__(
        self,
        env_worker_group_config: TrainEnvGroupConfig | EvalEnvGroupConfig,
        buffer_config: BufferConfig,
        batch_size: int,
        rollouts_per_example: int,
        model_name: str,
    ):
        self.logger = get_logger()
        self.env_worker_group = EnvWorkerGroup(env_worker_group_config)
        self.buffer_config = buffer_config
        self.model_name = model_name
        self.batch_size = batch_size
        self.rollouts_per_example = rollouts_per_example
        self.finished_rollouts: list[dict] = []

    def start(self):
        self.env_worker_group.start()
        self.buffer = Buffer(self.env_worker_group, self.buffer_config)

        asyncio.create_task(self.generate())

    async def generate(self):
        while True:
            if len(self.finished_rollouts) >= self.batch_size:
                await asyncio.sleep(0.1)
                break

            rollouts_left = self.batch_size - len(self.finished_rollouts)
            inputs = self.buffer.sample_inputs(n=rollouts_left // self.rollouts_per_example)
            tasks = []
            for env_name, example_id in inputs:
                task = asyncio.create_task(self.env_worker_group.run_group(env_name, example_id, self.model_name))
                tasks.append(task)

            rollouts = [rollout for rollouts in await asyncio.gather(*tasks) for rollout in rollouts]
            self.buffer.update(rollouts)
            self.buffer.sample_rollouts(n=self.batch_size)
            self.finished_rollouts.extend(rollouts)


class ContinuousRolloutScheduler(RolloutScheduler):
    """Continuously schedules group rollout requests.

    References:
    - AReal: https://arxiv.org/abs/2505.24298v1
    - PipelineRL: https://arxiv.org/abs/2509.19128v1
    """

    def __init__(
        self,
        env_worker_group_config: TrainEnvGroupConfig | EvalEnvGroupConfig,
        buffer_config: BufferConfig,
        batch_size: int,
        rollouts_per_example: int,
        model_name: str,
        oversampling_factor: float,
        schedule_rollouts: asyncio.Event,
    ):
        super().__init__(env_worker_group_config, buffer_config, batch_size, rollouts_per_example, model_name)
        self.oversampling_factor = oversampling_factor
        self.max_pending_groups = self.batch_size * self.oversampling_factor // self.rollouts_per_example
        self.schedule_rollouts = schedule_rollouts

        # tracks how 'stale' the rollout request is
        self.pending_tasks: dict[asyncio.Task, int] = {}  # task -> off_policy_steps
        self.finished_rollouts: list[dict] = []

    async def start(self):
        self.env_worker_group.start()
        self.buffer = Buffer(self.env_worker_group, self.buffer_config)

    async def schedule_group_rollout(self):
        """Asynchronously schedules a group rollout request."""
        env_name, example_id = self.buffer.sample_inputs(n=1)[0]
        task = asyncio.create_task(self.env_worker_group.run_group(env_name, example_id, self.model_name))
        self.pending_tasks[task] = 0

    async def generate(self) -> list[dict]:
        """Generate a batch of rollouts continuously."""

        await self.schedule_rollouts.wait()

        # Schedule initial tasks
        self.logger.debug("Starting to generate batch rollouts")
        while len(self.pending_tasks) < self.max_pending_groups:
            await self.schedule_group_rollout()

        batch_rollouts: list[dict] = []
        pbar = tqdm(total=self.batch_size, desc="Generating rollouts (train)")

        while len(batch_rollouts) < self.batch_size:
            # wait for at least one future to complete
            finished_tasks, _ = await asyncio.wait(
                self.pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            await self.schedule_rollouts.wait()

            for finished_task in finished_tasks:
                if len(batch_rollouts) >= self.batch_size:
                    batch_rollouts = batch_rollouts[: self.batch_size]
                    break

                # safely pop the future from tracking
                if self.pending_tasks.pop(finished_task, None) is None:
                    continue

                rollouts = finished_task.result()

                # Update buffer with results
                self.buffer.update(rollouts)
                accepted_rollouts = self.buffer.sample_rollouts(n=self.rollouts_per_example)

                batch_rollouts.extend(accepted_rollouts)
                pbar.update(len(accepted_rollouts))

                # refill
                await self.schedule_group_rollout()

        pbar.close()
        return batch_rollouts


class UpdatePolicyScheduler:
    """Scheduler that updates the policy to the latest available checkpoint."""

    def start(
        self,
        step: int,
        max_async_level: int,
        strict_async_level: bool,
        max_off_policy_steps: int,
        schedule_rollouts: asyncio.Event,
        output_dir: Path,
    ):
        self.max_async_level = max_async_level
        self.strict_async_level = strict_async_level
        self.max_off_policy_steps = max_off_policy_steps
        self.schedule_rollouts = schedule_rollouts
        self.step = step
        self.ckpt_step = step  # resume is always from the same policy
        self.output_dir = output_dir
        asyncio.create_task(self.update_policy_loop())

    async def update_policy_loop(self):
        while True:
            await self.update_policy()
            await asyncio.sleep(0.1)

    async def update_policy(self):
        """Updates the policy to the latest available checkpoint. Aborts rollout requests that are older than max_off_policy_steps."""
        latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.output_dir)) or 0
        async_away_ckpt_step = max(self.step - self.max_async_level, 0)
        next_ckpt_step = (
            async_away_ckpt_step if self.strict_async_level else max(async_away_ckpt_step, latest_ckpt_step)
        )

        if next_ckpt_step > self.ckpt_step:
            if next_ckpt_step == async_away_ckpt_step:
                self.logger.info(
                    f"Hit async barrier because we are >{self.max_async_level} step(s) async. Waiting for checkpoint {next_ckpt_step}"
                )
                self.checkpoint_ready.clear()
                wait_for_ckpt_start_time = time.perf_counter()
                await wait_for_path(get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
                self.wait_for_ckpt_time = time.perf_counter() - wait_for_ckpt_start_time
                self.logger.debug(f"Waited for checkpoint {next_ckpt_step} for {self.wait_for_ckpt_time:.2f}s")

            self.logger.debug(
                f"Got new policy with step {next_ckpt_step}. Updating weights and cancelling old rollout requests."
            )

            # Update weights on inference servers
            update_weights_start_time = time.perf_counter()
            await update_weights(
                self.admin_clients,
                get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step),
                lora_name=self.lora_name,
            )
            self.update_weights_time = time.perf_counter() - update_weights_start_time
            self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

            if self.lora_name is not None:
                self.model_name = self.lora_name

            # Update model name on all workers
            for workers in self.workers.values():
                for worker in workers:
                    worker.update_model_name(self.model_name)

            self.checkpoint_ready.set()

            # Handle off-policy tracking - cancel old requests
            futures_to_remove = []
            futures_to_update = []

            for future, info in self.inflight_group_rollouts.items():
                if info.off_policy_steps > self.max_off_policy_steps:
                    if not future.done():
                        future.cancel()
                    futures_to_remove.append((future, info.worker))
                else:
                    futures_to_update.append((future, info.off_policy_steps + 1, info.worker, info.request_id))

            # Remove cancelled
            for future, worker in futures_to_remove:
                self.inflight_group_rollouts.pop(future, None)
            self.cancelled_rollouts_count += len(futures_to_remove)

            # Update off-policy steps for remaining
            for future, off_policy_steps, worker, request_id in futures_to_update:
                if future in self.inflight_group_rollouts:
                    self.inflight_group_rollouts[future] = InflightRolloutInfo(
                        off_policy_steps=off_policy_steps,
                        worker=worker,
                        request_id=request_id,
                    )

            if len(futures_to_remove) > 0:
                self.logger.warning(
                    f"Cancelled {len(futures_to_remove)} old rollout requests (will refill naturally). Consider increasing max_off_policy_steps to avoid this."
                )

            self.ckpt_step = next_ckpt_step
