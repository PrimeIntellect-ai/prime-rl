"""
Scheduler that runs environments in subprocesses.

Isolates event loop lag from environment execution.
"""

import asyncio
import time
from pathlib import Path

from tqdm import tqdm

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.ckpt import Progress
from prime_rl.orchestrator.config import BufferConfig, EvalEnvGroupConfig, TrainEnvGroupConfig
from prime_rl.orchestrator.env_worker_group import EnvWorkerGroup
from prime_rl.utils.client import (
    setup_admin_clients,
    update_weights,
)
from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    wait_for_path,
)


class TrainRolloutScheduler:
    """Continuously schedules group rollout requests, sampling rollouts from a buffer."""

    def __init__(
        self,
        env_worker_group_config: TrainEnvGroupConfig,
        buffer_config: BufferConfig,
        batch_size: int,
        rollouts_per_example: int,
        model_name: str,
        oversampling_factor: float,
        schedule_rollouts: asyncio.Event | None = None,
        pending_tasks: dict[asyncio.Task, int] | None = None,
    ):
        self.logger = get_logger()
        self.env_worker_group = EnvWorkerGroup(env_worker_group_config)
        self.buffer_config = buffer_config
        self.model_name = model_name
        self.batch_size = batch_size
        self.rollouts_per_example = rollouts_per_example
        self.finished_rollouts: list[dict] = []
        self.oversampling_factor = oversampling_factor
        self.max_pending_groups = int(self.batch_size * self.oversampling_factor / self.rollouts_per_example)

        self.accepted_rollouts: list[dict] = []
        self.schedule_rollouts = schedule_rollouts or asyncio.Event()
        self.pending_tasks = pending_tasks or {}

    def start(self):
        self.env_worker_group.start()
        self.buffer = Buffer(self.env_worker_group, self.buffer_config)
        asyncio.create_task(self.generate())

    async def schedule_group_rollout(self):
        """Asynchronously schedules a group rollout request."""
        await self.schedule_rollouts.wait()  # only schedule if rollout scheduling is not blocked
        env_name, example_id = self.buffer.sample_inputs(n=1)[0]
        task = asyncio.create_task(self.env_worker_group.run_group(env_name, example_id, self.model_name))
        self.pending_tasks[task] = 0

    async def generate(self):
        """Generate a batch of rollouts continuously."""

        # initial fill
        self.logger.debug(f"Filling up {self.max_pending_groups} in-flight group rollout requests")
        while len(self.pending_tasks) < self.max_pending_groups:
            await self.schedule_group_rollout()

        pbar = tqdm(total=self.batch_size, desc="Generating rollouts (train)")

        while True:
            # wait for at least one future to complete
            finished_tasks, _ = await asyncio.wait(
                self.pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for finished_task in finished_tasks:
                # safely pop the future from tracking
                if self.pending_tasks.pop(finished_task, None) is None:
                    continue

                # store rollouts in rollout buffer
                rollouts = finished_task.result()
                self.buffer.update(rollouts)
                pbar.update(len(self.buffer.rollout_buffer))

                # refill
                await self.schedule_group_rollout()

    async def wait_for_batch(self):
        while len(self.buffer.rollout_buffer) < self.batch_size:
            await asyncio.sleep(0.1)
        return self.buffer.sample_rollouts(n=self.batch_size)


class EvalRolloutScheduler:
    """Scheduler that schedules and awaits num_examples * rollouts_per_example rollouts for evaluation per eval environment."""

    def __init__(self, env_worker_group_config: EvalEnvGroupConfig, model_name: str):
        self.env_worker_group = EnvWorkerGroup(env_worker_group_config)
        self.model_name = model_name

    def start(self):
        self.env_worker_group.start()

    async def generate_batch(self, env_name: str):
        num_examples = self.env_worker_group.get_dataset_size(env_name)
        tasks = []
        for example_id in range(num_examples):
            tasks.append(self.env_worker_group.run_group(env_name, example_id, self.model_name))
        return await asyncio.gather(*tasks)


class UpdatePolicyScheduler:
    """Scheduler that updates the policy to the latest available checkpoint."""

    def __init__(
        self,
        progress: Progress,
        max_async_level: int,
        strict_async_level: bool,
        max_off_policy_steps: int,
        client_config: ClientConfig,
        output_dir: Path,
        model_name: str,
        lora_name: str | None,
        schedule_rollouts: asyncio.Event,
        pending_tasks: dict[asyncio.Task, int],
    ):
        self.logger = get_logger()
        self.admin_clients = setup_admin_clients(client_config)
        self.max_async_level = max_async_level
        self.strict_async_level = strict_async_level
        self.max_off_policy_steps = max_off_policy_steps
        self.schedule_rollouts = schedule_rollouts
        self.progress = progress
        self.ckpt_step = progress.step  # resume is always from the same policy
        self.output_dir = output_dir
        self.model_name = model_name
        self.lora_name = lora_name
        self.pending_tasks = pending_tasks

    def start(self):
        self.schedule_rollouts.set()  # initially allow scheduling rollouts
        asyncio.create_task(self.update_policy_loop())

    async def update_policy_loop(self):
        while True:
            await self.update_policy()
            await asyncio.sleep(0.1)

    async def update_policy(self):
        """Updates the policy to the latest available checkpoint. Aborts rollout requests that are older than max_off_policy_steps."""
        latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.output_dir)) or 0
        async_away_ckpt_step = max(self.progress.step - self.max_async_level, 0)
        next_ckpt_step = (
            async_away_ckpt_step if self.strict_async_level else max(async_away_ckpt_step, latest_ckpt_step)
        )

        if next_ckpt_step > self.ckpt_step:
            if next_ckpt_step == async_away_ckpt_step:
                self.logger.info(
                    f"Hit async barrier because we are >{self.max_async_level} step(s) async. Waiting for checkpoint {next_ckpt_step}"
                )
                self.schedule_rollouts.clear()  # barrier enforced: block scheduling rollouts
                wait_for_ckpt_start_time = time.perf_counter()
                await wait_for_path(get_step_path(get_broadcast_dir(self.output_dir), next_ckpt_step) / "STABLE")
                self.wait_for_ckpt_time = time.perf_counter() - wait_for_ckpt_start_time
                self.logger.debug(f"Waited for checkpoint {next_ckpt_step} for {self.wait_for_ckpt_time:.2f}s")

            self.logger.debug(
                f"Got new policy with step {next_ckpt_step}. Updating weights and cancelling old rollout requests."
            )

            # Update weights on inference servers
            update_weights_start_time = time.perf_counter()
            await update_weights(
                self.admin_clients,
                get_step_path(get_broadcast_dir(self.output_dir), next_ckpt_step),
                lora_name=self.lora_name,
            )
            self.update_weights_time = time.perf_counter() - update_weights_start_time
            self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

            if self.lora_name is not None:
                self.model_name = self.lora_name

            self.schedule_rollouts.set()  # barrier cleared: allow scheduling rollouts

            # Handle off-policy tracking - cancel old requests
            tasks_to_remove, tasks_to_update = [], []
            for task, off_policy_steps in self.pending_tasks.items():
                if off_policy_steps > self.max_off_policy_steps:
                    if not task.done():
                        task.cancel()
                    tasks_to_remove.append(task)
                else:
                    tasks_to_update.append((task, off_policy_steps + 1))

            # safely remove cancelled tasks for stale groups
            for task in tasks_to_remove:
                self.pending_tasks.pop(task, None)

            # update off-policy steps for remaining
            for task, off_policy_steps in tasks_to_update:
                if task in self.pending_tasks:
                    self.pending_tasks[task] = off_policy_steps

            if len(tasks_to_remove) > 0:
                self.logger.warning(
                    f"Cancelled {len(tasks_to_remove)} old rollout requests. Consider increasing max_off_policy_steps to avoid this."
                )

            self.ckpt_step = next_ckpt_step


class TrainScheduler:
    """Scheduler that schedules rollouts and updates the policy for training."""

    def __init__(
        self,
        progress: Progress,
        max_async_level: int,
        strict_async_level: bool,
        max_off_policy_steps: int,
        client_config: ClientConfig,
        output_dir: Path,
        model_name: str,
        lora_name: str | None,
        env_worker_group_config: TrainEnvGroupConfig,
        buffer_config: BufferConfig,
        batch_size: int,
        rollouts_per_example: int,
        oversampling_factor: float,
    ):
        self.pending_tasks: dict[asyncio.Task, int] = {}
        self.schedule_rollouts = asyncio.Event()
        self.rollout_scheduler = TrainRolloutScheduler(
            env_worker_group_config,
            buffer_config,
            batch_size,
            rollouts_per_example,
            model_name,
            oversampling_factor,
            self.schedule_rollouts,
            self.pending_tasks,
        )
        self.update_policy_scheduler = UpdatePolicyScheduler(
            progress,
            max_async_level,
            strict_async_level,
            max_off_policy_steps,
            client_config,
            output_dir,
            model_name,
            lora_name,
            self.schedule_rollouts,
            self.pending_tasks,
        )

    def start(self):
        self.rollout_scheduler.start()
        self.update_policy_scheduler.start()

    async def wait_for_batch(self):
        return await self.rollout_scheduler.wait_for_batch()
