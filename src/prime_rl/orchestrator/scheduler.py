import asyncio
import time
from itertools import cycle
from typing import NamedTuple

import verifiers as vf
from httpx import AsyncClient
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from verifiers import Environment

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.utils.client import update_weights
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    sync_wait_for_path,
)
from prime_rl.utils.vf import generate_group


class InflightRolloutInfo(NamedTuple):
    """Metadata for an in-flight group rollout request."""

    off_policy_steps: int
    client: AsyncOpenAI


class Scheduler:
    """Asynchronously schedules group rollout requests and re-schedules them as they complete (continuous batching). Updates policy in between group rollout requests.

    References:
    - AReal: https://arxiv.org/abs/2505.24298v1
    - PipelineRL: https://arxiv.org/abs/2509.19128v1
    """

    def __init__(
        self,
        clients: list[AsyncOpenAI],
        admin_clients: list[AsyncClient],
        env: Environment,
        buffer: Buffer,
        tokenizer: PreTrainedTokenizerFast,
        config: OrchestratorConfig,
        oversampling_factor: float,
        max_async_level: int,
        max_off_policy_steps: int,
        strict_async_level: bool,
        lora_name: str | None = None,
    ):
        self.logger = get_logger()
        self.clients = clients
        self.admin_clients = admin_clients
        self.env = env
        self.buffer = buffer
        self.tokenizer = tokenizer
        self.config = config
        self.batch_size = config.batch_size
        self.rollouts_per_example = config.rollouts_per_example
        self.seq_len = config.seq_len
        self.problems_per_batch = int(oversampling_factor * self.batch_size // self.rollouts_per_example)
        self.max_async_level = max_async_level
        self.max_off_policy_steps = max_off_policy_steps
        self.strict_async_level = strict_async_level
        self.lora_name = lora_name
        self.inflight_group_rollouts: dict[asyncio.Task, InflightRolloutInfo] = {}
        self.cycle_clients = cycle(self.clients)
        self.step, self.ckpt_step = 0, 0
        self.update_weights_time, self.wait_for_ckpt_time = 0, 0
        self.sampling_args = get_sampling_args(config.sampling)
        self.model_name = self.config.model.name

    async def schedule_group_rollout(self, client: AsyncOpenAI | None = None):
        """Asynchronously schedules a group rollout request."""
        problem = self.buffer.sample_problems(n=1)[0]
        if client is None:
            client = next(self.cycle_clients)
        group_rollout_request = asyncio.create_task(
            generate_group(
                client=client,
                env=self.env,
                model_name=self.model_name,
                example=problem,
                rollouts_per_example=self.config.rollouts_per_example,
                sampling_args=self.sampling_args,
            )
        )
        await asyncio.sleep(0)
        self.inflight_group_rollouts[group_rollout_request] = InflightRolloutInfo(0, client)

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
                    f"Hit async barrier because we are >{self.max_async_level} step(s) async. Waiting for checkpoint {next_ckpt_step}"
                )
                wait_for_ckpt_start_time = time.perf_counter()
                sync_wait_for_path(get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
                self.wait_for_ckpt_time = time.perf_counter() - wait_for_ckpt_start_time
                self.logger.debug(f"Waited for checkpoint {next_ckpt_step} for {self.wait_for_ckpt_time:.2f}s")
            self.logger.debug(
                f"Got new policy with step {next_ckpt_step}. Updating weights and cancelling old rollout requests."
            )

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

            # Cancel old rollout requests
            tasks_to_remove = []
            tasks_to_update = []

            for task, (off_policy_steps, client) in self.inflight_group_rollouts.items():
                if off_policy_steps > self.max_off_policy_steps and task.cancel():
                    tasks_to_remove.append((task, client))
                else:
                    tasks_to_update.append((task, off_policy_steps + 1, client))

            # Remove cancelled tasks
            for task, client in tasks_to_remove:
                if self.inflight_group_rollouts.pop(task, None):
                    await self.schedule_group_rollout(client)

            # Update retention steps for remaining tasks
            for task, off_policy_steps, client in tasks_to_update:
                if self.inflight_group_rollouts.get(task, None):
                    self.inflight_group_rollouts[task] = InflightRolloutInfo(
                        off_policy_steps=off_policy_steps, client=client
                    )

            if len(tasks_to_remove) > 0:
                self.logger.warning(f"Cancelled and re-scheduled {len(tasks_to_remove)} old rollout requests.")

            self.ckpt_step = next_ckpt_step

    async def generate_batch(self, step: int) -> list[vf.State]:
        """Continuously schedules group rollouts, allowing them to be in-flight across steps."""
        self.step = step

        # Schedule initial tasks
        self.logger.debug("Starting to generate batch rollouts")
        while len(self.inflight_group_rollouts) < self.problems_per_batch:
            await self.schedule_group_rollout()  # Schedule requests in round-robin fashion

        batch_rollouts: list[vf.State] = []
        pbar = tqdm(total=self.config.batch_size, desc="Generating rollouts (train)")
        while len(batch_rollouts) < self.config.batch_size:
            finished_group_rollouts, _ = await asyncio.wait(
                self.inflight_group_rollouts, return_when=asyncio.FIRST_COMPLETED
            )

            for finished_group_rollout in finished_group_rollouts:
                if len(batch_rollouts) >= self.config.batch_size:
                    batch_rollouts = batch_rollouts[: self.config.batch_size]
                    break

                # Safely pop the task info. If it returns None, the task was removed externally.
                # This handles the race condition where update_policy() might have concurrently
                # cancelled the task and removed it from inflight_group_rollouts.
                popped_info = self.inflight_group_rollouts.pop(finished_group_rollout, None)
                if popped_info is None:
                    continue
                _, client = popped_info

                group_states: list[vf.State] = finished_group_rollout.result()

                self.buffer.update(group_states)
                accepted_rollouts = self.buffer.sample_rollouts(n=self.config.rollouts_per_example)

                batch_rollouts.extend(accepted_rollouts)
                pbar.update(len(accepted_rollouts))

                await self.schedule_group_rollout(client)

            self.logger.debug(
                f"Got {len(batch_rollouts)} rollout(s) in batch. Need {self.config.batch_size - len(batch_rollouts)} more."
            )

        return batch_rollouts

    @property
    def max_off_policy_level(self) -> int:
        if not self.inflight_group_rollouts:
            return 0
        return max(retention_step for retention_step, _ in self.inflight_group_rollouts.values())

    @property
    def min_off_policy_level(self) -> int:
        if not self.inflight_group_rollouts:
            return 0
        return min(retention_step for retention_step, _ in self.inflight_group_rollouts.values())

    @property
    def mean_off_policy_level(self) -> float:
        if not self.inflight_group_rollouts:
            return 0
        retention_steps = [retention_step for retention_step, _ in self.inflight_group_rollouts.values()]
        return sum(retention_steps) / len(retention_steps)

    @property
    def async_level(self) -> int:
        return self.step - self.ckpt_step

    def get_metrics(self) -> dict[str, float]:
        return {
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "batch/async_level": self.async_level,
            "batch/off_policy_level/max": self.max_off_policy_level,
            "batch/off_policy_level/mean": self.mean_off_policy_level,
            "batch/off_policy_level/min": self.min_off_policy_level,
        }
