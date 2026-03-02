from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple, cast

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.orchestrator.vf_utils import get_seq_len, run_rollout
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
    """Metadata for an in-flight request."""

    off_policy_steps: int
    client_config: vf.ClientConfig
    group_id: int | None = None


@dataclass
class GroupState:
    """Tracks the state of a rollout group (one example × N rollouts)."""

    example: dict
    rollouts_to_schedule: int
    completed_rollouts: list[vf.RolloutOutput] = field(default_factory=list)


class Scheduler:
    """
    Asynchronously manages scheduling of rollout requests and policy updates.
    Keeps a constant number of rollouts in-flight (continuous batching) and
    updates the policy as soon as it becomes available.

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
        deferred_group_scoring_tasks: set[str] | None = None,
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
        self.max_inflight_rollouts = max_inflight_rollouts
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

        self.deferred_group_scoring_tasks = set(deferred_group_scoring_tasks or ())
        if self.deferred_group_scoring_tasks:
            task_list = ", ".join(sorted(self.deferred_group_scoring_tasks))
            self.logger.info(f"Deferred group scoring active for task(s): {task_list}")

        # Track in-flight requests: task -> info
        self.inflight_requests: dict[asyncio.Task, InflightRolloutInfo] = {}

        # Track in-progress groups while rollouts are generated independently.
        self.next_group_id = 0
        self.groups: dict[int, GroupState] = {}

        self.step, self.ckpt_step = 0, 0
        self.checkpoint_ready = asyncio.Event()
        self.checkpoint_ready.set()
        self.update_weights_time, self.wait_for_ckpt_time = 0, 0
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

    def cancel_inflight_rollouts(self):
        """Cancel all in-flight rollout requests."""
        count = len(self.inflight_requests)
        for task in list(self.inflight_requests):
            task.cancel()
        self.inflight_requests.clear()
        self.groups.clear()
        self.cancelled_rollouts_count += count

    async def _select_least_loaded_client(self) -> vf.ClientConfig:
        """Select the client with the fewest in-flight tasks."""
        clients = self.inference_pool.clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.inference_pool.clients
        inflight_by_url = Counter()
        for info in self.inflight_requests.values():
            inflight_by_url[info.client_config.api_base_url] += 1
        return min(clients, key=lambda c: inflight_by_url[c.api_base_url])

    def drop_group(self, group_id: int) -> int:
        """Drop a group and cancel any remaining in-flight rollouts for it."""
        removed = 0
        for task, info in list(self.inflight_requests.items()):
            if info.group_id != group_id:
                continue
            task.cancel()
            self.inflight_requests.pop(task, None)
            removed += 1
        self.groups.pop(group_id, None)
        return removed

    async def schedule_rollout(self, group_id: int):
        """Asynchronously schedules a rollout request."""
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        group = self.groups.get(group_id)
        if group is None or group.rollouts_to_schedule <= 0:
            return
        group.rollouts_to_schedule -= 1
        client_config = await self._select_least_loaded_client()
        if group_id not in self.groups:
            return
        run_rollout_task = asyncio.create_task(
            run_rollout(
                env=self.env,
                client=client_config,
                example=group.example,
                model_name=self.model_name,
                sampling_args=self.sampling_args,
                max_retries=0,  # TODO: make configurable
            )
        )
        self.inflight_requests[run_rollout_task] = InflightRolloutInfo(
            off_policy_steps=0, client_config=client_config, group_id=group_id
        )

    @property
    def inflight_rollout_count(self) -> int:
        return len(self.inflight_requests)

    @property
    def inflight_sample_count(self) -> int:
        return self.inflight_rollout_count + sum(g.rollouts_to_schedule for g in self.groups.values())

    async def _schedule_next_request(self) -> bool:
        remaining_capacity = self.max_inflight_rollouts - self.inflight_rollout_count

        if remaining_capacity <= 0:
            return False

        for group_id, group in self.groups.items():
            if group.rollouts_to_schedule > 0:
                await self.schedule_rollout(group_id=group_id)
                return True

        example = self.buffer.sample_examples(n=1)[0]
        group_id = self.next_group_id
        self.next_group_id += 1
        self.groups[group_id] = GroupState(example=example, rollouts_to_schedule=self.rollouts_per_example)
        await self.schedule_rollout(group_id=group_id)
        return True

    async def _fill_inflight_requests(self) -> None:
        while await self._schedule_next_request():
            pass

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
                self.inference_pool.update_model_name(self.model_name)

            self.checkpoint_ready.set()

            self._update_off_policy()
            self.ckpt_step = next_ckpt_step

    def _update_off_policy(self) -> None:
        stale_group_ids = {
            info.group_id
            for info in self.inflight_requests.values()
            if info.group_id is not None and info.off_policy_steps > self.max_off_policy_steps
        }
        removed = sum(self.drop_group(gid) for gid in stale_group_ids)
        for task, info in list(self.inflight_requests.items()):
            self.inflight_requests[task] = info._replace(off_policy_steps=info.off_policy_steps + 1)

        self.cancelled_rollouts_count += removed
        if removed:
            self.logger.warning(
                f"Cancelled {removed} old rollout requests (will refill naturally). "
                f"Consider increasing max_off_policy_steps to avoid this."
            )

    def _should_defer_group_scoring(self, task: str) -> bool:
        return task in self.deferred_group_scoring_tasks and not self.config.buffer.skip_verification

    async def _score_group_if_deferred(self, completed_rollouts: list[vf.RolloutOutput]) -> list[vf.RolloutOutput]:
        if not completed_rollouts:
            return completed_rollouts
        task = completed_rollouts[0]["task"]
        if not self._should_defer_group_scoring(task):
            return completed_rollouts
        env_for_task = self.env.get_env_for_task(task)
        await env_for_task.rubric.score_group(cast(list[vf.State], completed_rollouts))
        return completed_rollouts

    async def generate_batch(self, step: int) -> list[vf.RolloutOutput]:
        """Continuously generates a batch of rollouts."""
        self.step = step
        batch_start_time = time.perf_counter()

        self.logger.debug("Starting to generate batch rollouts")
        await self._fill_inflight_requests()

        batch_rollouts: list[vf.RolloutOutput] = []
        batch_progress = 0
        pbar = ProgressTracker(
            total=self.batch_target, desc="Generating rollouts (train)", json_logging=self.json_logging, step=step
        )

        while batch_progress < self.batch_target:
            inflight_tasks = list(self.inflight_requests.keys())
            if not inflight_tasks:
                await self._fill_inflight_requests()
                inflight_tasks = list(self.inflight_requests.keys())
                if not inflight_tasks:
                    raise RuntimeError("No in-flight rollout requests and could not schedule new requests.")
                continue

            finished_tasks, _ = await asyncio.wait(
                inflight_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            await self.checkpoint_ready.wait()

            for finished_task in finished_tasks:
                if batch_progress >= self.batch_target:
                    break

                rollout_info = self.inflight_requests.pop(finished_task, None)
                if rollout_info is None:
                    continue

                group_id = rollout_info.group_id

                try:
                    group = self.groups.get(group_id)
                    if group is None:
                        continue
                    group.completed_rollouts.append(finished_task.result())
                    if len(group.completed_rollouts) < self.rollouts_per_example:
                        continue
                    completed_rollouts = self.groups.pop(group_id).completed_rollouts
                    completed_rollouts = await self._score_group_if_deferred(completed_rollouts)
                except asyncio.CancelledError:
                    if group_id is not None:
                        self.drop_group(group_id)
                    continue
                except Exception as e:
                    self.logger.warning(f"Rollout failed: {e}")
                    if group_id is not None:
                        self.drop_group(group_id)
                    continue

                self.buffer.update(completed_rollouts)
                accepted_rollouts = self.buffer.sample_rollouts(n=self.rollouts_per_example)
                batch_rollouts.extend(accepted_rollouts)
                progress_increment = self.get_batch_progress_increment(accepted_rollouts)
                batch_progress += progress_increment
                pbar.update(progress_increment)

            await self._fill_inflight_requests()

        batch_rollouts = self.finalize_batch_rollouts(batch_rollouts)
        pbar.close()
        self.last_batch_generation_time = time.perf_counter() - batch_start_time
        return batch_rollouts

    @property
    def max_off_policy_level(self) -> int:
        steps = [info.off_policy_steps for info in self.inflight_requests.values()]
        if not steps:
            return 0
        return max(steps)

    @property
    def min_off_policy_level(self) -> int:
        steps = [info.off_policy_steps for info in self.inflight_requests.values()]
        if not steps:
            return 0
        return min(steps)

    @property
    def mean_off_policy_level(self) -> float:
        steps = [info.off_policy_steps for info in self.inflight_requests.values()]
        if not steps:
            return 0
        return sum(steps) / len(steps)

    @property
    def async_level(self) -> int:
        return self.step - self.ckpt_step

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "batch/async_level": self.async_level,
            "batch/inflight_rollouts": self.inflight_rollout_count,
            "batch/inflight_samples": self.inflight_sample_count,
            "batch/off_policy_level/max": self.max_off_policy_level,
            "batch/off_policy_level/mean": self.mean_off_policy_level,
            "batch/off_policy_level/min": self.min_off_policy_level,
            "batch/cancelled_rollouts": self.cancelled_rollouts_count,
        }
        self.cancelled_rollouts_count = 0

        # Add inference pool metrics (e.g. elastic pool server counts)
        metrics.update(self.inference_pool.get_metrics())

        return metrics
