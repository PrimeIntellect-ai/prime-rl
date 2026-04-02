from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple, cast

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.utils import get_sampling_args
from prime_rl.orchestrator.vf_utils import get_seq_len, run_rollout
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
    """Metadata for an in-flight request."""

    off_policy_steps: int
    client_config: vf.ClientConfig
    task: str
    group_id: int | None = None


@dataclass
class GroupState:
    """Tracks the state of a rollout group (one example × N rollouts)."""

    example: dict
    rollouts_to_schedule: int
    completed_rollouts: list[vf.RolloutOutput] = field(default_factory=list)
    pinned_client: vf.ClientConfig | None = None


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
        enable_policy_updates: bool = True,
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
        self.enable_policy_updates = enable_policy_updates
        self.lora_name = lora_name
        initial_temp = compute_temperature(step=0, sampling_config=config.sampling, max_steps=config.max_steps)
        is_vllm = config.teacher_rollout_model is None
        self.sampling_args = get_sampling_args(config.sampling, temperature=initial_temp, is_vllm=is_vllm)
        self.model_name = self.config.model.name
        self.json_logging = config.log.json_logging

        # Inference pool - used for admin operations (adapter sync) and metrics
        self.inference_pool = inference_pool

        self.max_retries_by_task = {env.resolved_name: env.max_retries for env in config.env}
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
        self.update_policy_task: asyncio.Task | None = None
        self.inflight_policy_update_task: asyncio.Task | None = None
        self.policy_update_lock = asyncio.Lock()
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_task: dict[str, int] = defaultdict(int)
        self.errored_rollouts_by_task: dict[str, int] = defaultdict(int)
        self.total_rollouts_by_task: dict[str, int] = defaultdict(int)
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
        count = len(self.inflight_requests)
        await safe_cancel_all(list(self.inflight_requests))
        self.inflight_requests.clear()
        self.groups.clear()
        self.cancelled_rollouts_count += count

    @staticmethod
    def _client_identity(c: vf.ClientConfig) -> tuple[str, str | None]:
        return (c.api_base_url, c.extra_headers.get("X-data-parallel-rank"))

    async def _select_least_loaded_client(self) -> vf.ClientConfig:
        """Select the client with the fewest in-flight tasks.

        Uses (api_base_url, dp_rank) as identity rather than client_idx so that
        load tracking survives elastic pool refreshes (which reassign indices).
        """
        clients = self.inference_pool.clients
        while not clients:
            await asyncio.sleep(1)
            clients = self.inference_pool.clients
        inflight = Counter(self._client_identity(info.client_config) for info in self.inflight_requests.values())
        return min(clients, key=lambda c: inflight[self._client_identity(c)])

    async def drop_group(self, group_id: int) -> int:
        """Drop a group and cancel any remaining in-flight rollouts for it."""
        tasks_to_cancel = []
        for task, info in list(self.inflight_requests.items()):
            if info.group_id != group_id:
                continue
            self.inflight_requests.pop(task, None)
            tasks_to_cancel.append(task)
        self.groups.pop(group_id, None)
        await safe_cancel_all(tasks_to_cancel)
        return len(tasks_to_cancel)

    async def schedule_rollout(self, group_id: int):
        """Asynchronously schedules a rollout request."""
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        group = self.groups.get(group_id)
        if group is None or group.rollouts_to_schedule <= 0:
            return
        group.rollouts_to_schedule -= 1
        if group.pinned_client is not None:
            client_config = group.pinned_client
        else:
            client_config = await self._select_least_loaded_client()
            if group_id not in self.groups:
                return
            group.pinned_client = client_config
        run_rollout_task = asyncio.create_task(
            run_rollout(
                env=self.env,
                client=client_config,
                example=group.example,
                model_name=self.model_name,
                sampling_args=self.sampling_args,
                max_retries=self.max_retries_by_task.get(group.example["task"], 0),
            )
        )
        self.inflight_requests[run_rollout_task] = InflightRolloutInfo(
            off_policy_steps=0, client_config=client_config, task=group.example["task"], group_id=group_id
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
            await self.maybe_update_policy()
            await asyncio.sleep(1)

    def _compute_next_ckpt_step(self) -> int:
        latest_ckpt_step = get_latest_ckpt_step(get_broadcast_dir(self.config.output_dir)) or 0
        async_away_ckpt_step = max(self.step - self.max_async_level, 0)
        if self.strict_async_level:
            return async_away_ckpt_step
        return max(async_away_ckpt_step, latest_ckpt_step)

    async def _apply_policy_update(self, next_ckpt_step: int) -> None:
        async_away_ckpt_step = max(self.step - self.max_async_level, 0)
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

        update_weights_start_time = time.perf_counter()
        weights_path = get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step)
        await self.inference_pool.update_weights(weights_path, lora_name=self.lora_name, step=next_ckpt_step)
        self.update_weights_time = time.perf_counter() - update_weights_start_time
        self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

        self.ckpt_step = next_ckpt_step
        if self.lora_name is not None:
            self.model_name = self.lora_name
            self.inference_pool.update_model_name(self.model_name)

        self.checkpoint_ready.set()
        await self._update_off_policy()

    async def _get_or_start_policy_update_task(self, next_ckpt_step: int) -> asyncio.Task:
        async with self.policy_update_lock:
            task = self.inflight_policy_update_task
            if task is not None and not task.done():
                return task

            task = asyncio.create_task(self._apply_policy_update(next_ckpt_step))
            self.inflight_policy_update_task = task

            def _clear_inflight_policy_update(done_task: asyncio.Task) -> None:
                if self.inflight_policy_update_task is done_task:
                    self.inflight_policy_update_task = None

            task.add_done_callback(_clear_inflight_policy_update)
            return task

    async def maybe_update_policy(self):
        """Updates the policy to the latest available checkpoint. Aborts rollout requests that are older than the max retention steps."""
        if not self.enable_policy_updates:
            self.ckpt_step = self.step
            self.checkpoint_ready.set()
            return

        while True:
            next_ckpt_step = self._compute_next_ckpt_step()
            if next_ckpt_step <= self.ckpt_step:
                return

            task = await self._get_or_start_policy_update_task(next_ckpt_step)
            await asyncio.shield(task)

    async def _update_off_policy(self) -> None:
        stale_group_ids = {
            info.group_id
            for info in self.inflight_requests.values()
            if info.group_id is not None and info.off_policy_steps >= self.max_off_policy_steps
        }
        tasks_to_increment = [
            task
            for task, info in list(self.inflight_requests.items())
            if info.group_id is None or info.group_id not in stale_group_ids
        ]

        counts = await asyncio.gather(*(self.drop_group(gid) for gid in stale_group_ids))
        removed = sum(counts)
        for task in tasks_to_increment:
            info = self.inflight_requests.get(task)
            if info is None:
                continue
            self.inflight_requests[task] = info._replace(off_policy_steps=info.off_policy_steps + 1)

        self.cancelled_rollouts_count += removed
        if removed:
            self.logger.warning(
                f"Cancelled {removed} old rollout requests (will refill naturally). "
                f"Consider increasing max_off_policy_steps to avoid this."
            )

    def _should_defer_group_scoring(self, task: str) -> bool:
        return task in self.deferred_group_scoring_tasks and self.config.verification.enabled

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

        if self.enable_policy_updates:
            # Cancel the previous update policy task to avoid concurrent updates
            if self.update_policy_task is not None:
                await safe_cancel(self.update_policy_task)

            # Manually check the async barrier before starting the step, then re-create the update policy loop
            # This ensures that we respect max_async_level, while still listening for policy updates mid-step
            await self.maybe_update_policy()
            self.update_policy_task = asyncio.create_task(self.update_policy_loop())
        else:
            self.ckpt_step = step
            self.checkpoint_ready.set()

        batch_start_time = time.perf_counter()

        self.logger.debug("Starting to generate batch rollouts")

        batch_rollouts: list[vf.RolloutOutput] = []
        batch_progress = 0
        pbar = ProgressTracker(
            total=self.batch_target, desc="Generating rollouts (train)", json_logging=self.json_logging, step=step
        )

        while batch_progress < self.batch_target:
            await self._fill_inflight_requests()
            inflight_tasks = list(self.inflight_requests.keys())

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
                    rollout = finished_task.result()

                    task = rollout_info.task
                    self.total_rollouts_by_task[task] += 1
                    should_reschedule = False
                    if len(rollout["trajectory"]) == 0:
                        self.empty_rollouts_by_task[task] += 1
                        should_reschedule = True
                        self.logger.warning(
                            f"Empty trajectory in group {group_id} ({task}), re-scheduling "
                            f"({len(group.completed_rollouts)}/{self.rollouts_per_example} complete)"
                        )
                    if rollout["error"] is not None:
                        self.errored_rollouts_by_task[task] += 1
                        should_reschedule = True
                        self.logger.warning(
                            f"Rollout error in group {group_id} ({task}), re-scheduling "
                            f"({len(group.completed_rollouts)}/{self.rollouts_per_example} complete): "
                            f"{rollout['error']['error_chain_repr']}"
                        )
                    if should_reschedule:
                        group.rollouts_to_schedule += 1
                        continue

                    group.completed_rollouts.append(rollout)
                    if len(group.completed_rollouts) < self.rollouts_per_example:
                        continue
                    completed_rollouts = self.groups.pop(group_id).completed_rollouts
                    completed_rollouts = await self._score_group_if_deferred(completed_rollouts)
                except asyncio.CancelledError:
                    if group_id is not None:
                        await self.drop_group(group_id)
                    continue
                except Exception as e:
                    self.logger.warning(f"Rollout failed: {e}")
                    if group_id is not None:
                        await self.drop_group(group_id)
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

    async def stop(self) -> None:
        await self.cancel_inflight_rollouts()
        if self.update_policy_task is not None:
            await safe_cancel(self.update_policy_task)
            self.update_policy_task = None
        if self.inflight_policy_update_task is not None:
            await safe_cancel(self.inflight_policy_update_task)
            self.inflight_policy_update_task = None

    @property
    def max_off_policy_level(self) -> int:
        steps = [info.off_policy_steps for info in self.inflight_requests.values()]
        if not steps:
            return 0
        return max(steps)

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
        total_rollouts = sum(self.total_rollouts_by_task.values())
        metrics = {
            "time/wait_for_ckpt": self.wait_for_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "scheduler/async_level": self.async_level,
            "scheduler/inflight_rollouts": self.inflight_rollout_count,
            "scheduler/inflight_samples": self.inflight_sample_count,
            "scheduler/cancelled_rollouts": self.cancelled_rollouts_count,
            "empty_rollouts/all": sum(self.empty_rollouts_by_task.values()) / max(total_rollouts, 1),
            "errored_rollouts/all": sum(self.errored_rollouts_by_task.values()) / max(total_rollouts, 1),
            "off_policy_level/all/max": self.max_off_policy_level,
            "off_policy_level/all/mean": self.mean_off_policy_level,
        }
        for task in self.total_rollouts_by_task:
            task_total = max(self.total_rollouts_by_task[task], 1)
            metrics[f"empty_rollouts/{task}"] = self.empty_rollouts_by_task.get(task, 0) / task_total
            metrics[f"errored_rollouts/{task}"] = self.errored_rollouts_by_task.get(task, 0) / task_total
        by_task: dict[str, list[int]] = {}
        for info in self.inflight_requests.values():
            by_task.setdefault(info.task, []).append(info.off_policy_steps)
        for task, steps in by_task.items():
            metrics[f"off_policy_level/{task}/max"] = max(steps)
            metrics[f"off_policy_level/{task}/mean"] = sum(steps) / len(steps)
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_task.clear()
        self.errored_rollouts_by_task.clear()
        self.total_rollouts_by_task.clear()

        # Add inference pool metrics (e.g. elastic pool server counts)
        metrics.update(self.inference_pool.get_metrics())

        return metrics
