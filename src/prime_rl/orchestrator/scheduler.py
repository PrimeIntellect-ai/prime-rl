import asyncio
import copy
import time
from itertools import cycle
from typing import NamedTuple

from httpx import AsyncClient
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from verifiers import Environment
from verifiers.types import RolloutInput, State, TrajectoryStep
from verifiers.utils.async_utils import NullAsyncContext

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.types import RolloutState, RolloutStep
from prime_rl.orchestrator.utils import get_sampling_args, get_semaphore
from prime_rl.utils.client import update_weights
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_latest_ckpt_step,
    get_step_path,
    sync_wait_for_path,
)


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
        self.inflight_group_rollouts: dict[asyncio.Task, InflightRolloutInfo] = {}
        self.cycle_clients = cycle(self.clients)
        self.step, self.ckpt_step = 0, 0
        self.update_weights_time, self.wait_for_ckpt_time = 0, 0
        self.sampling_args = get_sampling_args(config.sampling)
        semaphore = get_semaphore()
        self.generation_semaphore = semaphore if semaphore is not None else NullAsyncContext()
        self.score_semaphore = NullAsyncContext()

    def _problem_to_rollout_input(self, problem: dict) -> RolloutInput:
        rollout_input: RolloutInput = {
            "prompt": problem["prompt"],
            "example_id": problem["example_id"],
            "task": problem.get("task") or self.env.env_id or "default",
        }
        if "answer" in problem:
            rollout_input["answer"] = problem["answer"]
        if "info" in problem:
            rollout_input["info"] = problem["info"]
        return rollout_input

    def _trajectory_step_to_rollout_step(self, step: TrajectoryStep, fallback_reward: float) -> RolloutStep:
        tokens = step.get("tokens")
        if tokens is None:
            raise RuntimeError(
                "Trajectory step is missing token data. Ensure vLLM is configured to return token_ids/logprobs."
            )
        is_truncated = bool(tokens.get("is_truncated") or tokens.get("overlong_prompt"))
        completion_mask = list(tokens["completion_mask"])
        if is_truncated and self.config.mask_truncated_completions:
            completion_mask = [0] * len(completion_mask)
        step_reward = step.get("reward")
        if step_reward is None:
            step_reward = fallback_reward
        return RolloutStep(
            prompt_ids=list(tokens["prompt_ids"]),
            prompt_mask=list(tokens["prompt_mask"]),
            completion_ids=list(tokens["completion_ids"]),
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            is_truncated=is_truncated,
            reward=float(step_reward or 0.0),
        )

    def _state_to_rollout_state(self, state: State) -> RolloutState | None:
        trajectory: list[TrajectoryStep] = state.get("trajectory", [])
        if not trajectory:
            self.logger.warning("Received rollout state with empty trajectory. Skipping.")
            return None

        reward = float(state.get("reward", 0.0) or 0.0)
        metrics = dict(state.get("metrics") or {})
        steps: list[RolloutStep] = []
        any_truncated = False

        for trajectory_step in trajectory:
            rollout_step = self._trajectory_step_to_rollout_step(trajectory_step, reward)
            any_truncated = any_truncated or rollout_step["is_truncated"]
            steps.append(rollout_step)

        if any_truncated and self.config.zero_truncated_completions:
            reward = 0.0
            for step in steps:
                step["reward"] = 0.0

        return RolloutState(
            example_id=int(state["example_id"]),
            task=state.get("task", "default"),
            reward=reward,
            metrics=metrics,
            steps=steps,
            is_truncated=any_truncated,
            advantage=None,
        )

    def _prepare_rollouts(self, states: list[State]) -> list[RolloutState]:
        sanitized_rollouts: list[RolloutState] = []
        for state in states:
            rollout_state = self._state_to_rollout_state(state)
            if rollout_state is not None:
                sanitized_rollouts.append(rollout_state)

        if not sanitized_rollouts:
            return []

        self.buffer.update(sanitized_rollouts)
        num_problems = len({rollout["example_id"] for rollout in sanitized_rollouts})
        accepted_rollouts = self.buffer.sample_rollouts(n=num_problems * self.config.rollouts_per_example)

        if not accepted_rollouts:
            return []

        completion_lengths = [
            max(
                1,
                sum(len(step["completion_ids"]) for step in rollout["steps"]),
            )
            for rollout in accepted_rollouts
        ]
        advantages = compute_advantages(
            rewards=[rollout["reward"] for rollout in accepted_rollouts],
            completion_lengths=completion_lengths,
            samples_per_problem=self.config.rollouts_per_example,
            advantage_config=self.config.advantage,
        )
        for rollout, advantage in zip(accepted_rollouts, advantages):
            rollout["advantage"] = advantage

        return accepted_rollouts

    async def schedule_group_rollout(self, client: AsyncOpenAI | None = None):
        """Asynchronously schedules a group rollout request."""
        problem = self.buffer.sample_problems(n=1)[0]
        if client is None:
            client = next(self.cycle_clients)
        group_rollout_request = asyncio.create_task(
            self.env.run_group(
                group_inputs=[
                    copy.deepcopy(self._problem_to_rollout_input(problem)) for _ in range(self.rollouts_per_example)
                ],
                client=client,
                model=self.config.model.name,
                gen_sampling_args=self.sampling_args,
                gen_sem=self.generation_semaphore,
                score_sem=self.score_semaphore,
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
                self.admin_clients, get_step_path(get_broadcast_dir(self.config.output_dir), next_ckpt_step)
            )
            self.update_weights_time = time.perf_counter() - update_weights_start_time
            self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

            # Cancel old rollout requests
            tasks_to_remove = []
            tasks_to_update = []

            for task, (off_policy_steps, client) in self.inflight_group_rollouts.items():
                if off_policy_steps > self.max_off_policy_steps:
                    task.cancel()
                    tasks_to_remove.append((task, client))
                else:
                    tasks_to_update.append((task, off_policy_steps + 1, client))

            # Remove cancelled tasks
            for task, client in tasks_to_remove:
                self.inflight_group_rollouts.pop(task)
                await self.schedule_group_rollout(client)

            # Update retention steps for remaining tasks
            for task, off_policy_steps, client in tasks_to_update:
                self.inflight_group_rollouts[task] = InflightRolloutInfo(
                    off_policy_steps=off_policy_steps, client=client
                )
            if len(tasks_to_remove) > 0:
                self.logger.warning(f"Cancelled and re-scheduled {len(tasks_to_remove)} old rollout requests.")

            self.ckpt_step = next_ckpt_step

    async def generate_batch(self, step: int, semaphore: asyncio.Semaphore | None = None) -> list[RolloutState]:
        """Continuously schedules group rollouts, allowing them to be in-flight across steps."""
        self.step = step

        # Schedule initial tasks
        self.logger.debug("Starting to generate batch rollouts")
        while len(self.inflight_group_rollouts) < self.problems_per_batch:
            await self.schedule_group_rollout()  # Schedule requests in round-robin fashion

        batch_rollouts: list[RolloutState] = []
        pbar = tqdm(total=self.config.batch_size, desc="Generating rollouts (train)")
        while len(batch_rollouts) < self.config.batch_size:
            finished_group_rollouts, _ = await asyncio.wait(
                self.inflight_group_rollouts, return_when=asyncio.FIRST_COMPLETED
            )

            for finished_group_rollout in finished_group_rollouts:
                if len(batch_rollouts) >= self.config.batch_size:
                    batch_rollouts = batch_rollouts[: self.config.batch_size]
                    break

                _, client = self.inflight_group_rollouts.pop(finished_group_rollout)
                group_states: list[State] = finished_group_rollout.result()

                accepted_rollouts = self._prepare_rollouts(states=group_states)
                if accepted_rollouts:
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
