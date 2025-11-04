import asyncio
import time
from itertools import cycle

from httpx import AsyncClient
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from verifiers import Environment
from verifiers.types import GenerateOutputs, ProcessedOutputs

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.buffer import Buffer, Rollout
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.utils import get_sampling_args, parse_is_truncated_completions
from prime_rl.utils.client import update_weights
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_latest_ckpt_step, get_step_path, get_weights_dir, sync_wait_for_path
from prime_rl.utils.vf import generate_group, make_rollouts


class Scheduler:
    """Asynchronously schedules group rollout requests and re-schedules them as they complete (continuous batching). Updates policy in between group rollout requests."""

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
        self.inflight_group_rollouts: dict[asyncio.Task, tuple[int, AsyncOpenAI]] = {}
        self.cycle_clients = cycle(self.clients)
        self.step, self.ckpt_step = 0, 0
        self.update_weights_time, self.wait_for_weight_ckpt_time = 0, 0
        self.sampling_args = get_sampling_args(config.sampling)

    def process_generate_outputs(
        self,
        generate_outputs: GenerateOutputs,
    ) -> list[Rollout]:
        processed_outputs: ProcessedOutputs = self.env.process_env_results_vllm(
            prompts=generate_outputs.prompt,
            completions=generate_outputs.completion,
            states=generate_outputs.state,
            rewards=generate_outputs.reward,
            processing_class=self.tokenizer,
            max_seq_len=self.seq_len,
            mask_env_responses=self.config.mask_env_responses,
            zero_truncated_completions=self.config.zero_truncated_completions,
            mask_truncated_completions=self.config.mask_truncated_completions,
        )

        # Compute advantages
        advantages = compute_advantages(
            rewards=processed_outputs.rewards,
            completion_lengths=list(map(len, processed_outputs.completion_ids)),
            samples_per_problem=self.config.rollouts_per_example,
            advantage_config=self.config.advantage,
        )

        # Parse whether the completions were truncated
        responses = [state["responses"] for state in generate_outputs.state]
        is_truncated = parse_is_truncated_completions(responses=responses)

        # Make rollouts
        rollouts = make_rollouts(
            generate_outputs,
            processed_outputs,
            generate_outputs.example_id,  # TODO: Bring back buffer ID here
            advantages,
            is_truncated,
        )

        # Update and sample rollouts from the buffer
        self.buffer.update(rollouts)
        num_problems = len(set(generate_outputs.example_id))
        accepted_rollouts = self.buffer.sample_rollouts(n=num_problems)

        return accepted_rollouts

    async def schedule_group_rollout(
        self, client: AsyncOpenAI | None = None, semaphore: asyncio.Semaphore | None = None
    ):
        """Asynchronously schedules a group rollout request."""
        problem = self.buffer.sample_problems(n=1)[0]
        if client is None:
            client = next(self.cycle_clients)
        group_rollout_request = asyncio.create_task(
            generate_group(
                client=client,
                env=self.env,
                model_name=self.config.model.name,
                problem=problem,
                rollouts_per_example=self.config.rollouts_per_example,
                sampling_args=self.sampling_args,
                semaphore=semaphore,
            )
        )
        await asyncio.sleep(0)
        self.inflight_group_rollouts[group_rollout_request] = (0, client)

    async def update_policy_loop(self):
        """Continuously checks for new policy checkpoints."""
        while True:
            await self.update_policy()
            await asyncio.sleep(1)

    async def update_policy(self):
        """Updates the policy to the latest available checkpoint. Aborts rollout requests that are older than the max retention steps."""
        latest_ckpt_step = get_latest_ckpt_step(get_weights_dir(self.config.output_dir)) or 0
        async_away_ckpt_step = max(self.step - self.max_async_level, 0)
        next_ckpt_step = max(async_away_ckpt_step, latest_ckpt_step)
        if next_ckpt_step > self.ckpt_step:
            if next_ckpt_step == async_away_ckpt_step:
                self.logger.debug(
                    f"Hit async barrier because we are >{self.max_async_level} steps async. Waiting for weight checkpoint {next_ckpt_step}"
                )
                wait_for_weight_ckpt_start_time = time.time()
                sync_wait_for_path(get_step_path(get_weights_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
                self.wait_for_weight_ckpt_time = time.time() - wait_for_weight_ckpt_start_time
                self.logger.debug(
                    f"Waited for weight checkpoint {next_ckpt_step} for {self.wait_for_weight_ckpt_time:.2f}s"
                )
            self.logger.debug(
                f"Got new policy with step {next_ckpt_step}. Updating weights and cancelling old rollout requests."
            )

            update_weights_start_time = time.time()
            await update_weights(
                self.admin_clients, get_step_path(get_weights_dir(self.config.output_dir), next_ckpt_step)
            )
            self.update_weights_time = time.time() - update_weights_start_time
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
                self.inflight_group_rollouts[task] = (off_policy_steps, client)
            if len(tasks_to_remove) > 0:
                self.logger.warning(f"Cancelled and re-scheduled {len(tasks_to_remove)} old rollout requests.")

            self.ckpt_step = latest_ckpt_step

    async def generate_batch(self, step: int, semaphore: asyncio.Semaphore | None = None) -> list[Rollout]:
        """Generates group rollouts continuously until the batch is filled. Updates the policy on the fly."""
        self.step = step

        # Schedule initial tasks
        self.logger.debug("Starting to generate batch rollouts")
        while len(self.inflight_group_rollouts) < self.problems_per_batch:
            await self.schedule_group_rollout()  # Schedule requests in round-robin fashion

        batch_rollouts: list[Rollout] = []
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
                generate_outputs: GenerateOutputs = finished_group_rollout.result()

                accepted_rollouts = self.process_generate_outputs(generate_outputs=generate_outputs)
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

    def metrics(self) -> dict:
        return {
            "time/wait_for_weight_ckpt": self.wait_for_weight_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "batch/async_level": self.async_level,
            "batch/off_policy_level/max": self.max_off_policy_level,
            "batch/off_policy_level/mean": self.mean_off_policy_level,
            "batch/off_policy_level/min": self.min_off_policy_level,
        }
