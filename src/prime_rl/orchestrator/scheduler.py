import asyncio
import time
from abc import ABC, abstractmethod
from itertools import cycle

from httpx import AsyncClient
from openai import AsyncOpenAI
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from verifiers import Environment
from verifiers.types import GenerateOutputs, ProcessedOutputs

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.buffer import Buffer, Rollout, make_rollouts
from prime_rl.orchestrator.config import (
    ARealSchedulerConfig,
    DefaultSchedulerConfig,
    OrchestratorConfig,
    SamplingConfig,
)
from prime_rl.orchestrator.utils import parse_is_truncated_completions
from prime_rl.utils.client import update_weights
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_latest_ckpt_step, get_step_path, get_weights_dir, wait_for_path
from prime_rl.utils.vf import generate_batch, generate_group


class Scheduler(ABC):
    """
    Abstract base class for schedulers. They are responsible for scheduling rollout and weight update requests.
    """

    def __init__(
        self,
        clients: list[AsyncOpenAI],
        admin_clients: list[AsyncClient],
        env: Environment,
        buffer: Buffer,
        tokenizer: PreTrainedTokenizerFast,
        config: OrchestratorConfig,
    ):
        self.logger = get_logger()
        self.clients = clients
        self.admin_clients = admin_clients
        self.env = env
        self.buffer = buffer
        self.tokenizer = tokenizer
        self.config = config
        self.oversampling_factor = config.scheduler.oversampling_factor
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.rollouts_per_example = config.rollouts_per_example
        self.problems_per_batch = int(self.oversampling_factor * self.batch_size // self.rollouts_per_example)
        self.max_concurrent = config.max_concurrent
        self.semaphore = asyncio.Semaphore(self.max_concurrent) if self.max_concurrent is not None else None
        self.ckpt_step = 0

        def prepare_sampling_args(sampling_config: SamplingConfig) -> dict:
            sampling_args = dict(sampling_config)
            # Convert SamplingConfig to vLLM OAI sampling args
            # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_2
            sampling_args["top_p"] = 1.0
            sampling_args["logprobs"] = True
            sampling_args["extra_body"] = {
                "return_tokens_as_token_ids": True,
                "top_k": -1,
                "min_p": 0.0,
            }
            sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")
            sampling_args["extra_body"]["repetition_penalty"] = sampling_args.pop("repetition_penalty")
            return sampling_args

        self.sampling_args = prepare_sampling_args(config.sampling)

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
            problem_ids=generate_outputs.example_id,
            prompt_tokens=processed_outputs.prompt_ids,
            prompt_masks=processed_outputs.prompt_mask,
            completion_tokens=processed_outputs.completion_ids,
            completion_masks=processed_outputs.completion_mask,
            completion_logprobs=processed_outputs.completion_logprobs,
            is_truncated=is_truncated,
            rewards=processed_outputs.rewards,
            advantages=advantages,
        )

        # Update and sample rollouts from the buffer
        self.buffer.update(rollouts)
        num_problems = len(set(generate_outputs.example_id))
        accepted_rollouts = self.buffer.sample_rollouts(n=num_problems)

        return accepted_rollouts

    @abstractmethod
    async def generate_batch(self, step: int) -> list[Rollout]:
        """Orchestrates rollout and update weight requests until a batch of rollouts is ready."""
        pass

    @abstractmethod
    def metrics(self) -> dict:
        """Exports metrics for monitoring."""
        pass


class DefaultScheduler(Scheduler):
    """Schedules all batch rollout requests, awaits them and loops, if necessary. Updates policy in between training steps."""

    def __init__(
        self,
        clients: list[AsyncOpenAI],
        admin_clients: list[AsyncClient],
        env: Environment,
        buffer: Buffer,
        tokenizer: PreTrainedTokenizerFast,
        config: OrchestratorConfig,
        scheduler_config: DefaultSchedulerConfig,
    ):
        super().__init__(clients, admin_clients, env, buffer, tokenizer, config)
        self.step = 0
        self.max_off_policy_steps = scheduler_config.max_off_policy_steps
        # Metrics
        self.wait_for_weight_ckpt_time = 0
        self.update_weights_time = 0

    async def update_policy(self, step: int):
        """Updates the policy to be exactly off-policy step away."""
        next_ckpt_step = max(step - self.max_off_policy_steps, 0)
        if step - self.ckpt_step > self.max_off_policy_steps:
            self.logger.debug(
                f"Hit async barrier because we are >{self.max_off_policy_steps} steps off-policy. Waiting for weight checkpoint {next_ckpt_step}"
            )
            wait_for_weight_ckpt_start_time = time.time()
            await wait_for_path(get_step_path(get_weights_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
            self.wait_for_weight_ckpt_time = time.time() - wait_for_weight_ckpt_start_time
            self.logger.debug(
                f"Waited for weight checkpoint {next_ckpt_step} for {self.wait_for_weight_ckpt_time:.2f}s"
            )
            self.logger.debug(f"Updating weights to step {next_ckpt_step}")
            update_weights_start_time = time.time()
            await update_weights(
                self.admin_clients, get_step_path(get_weights_dir(self.config.output_dir), next_ckpt_step)
            )
            self.update_weights_time = time.time() - update_weights_start_time
            self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")
            self.ckpt_step = next_ckpt_step

    async def _generate_batch(self) -> list[Rollout]:
        """Schedules and awaits all batch rollout requests synchronously."""
        batch_rollouts: list[Rollout] = []
        problems_left = self.problems_per_batch
        while True:
            generate_outputs = await generate_batch(
                clients=self.clients,
                env=self.env,
                model_name=self.config.model.name,
                problems=self.buffer.sample_problems(problems_left),
                rollouts_per_example=self.config.rollouts_per_example,
                sampling_args=self.sampling_args,
                semaphore=self.semaphore,
            )

            # Process outputs and update accepted rollouts
            accepted_rollouts = self.process_generate_outputs(generate_outputs=generate_outputs)
            batch_rollouts.extend(accepted_rollouts)

            # Break if we have enough rollouts to fill the batch
            if len(batch_rollouts) >= self.config.batch_size:
                batch_rollouts = batch_rollouts[: self.config.batch_size]
                break

            # On next iteration, sample the remaining problems to fill the batch
            problems_sampled = len(batch_rollouts) // self.config.rollouts_per_example
            problems_left = self.problems_per_batch - problems_sampled

        return batch_rollouts

    async def generate_batch(self, step: int) -> list[Rollout]:
        """Updates the policy at the beginning of the step and synchronously generates a batch of rollouts."""
        self.step = step
        await self.update_policy(step=step)
        return await self._generate_batch()

    @property
    def off_policy_level(self) -> int:
        return max(self.step - self.ckpt_step, 0)

    def metrics(self) -> dict:
        return {
            "batch/off_policy_level": self.off_policy_level,
            "time/wait_for_weight_ckpt": self.wait_for_weight_ckpt_time,
            "time/update_weights": self.update_weights_time,
        }


class ARealScheduler(Scheduler):
    """Asynchronously schedules group rollout requests and re-schedules them as they complete (continuous batching). Updates policy in between group rollout requests."""

    def __init__(
        self,
        clients: list[AsyncOpenAI],
        admin_clients: list[AsyncClient],
        env: Environment,
        buffer: Buffer,
        tokenizer: PreTrainedTokenizerFast,
        config: OrchestratorConfig,
        scheduler_config: ARealSchedulerConfig,
    ):
        super().__init__(clients, admin_clients, env, buffer, tokenizer, config)
        self.step = 0
        self.max_off_policy_steps = scheduler_config.max_off_policy_steps
        self.max_retention_steps = scheduler_config.max_retention_steps
        self.inflight_group_rollouts: dict[asyncio.Task, int] = {}
        self.cycle_clients = cycle(self.clients)
        self.update_weights_time = 0
        self.wait_for_weight_ckpt_time = 0
        asyncio.create_task(self.update_policy_loop())

    async def schedule_group_rollout(self):
        """Asynchronously schedules a group rollout request."""
        problem = self.buffer.sample_problems(n=1)[0]
        group_rollout_request = asyncio.create_task(
            generate_group(
                client=next(self.cycle_clients),
                env=self.env,
                model_name=self.config.model.name,
                problem=problem,
                rollouts_per_example=self.config.rollouts_per_example,
                sampling_args=self.sampling_args,
                semaphore=self.semaphore,
            )
        )
        await asyncio.sleep(0)
        self.inflight_group_rollouts[group_rollout_request] = 0

    async def update_policy_loop(self):
        """Loops updating the policy at a fixed interval."""
        while True:
            await self.update_policy()
            await asyncio.sleep(1)

    async def update_policy(self):
        """Updates the policy to the latest available checkpoint. Aborts rollout requests that are older than the max retention steps."""
        latest_ckpt_step = get_latest_ckpt_step(get_weights_dir(self.config.output_dir)) or 0
        fixed_off_policy_step = max(self.step - self.max_off_policy_steps, 0)
        next_ckpt_step = max(fixed_off_policy_step, latest_ckpt_step)
        if next_ckpt_step > self.ckpt_step:
            if next_ckpt_step == latest_ckpt_step:
                self.logger.debug(
                    f"Found new policy with step {next_ckpt_step}. Updating weights and cancelling old rollout requests."
                )
            else:
                self.logger.debug(
                    f"Hit async barrier because we are >{self.max_off_policy_steps} steps off-policy. Waiting for weight checkpoint {next_ckpt_step}"
                )
            wait_for_weight_ckpt_start_time = time.time()
            await wait_for_path(get_step_path(get_weights_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
            self.wait_for_weight_ckpt_time = time.time() - wait_for_weight_ckpt_start_time
            self.logger.debug(
                f"Waited for weight checkpoint {next_ckpt_step} for {self.wait_for_weight_ckpt_time:.2f}s"
            )

            update_weights_start_time = time.time()
            await update_weights(
                self.admin_clients, get_step_path(get_weights_dir(self.config.output_dir), next_ckpt_step)
            )
            self.update_weights_time = time.time() - update_weights_start_time
            self.logger.debug(f"Updated weights to step {next_ckpt_step} in {self.update_weights_time:.2f}s")

            # Cancel old rollout requests
            num_old_rollout_requests = 0
            for task, retention_step in self.inflight_group_rollouts.items():
                if retention_step > self.max_retention_steps:
                    task.cancel()
                    self.inflight_group_rollouts.pop(task)
                    await self.schedule_group_rollout()
                    num_old_rollout_requests += 1
                else:
                    self.inflight_group_rollouts[task] = retention_step + 1
            if num_old_rollout_requests > 0:
                self.logger.warning(f"Cancelled and re-scheduled {num_old_rollout_requests} old rollout requests.")
            self.ckpt_step = latest_ckpt_step

    async def generate_batch(self, step: int) -> list[Rollout]:
        """Generates group rollouts continuously until the batch is filled. Updates the policy on the fly."""
        self.step = step

        # Schedule initial tasks
        self.logger.info("Starting to generate batch rollouts")
        while len(self.inflight_group_rollouts) < self.problems_per_batch:
            await self.schedule_group_rollout()

        batch_rollouts: list[Rollout] = []
        while len(batch_rollouts) < self.config.batch_size:
            finished_group_rollouts, _ = await asyncio.wait(
                self.inflight_group_rollouts, return_when=asyncio.FIRST_COMPLETED
            )

            for finished_group_rollout in finished_group_rollouts:
                if len(batch_rollouts) >= self.config.batch_size:
                    batch_rollouts = batch_rollouts[: self.config.batch_size]
                    break

                self.inflight_group_rollouts.pop(finished_group_rollout)
                generate_outputs: GenerateOutputs = finished_group_rollout.result()

                accepted_rollouts = self.process_generate_outputs(generate_outputs=generate_outputs)
                batch_rollouts.extend(accepted_rollouts)

                await self.schedule_group_rollout()

            self.logger.debug(
                f"Got {len(batch_rollouts)} rollout(s) in batch. Need {self.config.batch_size - len(batch_rollouts)} more."
            )

        return batch_rollouts

    @property
    def max_retention_level(self) -> int:
        return max(self.inflight_group_rollouts.values())

    @property
    def off_policy_level(self) -> int:
        return max(self.step - self.ckpt_step, 0)

    def metrics(self) -> dict:
        return {
            "time/wait_for_weight_ckpt": self.wait_for_weight_ckpt_time,
            "time/update_weights": self.update_weights_time,
            "batch/off_policy_level": self.off_policy_level,
            "batch/max_retention_level": self.max_retention_level,
        }


def setup_scheduler(
    clients: list[AsyncOpenAI],
    admin_clients: list[AsyncClient],
    env: Environment,
    buffer: Buffer,
    tokenizer: PreTrainedTokenizerFast,
    config: OrchestratorConfig,
) -> Scheduler:
    if config.scheduler.type == "default":
        return DefaultScheduler(clients, admin_clients, env, buffer, tokenizer, config, config.scheduler)
    elif config.scheduler.type == "areal":
        return ARealScheduler(clients, admin_clients, env, buffer, tokenizer, config, config.scheduler)
    else:
        raise ValueError(f"Invalid scheduler type: {config.scheduler}")
