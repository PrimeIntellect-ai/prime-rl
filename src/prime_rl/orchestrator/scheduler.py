import asyncio
from abc import ABC, abstractmethod
from itertools import cycle

from httpx import AsyncClient
from openai import AsyncOpenAI
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from verifiers import Environment
from verifiers.types import GenerateOutputs, ProcessedOutputs

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.buffer import Buffer, Rollout, make_rollouts
from prime_rl.orchestrator.config import OrchestratorConfig, SamplingConfig
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
        self.clients, self.admin_clients, self.env, self.buffer, self.tokenizer, self.config = (
            clients,
            admin_clients,
            env,
            buffer,
            tokenizer,
            config,
        )
        self.problems_per_batch = self.config.batch_size // self.config.rollouts_per_example
        self.semaphore = (
            asyncio.Semaphore(self.config.max_concurrent) if self.config.max_concurrent is not None else None
        )

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
            max_seq_len=self.config.seq_len,
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
    async def update_policy(self, **kwargs):
        pass

    @abstractmethod
    async def generate_batch(self, **kwargs) -> list[Rollout]:
        pass

    @abstractmethod
    async def step(self, step: int) -> list[Rollout]:
        pass


class DefaultScheduler(Scheduler):
    """Schedules and awaits all batch rollout requests synchronously. Updates policy in between training steps."""

    def __init__(
        self,
        clients: list[AsyncOpenAI],
        admin_clients: list[AsyncClient],
        env: Environment,
        buffer: Buffer,
        tokenizer: PreTrainedTokenizerFast,
        config: OrchestratorConfig,
    ):
        super().__init__(clients, admin_clients, env, buffer, tokenizer, config)
        self.ckpt_step = 0
        self.off_policy_level = 0

    async def update_policy(self, step: int):
        """Updates the policy to be exactly off-policy step away."""
        next_ckpt_step = max(step - self.config.async_level, 0)
        if step - self.ckpt_step > self.config.async_level:
            await wait_for_path(get_step_path(get_weights_dir(self.config.output_dir), next_ckpt_step) / "STABLE")
            await update_weights(
                self.admin_clients, get_step_path(get_weights_dir(self.config.output_dir), next_ckpt_step)
            )
            self.ckpt_step = next_ckpt_step

        self.off_policy_level = step - self.ckpt_step

    async def generate_batch(self) -> list[Rollout]:
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

    async def step(self, step: int) -> list[Rollout]:
        """Updates the policy and generates a batch of rollouts."""
        await self.update_policy(step=step)
        return await self.generate_batch()


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
    ):
        super().__init__(clients, admin_clients, env, buffer, tokenizer, config)
        self.inflight_group_rollouts: list[asyncio.Task] = []
        self.cycle_clients = cycle(self.clients)

    async def schedule_task(self):
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
        self.inflight_group_rollouts.append(group_rollout_request)

    async def update_policy(self, step: int):
        """Updates the policy to the latest available checkpoint."""
        latest_ckpt_step = get_latest_ckpt_step(get_weights_dir(self.config.output_dir)) or 0

        if latest_ckpt_step > self.ckpt_step:
            await update_weights(
                self.admin_clients, get_step_path(get_weights_dir(self.config.output_dir), latest_ckpt_step)
            )
            self.ckpt_step = latest_ckpt_step
        self.off_policy_level = step - self.ckpt_step

    async def generate_batch(self, step: int) -> list[Rollout]:
        # Schedule initial tasks
        while len(self.inflight_group_rollouts) < self.problems_per_batch:
            await self.schedule_task()

        batch_rollouts: list[Rollout] = []
        while len(batch_rollouts) < self.config.batch_size:
            finished_group_rollouts, _ = await asyncio.wait(
                self.inflight_group_rollouts, return_when=asyncio.FIRST_COMPLETED
            )

            for finished_group_rollout in finished_group_rollouts:
                if len(batch_rollouts) >= self.config.batch_size:
                    batch_rollouts = batch_rollouts[: self.config.batch_size]
                    break

                self.inflight_group_rollouts.remove(finished_group_rollout)
                generate_outputs: GenerateOutputs = finished_group_rollout.result()

                accepted_rollouts = self.process_generate_outputs(generate_outputs=generate_outputs)
                batch_rollouts.extend(accepted_rollouts)

                await self.schedule_task()

        await self.update_policy(step=step)

        return batch_rollouts

    async def step(self, step: int) -> list[Rollout]:
        """Generates group rollouts continuously until the batch is filled."""
        return await self.generate_batch(step)


def setup_scheduler(
    clients: list[AsyncOpenAI],
    admin_clients: list[AsyncClient],
    env: Environment,
    buffer: Buffer,
    tokenizer: PreTrainedTokenizerFast,
    config: OrchestratorConfig,
) -> Scheduler:
    if config.scheduler == "default":
        return DefaultScheduler(clients, admin_clients, env, buffer, tokenizer, config)
    elif config.scheduler == "areal":
        return ARealScheduler(clients, admin_clients, env, buffer, tokenizer, config)
    else:
        raise ValueError(f"Invalid scheduler type: {config.scheduler}")
