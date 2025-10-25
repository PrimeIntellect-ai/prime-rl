import asyncio
from abc import ABC, abstractmethod
from itertools import cycle

from openai import AsyncOpenAI
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from verifiers import Environment
from verifiers.types import GenerateOutputs, ProcessedOutputs

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.buffer import Buffer, Rollout, make_rollouts
from prime_rl.orchestrator.config import OrchestratorConfig, SamplingConfig
from prime_rl.orchestrator.utils import parse_is_truncated_completions
from prime_rl.utils.client import OAI_PRIORITY
from prime_rl.utils.logger import get_logger
from prime_rl.utils.vf import generate_batch, generate_group


class Scheduler(ABC):
    """Abstract base class for schedulers."""

    def __init__(
        self,
        clients: list[AsyncOpenAI],
        env: Environment,
        buffer: Buffer,
        tokenizer: PreTrainedTokenizerFast,
        config: OrchestratorConfig,
    ):
        self.logger = get_logger()
        self.clients, self.env, self.buffer, self.tokenizer = clients, env, buffer, tokenizer
        self.config = config
        self.oversampling_factor = config.scheduler.oversampling_factor
        self.batch_size = config.batch_size
        self.seq_len = config.seq_len
        self.rollouts_per_example = config.rollouts_per_example
        self.problems_per_batch = int(self.oversampling_factor * self.batch_size // self.rollouts_per_example)

        self.max_concurrent = config.max_concurrent
        self.semaphore = asyncio.Semaphore(self.max_concurrent) if self.max_concurrent is not None else None

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
            sampling_args["extra_body"]["priority"] = OAI_PRIORITY
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
    async def generate_batch(self) -> list[Rollout]:
        pass


class DefaultScheduler(Scheduler):
    def __init__(
        self,
        buffer: Buffer,
        env: Environment,
        tokenizer: PreTrainedTokenizerFast,
        config: OrchestratorConfig,
        clients: list[AsyncOpenAI],
    ):
        super().__init__(clients, env, buffer, tokenizer, config)

    async def generate_batch(self) -> list[Rollout]:
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


class ARealScheduler(Scheduler):
    """Scheduler for AREAL training."""

    def __init__(
        self,
        buffer: Buffer,
        env: Environment,
        tokenizer: PreTrainedTokenizerFast,
        config: OrchestratorConfig,
        clients: list[AsyncOpenAI],
    ):
        super().__init__(clients, env, buffer, tokenizer, config)
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

    async def generate_batch(self) -> list[Rollout]:
        # Schedule initial tasks
        self.logger.info("Starting to generate batch rollouts")
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

            self.logger.debug(
                f"Got {len(batch_rollouts)} rollouts in batch. Need {self.config.batch_size - len(batch_rollouts)} more. Current in-flight group rollout requests: {len(self.inflight_group_rollouts)}"
            )

        self.logger.debug("Batch done.")
        return batch_rollouts


def setup_scheduler(
    buffer: Buffer,
    env: Environment,
    tokenizer: PreTrainedTokenizerFast,
    config: OrchestratorConfig,
    clients: list[AsyncOpenAI],
) -> Scheduler:
    if config.scheduler.type == "default":
        return DefaultScheduler(buffer, env, tokenizer, config, clients)
    elif config.scheduler.type == "areal":
        return ARealScheduler(buffer, env, tokenizer, config, clients)
    else:
        raise ValueError(f"Invalid scheduler type: {config.scheduler}")
