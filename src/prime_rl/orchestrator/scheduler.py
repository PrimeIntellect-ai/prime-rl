import asyncio
import time
from itertools import cycle

import torch
from openai import AsyncOpenAI
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from verifiers import Environment
from verifiers.types import GenerateOutputs, ProcessedOutputs

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.buffer import Buffer, Rollout, make_rollouts
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.utils import parse_is_truncated_completions
from prime_rl.utils.logger import get_logger
from prime_rl.utils.vf import generate_batch, generate_group


def process_generate_outputs(
    generate_outputs: GenerateOutputs,
    buffer: Buffer,
    env: Environment,
    tokenizer: PreTrainedTokenizerFast,
    config: OrchestratorConfig,
    accepted_rollouts: list[Rollout],
):
    logger = get_logger()
    logger.debug("Processing environment results")
    process_env_results_start_time = time.time()
    processed_outputs: ProcessedOutputs = env.process_env_results_vllm(
        prompts=generate_outputs.prompt,
        completions=generate_outputs.completion,
        states=generate_outputs.state,
        rewards=generate_outputs.reward,
        processing_class=tokenizer,
        max_seq_len=config.seq_len,
        mask_env_responses=config.mask_env_responses,
        zero_truncated_completions=config.zero_truncated_completions,
        mask_truncated_completions=config.mask_truncated_completions,
    )
    process_env_results_time = time.time() - process_env_results_start_time
    logger.debug(f"Processed environment results in {process_env_results_time:.2f}s")

    # Extract individual reward function metrics from generate_outputs
    individual_reward_outputs = {}
    for func_name, func_rewards in generate_outputs.metrics.items():
        individual_reward_outputs[func_name] = torch.tensor(func_rewards)
        logger.debug(f"Collected {len(func_rewards)} rewards for {func_name}")

    # Compute advantages
    advantages = compute_advantages(
        rewards=processed_outputs.rewards,
        completion_lengths=list(map(len, processed_outputs.completion_ids)),
        samples_per_problem=config.rollouts_per_example,
        advantage_config=config.advantage,
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
    buffer.update(rollouts)
    sampled_rollouts = buffer.sample_rollouts(n=1)
    accepted_rollouts.extend(sampled_rollouts)


async def areal_loop(
    clients: list[AsyncOpenAI],
    buffer: Buffer,
    env: Environment,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    rollouts_per_example: int,
    sampling_args: dict,
    config: OrchestratorConfig,
    **kwargs,
):
    accepted_rollouts: list[Rollout] = []
    inflight_tasks: list[asyncio.Task] = []
    cycle_clients = cycle(clients)
    logger = get_logger()

    # re-fill the inflight tasks to batch size
    problems_per_batch = batch_size // rollouts_per_example
    logger.debug(
        f"Re-filling inflight examples to batch size {problems_per_batch} ({problems_per_batch - len(inflight_tasks)} new prolems)"
    )
    while len(inflight_tasks) < problems_per_batch:
        task = asyncio.create_task(
            generate_group(
                client=next(cycle_clients),
                env=env,
                model_name=config.model.name,
                problem=buffer.sample_problems(n=1)[0],
                rollouts_per_example=config.rollouts_per_example,
                sampling_args=sampling_args,
                semaphore=None,
            )
        )
        await asyncio.sleep(0)
        inflight_tasks.append(task)

    logger.debug("Done! Waiting for requests to complete...")

    while len(accepted_rollouts) < config.batch_size:
        logger.debug(
            f"Waiting for {config.batch_size - len(accepted_rollouts)} rollouts to complete (inflight tasks: {len(inflight_tasks)})"
        )
        done, _ = await asyncio.wait(inflight_tasks, return_when=asyncio.FIRST_COMPLETED)
        logger.debug(f"Completed {len(done)} group rollout requests")

        for task in done:
            if len(accepted_rollouts) == config.batch_size:
                break

            inflight_tasks.remove(task)
            generate_outputs: GenerateOutputs = task.result()

            process_generate_outputs(
                generate_outputs=generate_outputs,
                buffer=buffer,
                env=env,
                tokenizer=tokenizer,
                config=config,
                accepted_rollouts=accepted_rollouts,
            )
            logger.debug(f"Processed result, now have {len(accepted_rollouts)} rollouts")

            # Schedule new tasks
            logger.debug("Scheduling new task")
            new_task = asyncio.create_task(
                generate_group(
                    client=next(cycle_clients),
                    env=env,
                    model_name=config.model.name,
                    problem=buffer.sample_problems(n=1)[0],
                    rollouts_per_example=config.rollouts_per_example,
                    sampling_args=sampling_args,
                    semaphore=None,
                )
            )
            await asyncio.sleep(0)
            inflight_tasks.append(new_task)

    return accepted_rollouts


async def default_loop(
    clients: list[AsyncOpenAI],
    buffer: Buffer,
    env: Environment,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    rollouts_per_example: int,
    sampling_args: dict,
    config: OrchestratorConfig,
    **kwargs,
):
    logger = get_logger()
    logger.debug("Starting default scheduling loop")
    accepted_rollouts: list[Rollout] = []
    problems_per_batch = batch_size // rollouts_per_example
    problems_to_sample = batch_size // rollouts_per_example
    while True:
        problems = buffer.sample_problems(problems_to_sample)
        semaphore = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent is not None else None
        logger.debug(f"Generating rollouts and completions for {len(problems)} examples")
        generate_start_time = time.time()
        generate_outputs = await generate_batch(
            clients=clients,
            env=env,
            model_name=config.model.name,
            problems=problems,
            rollouts_per_example=config.rollouts_per_example,
            sampling_args=sampling_args,
            semaphore=semaphore,
        )
        generate_time = time.time() - generate_start_time
        logger.debug(f"Generated rollouts and completions in {generate_time:.2f}s")

        logger.debug("Processing environment results")
        process_env_results_start_time = time.time()
        processed_outputs: ProcessedOutputs = env.process_env_results_vllm(
            prompts=generate_outputs.prompt,
            completions=generate_outputs.completion,
            states=generate_outputs.state,
            rewards=generate_outputs.reward,
            processing_class=tokenizer,
            max_seq_len=config.seq_len,
            mask_env_responses=config.mask_env_responses,
            zero_truncated_completions=config.zero_truncated_completions,
            mask_truncated_completions=config.mask_truncated_completions,
        )
        process_env_results_time = time.time() - process_env_results_start_time
        logger.debug(f"Processed environment results in {process_env_results_time:.2f}s")

        # Extract individual reward function metrics from generate_outputs
        individual_reward_outputs = {}
        for func_name, func_rewards in generate_outputs.metrics.items():
            individual_reward_outputs[func_name] = torch.tensor(func_rewards)
            logger.debug(f"Collected {len(func_rewards)} rewards for {func_name}")

        # Compute advantages
        advantages = compute_advantages(
            rewards=processed_outputs.rewards,
            completion_lengths=list(map(len, processed_outputs.completion_ids)),
            samples_per_problem=config.rollouts_per_example,
            advantage_config=config.advantage,
        )

        # Parse whether the completions were truncated
        responses = [state["responses"] for state in generate_outputs.state]
        is_truncated = parse_is_truncated_completions(responses=responses)

        # Make rollouts
        rollouts = make_rollouts(
            problem_ids=[problem["id"] for problem in problems for _ in range(config.rollouts_per_example)],
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
        buffer.update(rollouts)
        sampled_rollouts = buffer.sample_rollouts(problems_to_sample)
        accepted_rollouts.extend(sampled_rollouts)

        # Break if we have enough rollouts to fill the batch
        if len(accepted_rollouts) >= config.batch_size:
            accepted_rollouts = accepted_rollouts[: config.batch_size]
            break

        # On next iteration, sample the remaining problems to fill the batch
        problems_sampled = len(accepted_rollouts) // config.rollouts_per_example
        problems_to_sample = problems_per_batch - problems_sampled

    return accepted_rollouts
