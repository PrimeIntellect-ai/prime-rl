import asyncio
import json
import shutil
import time
from multiprocessing.queues import Queue
from pathlib import Path

# Import environment before any other imports
# ruff: noqa: I001,F401
from zeroband.orchestrator import envs

import lovely_tensors as lt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from zeroband.eval.utils import run_benchmark
from zeroband.orchestrator.ckpt import CheckpointManager, Progress
from zeroband.orchestrator.client import (
    check_has_model,
    check_health,
    generate_completion,
    reload_weights,
    reset_weights,
    setup_client,
    tokenize,
)
from zeroband.orchestrator.config import OrchestratorConfig
from zeroband.orchestrator.data import prepare_batch
from zeroband.orchestrator.logger import setup_logger
from zeroband.orchestrator.utils import (
    compute_advantages,
    compute_rewards,
    parse_completions,
    parse_logprobs,
    parse_output_tokens,
    print_benchmark,
    wait_for_weight_checkpoint,
)
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit, to_col_format


@clean_exit
async def orchestrate(config: OrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting orchestrator")
    logger.debug(f"ClientConfig({config.client})")
    logger.debug(f"ModelConfig({config.model})")
    logger.debug(f"DataConfig({config.data})")
    logger.debug(f"SamplingConfig({config.sampling})")
    logger.debug(f"EvaluationConfig({config.eval})")

    # Setup client
    logger.info(f"Initializing OpenAI client ({config.client.base_url})")
    client = setup_client(config.client)

    # Load tokenizer
    logger.info(f"Initializing tokenizer for {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Setup monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    monitor = setup_monitor(config.monitor, None, tokenizer, config)

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(client)
    await check_has_model(client, config.model.name)
    logger.success("Inference pool ready")

    # Get checkpoint manager
    if config.ckpt:
        ckpt_manager = CheckpointManager(config.ckpt)

    # Reset weights to base model if starting from scratch
    progress = Progress()
    ckpt_step = 0
    if config.ckpt and config.ckpt.resume_step:
        logger.info(f"Resuming training from checkpoint step `{config.ckpt.resume_step}`")
        ckpt_manager.load(progress, step=config.ckpt.resume_step)
        ckpt_step = max(progress.step - config.async_level, 0)
        await reload_weights(client, config.weights_path, ckpt_step)
    else:
        logger.info("Training from scratch. Resetting weights to base model")
        await reset_weights(client)

    # Load dataset
    # TODO: Change to verifiers environment
    dataset: Dataset = load_dataset(config.data.name, split=config.data.split)

    # Optionally, filter dataset for samples within difficulty range
    if config.data.difficulty_filtering:
        field = config.data.difficulty_filtering.solve_rate_field
        min_rate = config.data.difficulty_filtering.min_solve_rate
        max_rate = config.data.difficulty_filtering.max_solve_rate
        logger.info(f"Filtering dataset for difficulty in [{min_rate}, {max_rate}] at field {field}")

        dataset = dataset.filter(lambda x: x[field] >= min_rate and x[field] <= max_rate)

    dataset = dataset.shuffle(seed=config.seed)

    # Iterate over dataset in batches
    steps_per_epoch = len(dataset) // (config.batch_size // config.sampling.n)
    logger.info(f"Starting orchestrator loop (max_steps={config.max_steps}, {steps_per_epoch=})")
    last_eval_step = -1
    while True:
        # Save checkpoint (if we are not at the first step)
        save_ckpt_time = 0
        if config.ckpt and config.ckpt.interval and progress.step > 0 and progress.step % config.ckpt.interval == 0:
            logger.debug(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(progress, step=progress.step)
            save_ckpt_time = time.time() - save_ckpt_start_time

        # Break if we have reached the maximum number of steps
        if config.max_steps and progress.step >= config.max_steps:
            break

        # Check if we need to start a new epoch
        epoch_step = progress.step % steps_per_epoch
        if epoch_step == 0:
            progress.epoch += 1
            # Reshuffle dataset at the beginning of each epoch
            dataset = dataset.shuffle(seed=(config.seed or 0) + progress.epoch)

        logger.info(f"Starting orchestrator step {progress.step} ({ckpt_step=}, epoch={progress.epoch}, {epoch_step=})")
        step_start_time = time.time()

        # Get the batch
        problems_per_batch = config.batch_size // config.sampling.n
        start_idx = epoch_step * problems_per_batch
        indices = range(start_idx, start_idx + problems_per_batch)
        problems = dataset.select(indices).to_list() * config.sampling.n
        prompts = [problem["prompt"] for problem in problems]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        # Optionally, wait for the next checkpoint to be available
        wait_for_weight_ckpt_time, reload_weights_time = 0, 0
        if progress.step - ckpt_step > config.async_level:
            logger.debug(
                f"Hit async barrier because step {progress.step} is {progress.step - ckpt_step} (>{config.async_level}) steps ahead of checkpoint step {ckpt_step}."
            )

            # Wait for the checkpoint to be available
            ckpt_step = progress.step - config.async_level
            logger.info(f"Waiting for weight checkpoint {ckpt_step}")
            wait_for_weight_ckpt_start_time = time.time()
            wait_for_weight_checkpoint(config.weights_path, ckpt_step)
            wait_for_weight_ckpt_time = time.time() - wait_for_weight_ckpt_start_time
            logger.debug(f"Waited {wait_for_weight_ckpt_time:.2f}s for weight checkpoint")

            # Reload the weights
            logger.info(f"Reloading weight checkpoint {ckpt_step}")
            reload_weights_start_time = time.time()
            await reload_weights(client, config.weights_path, ckpt_step)
            reload_weights_time = time.time() - reload_weights_start_time
            logger.debug(f"Reloaded weights in {reload_weights_time:.2f}s")

        # Optionally, run online evals at the specified interval
        time_eval = 0
        if (
            config.eval
            and config.eval.interval
            and ckpt_step % config.eval.interval == 0
            and ckpt_step > last_eval_step
            and (ckpt_step == 0 and config.eval.eval_base_model or ckpt_step > 0)
        ):
            last_eval_step = ckpt_step
            logger.info(f"Running evals for checkpoint step {ckpt_step}")
            time_before_evals = time.time()
            for benchmark in config.eval.benchmarks:
                await run_benchmark(
                    client,
                    benchmark,
                    config.model,
                    config.sampling,
                    ckpt_step,
                    monitor=monitor,
                )
            time_eval = time.time() - time_before_evals
            logger.info(f"Evaluated in {time_eval:.2f}s")

        # Get the completions for the batch
        # TODO: Integrate with async (multi-turn) rollout function from verifiers
        logger.info(f"Sending {len(batch_messages)} inference requests")
        generate_completions_start_time = time.time()
        # These calls are intentionally non-concurrent because we found that /tokenize is sometimes returning a httpx.ReadError when calling this endpoint concurrently
        input_tokens = [await tokenize(client, config.model, messages) for messages in batch_messages]
        chat_completions = await asyncio.gather(
            *(
                generate_completion(client, config.model, config.sampling, messages, len(input_tokens))
                for messages, input_tokens in zip(batch_messages, input_tokens)
            )
        )
        generate_completions_time = time.time() - generate_completions_start_time
        logger.debug(f"Received {len(chat_completions)} inference responses in {generate_completions_time:.2f}s")

        # Parse chat completions responses
        completions = parse_completions(chat_completions)
        output_tokens = parse_output_tokens(chat_completions)
        output_logprobs = parse_logprobs(chat_completions)

        # Get the rewards for the completions
        # TODO: Integrate with async scoring function from verifiers
        logger.info("Computing rewards and advantages")
        compute_rewards_start_time = time.time()
        task_types = [problem["task_type"] for problem in problems]
        verification_infos = [json.loads(problem["verification_info"]) for problem in problems]
        rewards = compute_rewards(completions, task_types, verification_infos)
        advantages = compute_advantages(rewards, config.sampling.n)
        compute_rewards_time = time.time() - compute_rewards_start_time
        logger.debug(f"Computed rewards and advantages in {compute_rewards_time:.2f}s")
        logger.debug(f"Computed rewards: {lt.lovely(torch.tensor(rewards))}")
        logger.debug(f"Computed advantages: {lt.lovely(torch.tensor(advantages))}")

        # Compute batch metrics
        num_input_tokens = sum(completion.usage.prompt_tokens for completion in chat_completions)
        num_output_tokens = sum(completion.usage.completion_tokens for completion in chat_completions)
        num_tokens = num_input_tokens + num_output_tokens
        progress.total_tokens += num_tokens
        progress.total_samples += config.batch_size
        progress.total_problems += config.batch_size // config.sampling.n
        throughput = num_tokens / (generate_completions_time + compute_rewards_time)
        avg_seq_length = num_tokens / config.batch_size

        # Log samples to W&B table if enabled
        if monitor.wandb:
            monitor.wandb.log_samples(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                rewards=rewards,
                advantages=advantages,
                step=progress.step,
            )

        # Write serialized batch to disk for trainer workers to consume
        all_data_ranks_batches = prepare_batch(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            advantages=advantages,
            temperature=config.sampling.temperature,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            micro_batch_size=config.micro_batch_size,
            num_train_workers=config.num_train_workers,
            seq_len=config.seq_len,
            collate_mode=config.collate_mode,
        )

        step_path = Path(config.rollout_path) / f"step_{progress.step}"
        step_path.mkdir(parents=True, exist_ok=True)
        for i, batches in enumerate(all_data_ranks_batches):
            batch_path = step_path / f"rank_{i}.pt"
            tmp_path = batch_path.with_suffix(".tmp")
            logger.debug(f"Saving rollouts for step {progress.step} for rank {i} to {batch_path}")
            torch.save(batches, tmp_path)
            tmp_path.rename(batch_path)

        # Log step metrics
        step_time = time.time() - step_start_time
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {np.mean(rewards):.2f} | Advantage: {np.mean(advantages):.2f} | Throughput: {throughput:.1f} tokens/s | Seq. Length: {avg_seq_length:.1f} tokens/sample"
        logger.success(step_message)

        # Log progress metrics to monitor
        progress_metrics = {
            "progress/orchestrator/total_tokens": progress.total_tokens,
            "progress/orchestrator/total_samples": progress.total_samples,
            "progress/orchestrator/epoch": progress.epoch,
            "progress/orchestrator/step": ckpt_step,  # Shared W&B axis
            "step": progress.step,
        }
        monitor.log(progress_metrics)

        # Log perfrmance metrics to monitor
        perf_metrics = {
            "perf/infer/throughput": throughput,
            "perf/infer/seq_len": avg_seq_length,
            "step": progress.step,
        }
        monitor.log(perf_metrics)

        # Log rewards metrics to monitor
        reward_metrics = {
            "reward/reward": np.mean(rewards),
            "reward/reward_std": np.std(rewards),
            "reward/advantage": np.mean(advantages),
            "reward/advantage_std": np.std(advantages),
            "step": progress.step,
        }
        monitor.log(reward_metrics)

        # Log time metrics to monitor
        time_metrics = {
            "time/orchestrator": step_time,
            "time/orchestrator/wait_for_weight_ckpt": wait_for_weight_ckpt_time,
            "time/orchestrator/generate_completions": generate_completions_time,
            "time/orchestrator/compute_rewards": compute_rewards_time,
            "time/orchestrator/reload_weights": reload_weights_time,
            "time/orchestrator/save_ckpt": save_ckpt_time,
            "time/orchestrator/eval": time_eval,
            "step": progress.step,
        }
        monitor.log(time_metrics)

        # Increment progress
        progress.step += 1

    logger.success("Orchestrator finished.")

    # Optionally, print benchmark table
    if config.bench:
        print_benchmark(to_col_format(monitor.history))


def main():
    """Main entry-point for orchestrator. Run using `uv run orchestrator`"""
    import asyncio

    asyncio.run(orchestrate(parse_argv(OrchestratorConfig)))


if __name__ == "__main__":
    main()
