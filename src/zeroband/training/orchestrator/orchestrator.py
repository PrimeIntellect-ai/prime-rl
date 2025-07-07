import asyncio
import json
import shutil
import time
from multiprocessing.queues import Queue
from pathlib import Path

# Import environment before any other imports
# ruff: noqa: I001,F401
from zeroband.training.orchestrator import envs

import lovely_tensors as lt
import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from zeroband.eval.utils import run_benchmark
from zeroband.training.orchestrator.ckpt import get_ckpt_manager, Progress
from zeroband.training.orchestrator.client import (
    check_has_model,
    check_health,
    generate_completion,
    reload_weights,
    reset_weights,
    setup_client,
    tokenize,
)
from zeroband.training.orchestrator.config import OrchestratorConfig
from zeroband.training.orchestrator.data import prepare_batch
from zeroband.training.orchestrator.logger import setup_logger
from zeroband.training.orchestrator.utils import (
    compute_advantages,
    compute_rewards,
    parse_completions,
    parse_logprobs,
    parse_output_tokens,
    wait_for_weight_checkpoint,
)
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit


@clean_exit
async def orchestrate(config: OrchestratorConfig, setup_queue: Queue | None = None):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting orchestrator")
    logger.debug(f"ClientConfig({config.client})")
    logger.debug(f"ModelConfig({config.model})")
    logger.debug(f"DataConfig({config.data})")
    logger.debug(f"SamplingConfig({config.sampling})")
    logger.debug(f"EvaluationConfig({config.eval})")

    # Prepare paths to communicate with the trainer
    if config.clean:
        logger.info("Cleaning checkpoint, logs, checkpoint weights and rollout directories")
        logger.debug(f"Cleaning checkpoint path ({config.ckpt.path})")
        shutil.rmtree(config.ckpt.path, ignore_errors=True)

        logger.debug(f"Cleaning logs path ({config.log.path})")
        shutil.rmtree(config.log.path, ignore_errors=True)

        logger.debug(f"Cleaning checkpoint weights path ({config.weights_path})")
        shutil.rmtree(config.weights_path, ignore_errors=True)

        logger.debug(f"Cleaning rollout path ({config.rollout_path})")
        shutil.rmtree(config.rollout_path, ignore_errors=True)

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
    logger.debug("Waiting for inference pool to be ready")
    await check_health(client)
    await check_has_model(client, config.model.name)
    logger.success("Inference pool ready")

    # Get checkpoint manager
    ckpt_manager = get_ckpt_manager(config.ckpt.path)

    # Reset weights to base model if starting from scratch
    progress = Progress()
    ckpt_step = 0
    if config.ckpt.resume_step:
        logger.info(f"Resuming training from checkpoint step `{config.ckpt.resume_step}`")
        ckpt_manager.load(progress, step=config.ckpt.resume_step)
        ckpt_step = max(progress.step - config.async_level, 0)
        await reload_weights(client, config.weights_path, ckpt_step)
    else:
        logger.info("Training from scratch. Resetting weights to base model")
        await reset_weights(client)

    # Signal that setup is complete
    if setup_queue is not None:
        logger.info("Signaling trainer that orchestrator setup is complete")
        setup_queue.put("ready")

    # Optionally, run evals on base model (only if starting from scratch)
    if config.eval and not config.ckpt.resume_step:
        logger.info("Running evals on base model")
        for benchmark in config.eval.benchmarks:
            await run_benchmark(client, benchmark, config.model, config.sampling, step=0, use_tqdm=True)

    # Load dataset
    # TODO: Change to verifiers environment
    dataset: Dataset = load_dataset(config.data.name, split=config.data.split)
    dataset = dataset.shuffle(seed=config.seed)

    # Iterate over dataset in batches
    steps_per_epoch = len(dataset) // (config.batch_size // config.sampling.n)
    logger.info(f"Starting training loop ({steps_per_epoch=})")
    last_eval_step = -1
    while True:
        if config.max_steps and progress.step >= config.max_steps:
            break

        # Check if we need to start a new epoch
        epoch_step = progress.step % steps_per_epoch
        if epoch_step == 0:
            progress.epoch += 1
            logger.info(f"Starting epoch {progress.epoch}")
            # Reshuffle dataset at the beginning of each epoch
            dataset = dataset.shuffle(seed=(config.seed or 0) + progress.epoch)

        logger.debug(f"Orchestrator step {progress.step} ({ckpt_step=})")
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
            logger.debug(f"Waiting for weight checkpoint for step {ckpt_step}")
            wait_for_weight_ckpt_start_time = time.time()
            wait_for_weight_checkpoint(config.weights_path, ckpt_step)
            wait_for_weight_ckpt_time = time.time() - wait_for_weight_ckpt_start_time
            logger.debug(f"Waited {wait_for_weight_ckpt_time:.2f}s for weight checkpoint")

            # Reload the weights
            logger.debug(f"Reloading weights for step {ckpt_step}")
            reload_weights_start_time = time.time()
            await reload_weights(client, config.weights_path, ckpt_step)
            reload_weights_time = time.time() - reload_weights_start_time
            logger.debug(f"Reloaded weights in {reload_weights_time:.2f}s")

        # Optionally, run online evals at the specified interval
        if (
            config.eval
            and config.eval.online
            and ckpt_step % config.eval.online.interval == 0
            and ckpt_step > last_eval_step
        ):
            last_eval_step = ckpt_step
            logger.info(f"Running evals for checkpoint step {ckpt_step}")
            for benchmark in config.eval.benchmarks:
                await run_benchmark(
                    client,
                    benchmark,
                    config.model,
                    config.sampling,
                    ckpt_step,
                )

        # Get the completions for the batch
        # TODO: Integrate with async (multi-turn) rollout function from verifiers
        logger.debug(f"Sending {len(batch_messages)} inference requests for step {progress.step}")
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

        # Parse chat completions responses
        completions = parse_completions(chat_completions)
        output_tokens = parse_output_tokens(chat_completions)
        output_logprobs = parse_logprobs(chat_completions)

        # Get the rewards for the completions
        # TODO: Integrate with async scoring function from verifiers
        logger.debug(f"Computing rewards and advantages for step {progress.step}")
        compute_rewards_start_time = time.time()
        task_types = [problem["task_type"] for problem in problems]
        verification_infos = [json.loads(problem["verification_info"]) for problem in problems]
        rewards = compute_rewards(completions, task_types, verification_infos)
        advantages = compute_advantages(rewards, config.sampling.n)
        logger.debug(f"Computed rewards: {lt.lovely(torch.tensor(rewards))}")
        logger.debug(f"Computed advantages: {lt.lovely(torch.tensor(advantages))}")

        compute_rewards_time = time.time() - compute_rewards_start_time

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

        save_ckpt_time = 0
        if config.ckpt and config.ckpt.interval and (progress.step + 1) % config.ckpt.interval == 0:
            logger.debug(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(progress, step=progress.step + 1)
            save_ckpt_time = time.time() - save_ckpt_start_time

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
        step_message = f"Orchestrator | step {progress.step} | Time:{step_time:.2f}s | Avg. Reward: {np.mean(rewards):.2f} | Avg. Advantage: {np.mean(advantages):.2f} | Throughput: {throughput:.1f} tokens/s | Avg. Seq. Length: {avg_seq_length:.1f} tokens/sample"
        logger.info(step_message)

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
            "step": progress.step,
        }
        monitor.log(time_metrics)

        # Increment progress
        progress.step += 1

    logger.success("Orchestrator finished.")


def run_orchestrator(config: OrchestratorConfig, setup_queue: Queue | None = None):
    """Utility function to run the orchestrator as a sidecar process in a synchronous context."""
    import asyncio

    asyncio.run(orchestrate(config, setup_queue))


def main():
    """Main entry-point for orchestrator. Run using `uv run orchestrator`"""
    run_orchestrator(parse_argv(OrchestratorConfig))


if __name__ == "__main__":
    main()
