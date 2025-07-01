import asyncio
import json
import math
import shutil
import time
from multiprocessing.queues import Queue
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from zeroband.eval.utils import run_benchmark
from zeroband.training.orchestrator.client import (
    check_has_model,
    check_health,
    generate_completion,
    reload_weights,
    reset_weights,
    setup_client,
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

# TODO: Log samples to wandb
# TODO: Add reward, seqlen, task specific reward to wandb


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
    if config.rollout.clean:
        logger.debug(f"Cleaning rollout path ({config.rollout.path})")
        shutil.rmtree(config.rollout.path, ignore_errors=True)

    if config.weights.clean:
        logger.debug(f"Cleaning weights path ({config.weights.path})")
        shutil.rmtree(config.weights.path, ignore_errors=True)

    # Setup monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    monitor = setup_monitor(config.monitor, None, config)

    # Setup client
    logger.info(f"Initializing OpenAI client ({config.client.base_url})")
    client = setup_client(config.client)

    # Load tokenizer
    logger.info(f"Initializing tokenizer for {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(client)
    await check_has_model(client, config.model.name)
    logger.success("Inference pool ready")

    # Reset weights to base model to allow reusing inference server across runs
    logger.info("Resetting weights to base model")
    await reset_weights(client)

    # Signal that setup is complete
    if setup_queue is not None:
        logger.info("Signaling trainer that orchestrator setup is complete")
        setup_queue.put("ready")

    # Optionally, run evals on base model
    if config.eval:
        logger.info("Running evals on base model")
        for benchmark in config.eval.benchmarks:
            await run_benchmark(client, benchmark, config.model, config.sampling, step=0, use_tqdm=True)

    # Load dataset (TODO: Change to verifiers)
    dataset: Dataset = load_dataset(config.data.name, split=config.data.split)
    dataset = dataset.shuffle(seed=config.seed)

    # Iterate over dataset in batches
    max_steps = config.max_steps or math.inf
    steps_per_epoch = len(dataset) // (config.batch_size // config.sampling.n)
    logger.info(f"Starting training loop (max_steps={max_steps}, steps_per_epoch={steps_per_epoch})")
    total_tokens, total_samples = 0, 0
    ckpt_step = 0
    last_eval_step = -1
    epoch = 0

    for step in range(1, int(max_steps) + 1):
        # Check if we need to start a new epoch
        epoch_step = (step - 1) % steps_per_epoch
        if epoch_step == 0:
            epoch += 1
            logger.info(f"Starting epoch {epoch}")
            # Reshuffle dataset at the beginning of each epoch
            dataset = dataset.shuffle(seed=(config.seed or 0) + epoch - 1)

        logger.info(
            f"Orchestrator step {step} (epoch: {epoch}, epoch_step: {epoch_step + 1}/{steps_per_epoch}, checkpoint step: {ckpt_step})"
        )
        step_start_time = time.time()

        # Get the batch
        problems_per_batch = config.batch_size // config.sampling.n
        start_idx = epoch_step * problems_per_batch
        indices = range(start_idx, start_idx + problems_per_batch)
        problems = dataset.select(indices).to_list() * config.sampling.n
        prompts = [problem["prompt"] for problem in problems]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        # Optionally, wait for the next checkpoint to be available
        async_level = step - 1 - ckpt_step  # How many steps training ahead
        if async_level > config.async_level:
            ckpt_step = step - 1 - config.async_level
            logger.warning(
                f"Hit async barrier because step {step} is {async_level} (>{config.async_level}) steps ahead of checkpoint step {ckpt_step}."
            )
            wait_for_weight_checkpoint(config.weights.path, ckpt_step)
            await reload_weights(client, config.weights.path, ckpt_step)

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
                    use_tqdm=config.use_tqdm,
                )

        # Get the completions for the batch
        # TODO: Integrate with async (multi-turn) rollout function from verifiers
        logger.info(f"Sending {len(batch_messages)} inference requests for step {step}")
        generate_completions_start_time = time.time()
        chat_completions = await asyncio.gather(
            *(generate_completion(client, config.model, config.sampling, messages) for messages in batch_messages)
        )
        generate_completions_time = time.time() - generate_completions_start_time

        # Get the rewards for the completions
        # TODO: Integrate with async scoring function from verifiers
        logger.info(f"Computing rewards for step {step}")
        compute_rewards_start_time = time.time()
        completions = parse_completions(chat_completions)
        task_types = [problem["task_type"] for problem in problems]
        verification_infos = [json.loads(problem["verification_info"]) for problem in problems]
        rewards = compute_rewards(completions, task_types, verification_infos)
        advantages = compute_advantages(rewards, config.sampling.n)
        compute_rewards_time = time.time() - compute_rewards_start_time

        # Compute batch metrics
        num_input_tokens = sum(completion.usage.prompt_tokens for completion in chat_completions)
        num_output_tokens = sum(completion.usage.completion_tokens for completion in chat_completions)
        num_tokens = num_input_tokens + num_output_tokens
        total_tokens += num_tokens
        total_samples += config.batch_size
        throughput = num_tokens / (generate_completions_time + compute_rewards_time)
        avg_seq_length = num_tokens / config.batch_size

        # Write serialized batch to disk for trainer workers to consume
        all_data_ranks_batches = prepare_batch(
            prompts=prompts,
            output_tokens=parse_output_tokens(chat_completions),
            output_logprobs=parse_logprobs(chat_completions),
            advantages=advantages,
            temperature=config.sampling.temperature,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            micro_batch_size=config.micro_batch_size,
            num_train_workers=config.num_train_workers,
            seq_len=config.seq_len,
            collate_mode=config.collate_mode,
        )

        step_path = Path(config.rollout.path) / f"step_{step}"
        step_path.mkdir(parents=True, exist_ok=True)
        for i, batches in enumerate(all_data_ranks_batches):
            batch_path = step_path / f"rank_{i}.pt"
            tmp_path = batch_path.with_suffix(".tmp")
            logger.debug(f"Saving rollouts for step {step} for rank {i} to {batch_path}")
            torch.save(batches, tmp_path)
            tmp_path.rename(batch_path)

        # Log step metrics
        step_time = time.time() - step_start_time
        step_message = f"Finished orchestrator step {step} in {step_time:.2f}s (Avg. Reward: {np.mean(rewards):.2f}, Throughput: {throughput:.1f} tokens/s, Avg. Seq. Length: {avg_seq_length:.1f} tokens/sample)"
        logger.success(step_message)

        # Log progress metrics to monitor
        progress_metrics = {
            "progress/infer/total_tokens": total_tokens,
            "progress/infer/total_samples": total_samples,
            "progress/train/step": ckpt_step,  # Shared W&B axis
            "progress/train/epoch": epoch,
            "step": step,
        }
        monitor.log(progress_metrics)

        # Log perfrmance metrics to monitor
        perf_metrics = {
            "perf/infer/throughput": throughput,
            "perf/infer/seq_len": avg_seq_length,
            "step": step,
        }
        monitor.log(perf_metrics)

        # Log rewards metrics to monitor
        reward_metrics = {"reward/mean": np.mean(rewards), "step": step}
        monitor.log(reward_metrics)

        # Log time metrics to monitor
        time_metrics = {
            "time/infer": step_time,
            "time/infer/generate_completions": generate_completions_time,
            "time/infer/compute_rewards": compute_rewards_time,
            "step": step,
        }
        monitor.log(time_metrics)

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
