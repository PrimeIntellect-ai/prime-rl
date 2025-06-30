import asyncio
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from zeroband.eval.utils import run_benchmark
from zeroband.training.orchestrator.config import OrchestratorConfig
from zeroband.training.orchestrator.data import prepare_batch
from zeroband.training.orchestrator.logger import setup_logger
from zeroband.training.orchestrator.utils import (
    compute_advantages,
    compute_rewards,
    generate_completion,
    health_check,
    load_checkpoint,
    setup_client,
    wait_for_checkpoint,
)
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit

# todo: add sample to wandb
# todo: add reward, seqlen, task specific reward to wandb


@clean_exit
async def orchestrate(config: OrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting training orchestrator")

    # Prepare paths to communicate with the trainer
    if config.rollout.clean:
        logger.info(f"Cleaning rollout path ({config.rollout.path})")
        shutil.rmtree(config.rollout.path, ignore_errors=True)

    if config.checkpoints.clean:
        logger.info(f"Cleaning checkpoints path ({config.checkpoints.path})")
        shutil.rmtree(config.checkpoints.path, ignore_errors=True)

    # Setup monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    monitor = setup_monitor(config.monitor)

    # Setup client
    logger.info(f"Initializing OpenAI client ({config.client.base_url})")
    client = setup_client(config.client)

    # Load tokenizer
    logger.info(f"Initializing tokenizer for {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Check health of the client
    await health_check(client)

    # Optionally, run evals on base model
    if config.eval:
        logger.info(f"Running evals on base model {config.model.name}")
        for benchmark in config.eval.benchmarks:
            await run_benchmark(client, benchmark, config.model, config.sampling, step=0, use_tqdm=True)

    # Load dataset (TODO: Change to verifiers)
    dataset: Dataset = load_dataset(config.data.name, split=config.data.split)
    dataset = dataset.shuffle(seed=config.seed)

    # Iterate over dataset in batches
    num_batches = len(dataset) // config.batch_size
    total_steps = min(config.max_steps or math.inf, num_batches)
    logger.info(
        f"Starting training loop with {total_steps} steps ({total_steps} batches of {config.batch_size} samples each)"
    )
    ckpt_step = 0
    last_eval_step = -1
    for step in range(1, total_steps + 1):
        logger.info(f"Starting training step {step}")

        # Get the batch
        problems_per_batch = config.batch_size // config.sampling.n
        indices = range(step * problems_per_batch, (step + 1) * problems_per_batch)
        problems = dataset.select(indices).to_list() * config.sampling.n
        prompts = [problem["prompt"] for problem in problems]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        # Optionally, wait for the next checkpoint to be available
        async_level = step - 1 - ckpt_step  # How many steps training ahead
        if async_level > config.async_level:
            ckpt_step = step - 1 - config.async_level
            logger.info(f"Hit async barrier {async_level} > {config.async_level}")
            wait_for_checkpoint(config.checkpoints.path, ckpt_step)
            await load_checkpoint(client, config.checkpoints.path, ckpt_step)

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
        logger.info(
            f"Sending {len(batch_messages)} inference requests for training step {step} (checkpoint step: {ckpt_step})"
        )
        start_time = time.time()
        chat_completions = await asyncio.gather(
            *(generate_completion(client, config.model, config.sampling, messages) for messages in batch_messages)
        )
        generate_time = time.time() - start_time
        logger.success(f"Received {len(chat_completions)} completions in {generate_time:.2f}s")

        # Get the rewards for the completions
        # TODO: How are we getting the rewards? Is this handled in verifiers or does the orchestrator make a request to a dedicated reward server? For now, we use dummy rewards
        completions = [chat_completion.choices[0].message.content for chat_completion in chat_completions]
        task_types = [problem["task_type"] for problem in problems]
        verification_infos = [json.loads(problem["verification_info"]) for problem in problems]
        rewards = compute_rewards(completions, task_types, verification_infos)
        advantages = compute_advantages(rewards, config.sampling.n)
        logger.info(f"Computed rewards (average reward: {np.mean(rewards):.2f}")

        # Compute batch metrics
        num_input_tokens = sum(completion.usage.prompt_tokens for completion in chat_completions)
        num_output_tokens = sum(completion.usage.completion_tokens for completion in chat_completions)
        num_tokens = num_input_tokens + num_output_tokens
        throughput = num_tokens / generate_time
        avg_seq_length = num_tokens / config.batch_size

        # Update total metrics
        # TODO(Mika): Use some form of metrics averaging class to make this less verbose
        # total_num_input_tokens += num_input_tokens
        # total_num_output_tokens += num_output_tokens
        # total_num_tokens += num_tokens
        # total_generate_time += generate_time
        # avg_throughput = total_num_tokens / total_generate_time
        # avg_seq_length = total_num_tokens / step

        # Log metrics to stdout
        logger.info(
            f"Throughput: {throughput:.1f} tokens/s ({num_tokens} tokens in {generate_time:.2f}s, {avg_seq_length:.1f} tokens/sample)"
        )

        # Log metrics to monitor
        progress_metrics = {
            "perf/throughput": throughput,
            "perf/avg_seq_length": avg_seq_length,
            "perf/num_input_tokens": num_input_tokens,
            "perf/num_output_tokens": num_output_tokens,
            "perf/num_tokens": num_tokens,
            "step": step,
        }
        monitor.log(progress_metrics)

        # Write step parquet file
        all_data_ranks_batches = prepare_batch(
            prompts=prompts,
            completions=completions,
            advantages=advantages,
            temperature=config.sampling.temperature,
            tokenizer=tokenizer,
            train_config=config.train,
        )

        for i, batches in enumerate(all_data_ranks_batches):
            save_folder = Path(config.rollout.path) / f"step_{step}"
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / f"rollout_{i}.pt"
            tmp_path = save_path.with_suffix(".tmp")
            torch.save(batches, tmp_path)
            tmp_path.rename(save_path)
            logger.info(f"Saved rollouts for step {step} to {save_path}")

    logger.success("Training completed.")


def main():
    asyncio.run(orchestrate(parse_argv(OrchestratorConfig)))


if __name__ == "__main__":
    main()
