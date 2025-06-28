import asyncio
import math
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, load_dataset
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from zeroband.training.orchestrator.config import OrchestratorConfig
from zeroband.training.orchestrator.logger import setup_logger
from zeroband.training.orchestrator.utils import (
    compute_advantages,
    compute_rewards,
    generate_completion,
    get_parquet,
    health_check,
    load_checkpoint,
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
    logger.info("Starting orchestrator")

    # Prepare paths to communicate with the trainer
    if config.rollout.clean:
        logger.info(f"Cleaning rollout path ({config.rollout.path})")
        shutil.rmtree(config.rollout.path, ignore_errors=True)

    if config.checkpoints.clean:
        logger.info(f"Cleaning checkpoints path ({config.checkpoints.path})")
        shutil.rmtree(config.checkpoints.path, ignore_errors=True)

    # Setup monitor
    monitor = setup_monitor(config.monitor)

    # Setup client
    client = AsyncOpenAI(base_url=config.client.base_url, api_key=config.client.api_key)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.completion.model)

    # Check health of the client
    await health_check(client)

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
    for step in range(1, total_steps + 1):
        logger.info(f"Starting training step {step}")

        # Get the batch
        problems_per_batch = config.batch_size // config.samples_per_problem
        indices = range(step * problems_per_batch, (step + 1) * problems_per_batch)
        problems = dataset.select(indices).to_list() * config.samples_per_problem
        prompts = [problem["prompt"] for problem in problems]
        batch_messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

        # Optionally, update the checkpoint step if we are ahead
        # TODO(Mika): This seems silly
        if step - 1 - ckpt_step > config.async_level:
            wait_time = 0
            ckpt_step = step - 1 - config.async_level
            logger.info(f"Hit async barrier, waiting for checkpoint step {ckpt_step}")
            while True:
                stable_file = (
                    Path(config.checkpoints.path) / f"step_{ckpt_step}" / "stable"
                )
                if stable_file.exists():
                    logger.info(
                        f"Found checkpoint for step {ckpt_step} at {stable_file}. Reloading model weights."
                    )
                    await load_checkpoint(client, config.checkpoints.path, ckpt_step)
                    break
                if wait_time % 10 == 0 and wait_time > 0:  # Every log_interval seconds
                    logger.info(
                        f"Waiting for checkpoint for step {ckpt_step} at {stable_file} for {wait_time} seconds"
                    )
                time.sleep(1)
                wait_time += 1

        # Get the completions for the batch
        logger.info(
            f"Sending {len(batch_messages)} inference requests for training step {step} (checkpoint step: {ckpt_step})"
        )
        start_time = time.time()
        chat_completions = await asyncio.gather(
            *(
                generate_completion(client, config.completion, messages)
                for messages in batch_messages
            )
        )
        generate_time = time.time() - start_time
        logger.success(
            f"Received {len(chat_completions)} completions in {generate_time:.2f}s"
        )

        # Get the rewards for the completions
        # TODO: How are we getting the rewards? Is this handled in verifiers or does the orchestrator make a request to a dedicated reward server? For now, we use dummy rewards
        completions = [
            chat_completion.choices[0].message.content
            for chat_completion in chat_completions
        ]
        task_types = [problem["task_type"] for problem in problems]
        verification_infos = [problem["verification_info"] for problem in problems]
        rewards = compute_rewards(completions, task_types, verification_infos)

        # Compute advantages
        advantages = compute_advantages(rewards, config.samples_per_problem)

        # Compute batch metrics
        num_input_tokens = sum(
            completion.usage.prompt_tokens for completion in chat_completions
        )
        num_output_tokens = sum(
            completion.usage.completion_tokens for completion in chat_completions
        )
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
        table = get_parquet(
            prompts=prompts,
            completions=completions,
            rewards=rewards,
            advantages=advantages,
            temperature=config.completion.temperature,
            tokenizer=tokenizer,
        )

        # Save outputs to parquet file
        step_path = Path(config.rollout.path)
        step_path.mkdir(parents=True, exist_ok=True)
        save_path = step_path / f"step_{step}.parquet"
        logger.info(f"Saving batch outputs to {save_path}")
        pq.write_table(table, save_path)

    logger.success("Training completed.")


def main():
    asyncio.run(orchestrate(parse_argv(OrchestratorConfig)))


if __name__ == "__main__":
    main()
