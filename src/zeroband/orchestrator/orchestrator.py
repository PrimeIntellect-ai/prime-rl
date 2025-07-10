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
from transformers import AutoTokenizer

from zeroband.eval.utils import run_benchmark
from zeroband.orchestrator.ckpt import CheckpointManager, Progress
from zeroband.environments.registry import get_environment
from zeroband.orchestrator.client import (
    tokenize,
    check_has_model,
    check_health,
    reload_weights,
    reset_weights,
    setup_client,
)
from zeroband.orchestrator.config import OrchestratorConfig
from zeroband.orchestrator.data import prepare_batch
from zeroband.orchestrator.logger import setup_logger
from zeroband.orchestrator.utils import (
    parse_completion_tokens,
    parse_completion_logprobs,
    compute_advantages,
    wait_for_weight_checkpoint,
    print_benchmark,
)
from zeroband.utils.monitor import setup_monitor
from zeroband.utils.pydantic_config import parse_argv
from zeroband.utils.utils import clean_exit, to_col_format


@clean_exit
async def orchestrate(config: OrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting orchestrator")

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

    # Load environment and extract dataset
    logger.info(f"Loading environment {config.environment.id} with args {config.environment.args}")
    vf_env = get_environment(config.environment.id, config.environment.args)
    dataset = vf_env.get_dataset(seed=config.seed)

    # Load tokenizer -- placeholder until reworking verifiers to use vLLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Iterate over dataset in batches
    max_steps = config.max_steps or int(1e9)
    steps_per_epoch = len(dataset) // (config.batch_size // config.rollouts_per_prompt)
    logger.info(f"Starting orchestrator loop ({max_steps=}, {steps_per_epoch=})")
    ckpt_step = 0
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

        # Get the batch
        problems_per_batch = config.batch_size // config.rollouts_per_prompt
        start_idx = epoch_step * problems_per_batch
        indices = range(start_idx, start_idx + problems_per_batch)
        problems = dataset.select(indices).to_list() * config.rollouts_per_prompt

        # prepare inputs for verifiers generation
        inputs = {
            "prompt": [problem["prompt"] for problem in problems],
            "info": [problem.get("info", {}) for problem in problems],
            "task": [problem["task"] for problem in problems],
            "answer": [problem.get("answer", "") for problem in problems],
        }

        # generate completions + rewards with verifiers
        logger.info(f"Sending {len(problems)} requests to environments")
        generate_completions_start_time = time.time()
        sampling_args = dict(config.sampling)
        sampling_args["logprobs"] = True

        # sanitize for vLLM OpenAI client
        sampling_args["extra_body"] = {"return_tokens_as_token_ids": True}
        if "top_k" in sampling_args:
            sampling_args["extra_body"]["top_k"] = sampling_args.pop("top_k")
        if "min_p" in sampling_args:
            sampling_args["extra_body"]["min_p"] = sampling_args.pop("min_p")
        if "min_tokens" in sampling_args:
            sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")

        outputs = await vf_env.a_generate(
            inputs=inputs, client=client, model=config.model.name, sampling_args=sampling_args
        )

        results = vf_env.process_env_results(
            prompts=outputs["prompt"],
            completions=outputs["completion"],
            states=outputs["state"],
            rewards=outputs["reward"],
            processing_class=tokenizer,
            max_seq_length=config.seq_len or -1,
            mask_truncated_completions=config.mask_truncated_completions,
            mask_env_responses=config.mask_env_responses,
        )
        generate_completions_time = time.time() - generate_completions_start_time

        # This parsing only works for single-turn vLLM responses for now
        # TODO: Generically, parse vLLM output tokens if possible, else fall-back to regular tokenization
        assert all(len(s["responses"]) == 1 for s in outputs["state"])  # Single-turn responses
        chat_completions = [s["responses"][0] for s in outputs["state"]]
        prompt_tokens = [await tokenize(client, config.model, prompt) for prompt in inputs["prompt"]]
        completion_tokens = parse_completion_tokens(chat_completions)
        completion_logprobs = parse_completion_logprobs(chat_completions)
        prompt_mask = [[0] * len(prompt_tokens[i]) for i in range(len(prompt_tokens))]
        completion_mask = [[1] * len(completion_tokens[i]) for i in range(len(completion_tokens))]

        # Truncate
        for i in range(len(prompt_tokens)):
            if len(prompt_tokens[i]) + len(completion_tokens[i]) > config.seq_len:
                truncate_at = config.seq_len - len(prompt_tokens[i])
                completion_tokens[i] = completion_tokens[i][:truncate_at]
                completion_mask[i] = completion_mask[i][:truncate_at]
                completion_logprobs[i] = completion_logprobs[i][:truncate_at]
            if len(prompt_tokens[i]) != len(results["prompt_ids"][i]):
                logger.warning(
                    f"Mismatch in prompt tokens for problem {i}. {len(prompt_tokens[i])} (vLLM) != {len(results['prompt_ids'][i])} (verifiers)"
                )
            if len(completion_tokens[i]) != len(results["completion_ids"][i]):
                logger.warning(
                    f"Mismatch in completion tokens for problem {i}. {len(completion_tokens[i])} (vLLM) != {len(results['completion_ids'][i])} (verifiers)"
                )

        # prompt_tokens = results["prompt_ids"]
        # prompt_mask = results["prompt_mask"]
        # completion_tokens = results["completion_ids"]
        # completion_mask = results["completion_mask"]
        rewards = results["rewards"]

        # TODO: parse individiual reward functions for logging
        advantages = compute_advantages(rewards, config.rollouts_per_prompt)
        logger.debug(f"Computed rewards: {lt.lovely(torch.tensor(rewards))}")
        logger.debug(f"Computed advantages: {lt.lovely(torch.tensor(advantages))}")

        # compute batch metrics
        num_prompt_tokens = sum(len(prompt_tokens[i]) for i in range(len(prompt_tokens)))
        num_completion_tokens = sum(len(completion_tokens[i]) for i in range(len(completion_tokens)))
        num_tokens = num_prompt_tokens + num_completion_tokens

        progress.total_tokens += num_tokens
        progress.total_samples += config.batch_size
        progress.total_problems += config.batch_size // config.rollouts_per_prompt
        throughput = num_tokens / (generate_completions_time)
        avg_seq_length = num_tokens / config.batch_size

        # Log samples to W&B table if enabled
        if monitor.wandb:
            monitor.wandb.log_samples(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                rewards=rewards,
                advantages=advantages,
                step=progress.step,
            )

        # Write serialized batch to disk for trainer workers to consume
        all_data_ranks_batches = prepare_batch(
            prompt_tokens=prompt_tokens,
            prompt_mask=prompt_mask,
            completion_tokens=completion_tokens,
            completion_mask=completion_mask,
            completion_logprobs=completion_logprobs,
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
