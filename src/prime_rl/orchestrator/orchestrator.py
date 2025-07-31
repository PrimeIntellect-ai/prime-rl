import asyncio
import time
from loguru import logger
from pathlib import Path

# Import environment before any other imports
# ruff: noqa: I001,F401
from prime_rl.orchestrator import envs

import lovely_tensors as lt
import numpy as np
import torch
from verifiers import load_environment
from transformers import AutoTokenizer

from prime_rl.eval.utils import run_benchmark
from prime_rl.orchestrator.ckpt import CheckpointManager, Progress
from prime_rl.orchestrator.client import (
    check_has_model,
    check_health,
    reload_weights,
    reset_weights,
    setup_client,
)
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.buffer import setup_buffer, make_rollouts, Rollout
from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.orchestrator.logger import setup_logger
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.utils import (
    wait_for_weight_checkpoint,
    print_benchmark,
)
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, to_col_format, format_num


@clean_exit
@logger.catch(reraise=True)
async def orchestrate(config: OrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(config.log)
    logger.info("Starting orchestrator")

    # Print warning if running in benchmark mode
    if config.bench:
        logger.warning(
            f"Running in benchmark mode (max_steps={config.max_steps}, async_level={format_num(config.async_level, precision=0)})"
        )

    # Setup client
    logger.info(f"Initializing OpenAI client ({config.client.host}:{config.client.port})")
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
    ckpt_manager = None
    if config.ckpt:
        logger.info(f"Initializing checkpoint manager ({config.ckpt})")
        ckpt_manager = CheckpointManager(config.ckpt)

    # Reset weights to base model if starting from scratch
    progress = Progress()
    ckpt_step = 0
    if config.ckpt and ckpt_manager and config.ckpt.resume_step:
        logger.info(f"Resuming training from checkpoint step `{config.ckpt.resume_step}`")
        ckpt_manager.load(progress, step=config.ckpt.resume_step)
        ckpt_step = max(progress.step - config.async_level, 0)
        await reload_weights(client, config.weights_path, ckpt_step)
    else:
        logger.info("Training from scratch. Resetting weights to base model")
        await reset_weights(client)

    # Load environment and extract dataset
    logger.info(f"Loading environment {config.environment.id} with args {config.environment.args}")
    vf_env = load_environment(config.environment.id, **config.environment.args)
    dataset = vf_env.get_dataset(seed=config.seed)

    # Setup buffer
    logger.info(f"Setting up buffer ({config.buffer})")
    buffer = setup_buffer(dataset, config.buffer)

    # Load tokenizer -- placeholder until reworking verifiers to use vLLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Iterate over dataset in batches
    max_steps = config.max_steps or int(1e9)
    logger.info(f"Starting orchestrator loop ({max_steps=}")
    ckpt_step = 0
    last_eval_step = -1
    is_first_step = True
    while True:
        # Save checkpoint (if we are not at the first step)
        save_ckpt_time = 0
        if (
            config.ckpt
            and ckpt_manager
            and config.ckpt.interval
            and not is_first_step
            and progress.step % config.ckpt.interval == 0
        ):
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(progress, step=progress.step)
            save_ckpt_time = time.time() - save_ckpt_start_time

        # Break if we have reached the maximum number of steps
        if config.max_steps and progress.step >= config.max_steps:
            break

        logger.info(f"Starting orchestrator step {progress.step} ({ckpt_step=})")
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
        eval_time = 0
        if (
            config.eval
            and config.eval.interval
            and ckpt_step % config.eval.interval == 0
            and ckpt_step > last_eval_step
            and ((ckpt_step == 0 and config.eval.eval_base_model) or ckpt_step > 0)
        ):
            last_eval_step = ckpt_step
            logger.info(f"Running evals for checkpoint step {ckpt_step}")
            eval_start_time = time.time()
            await asyncio.gather(
                *[
                    run_benchmark(
                        client,
                        benchmark,
                        config.model,
                        config.sampling,
                        rollouts_per_prompt=rollouts_per_prompt,
                        ckpt_step=ckpt_step,
                        monitor=monitor,
                        step=progress.step,
                    )
                    for benchmark, rollouts_per_prompt in zip(config.eval.benchmarks, config.eval.rollouts_per_prompt)
                ]
            )
            eval_time = time.time() - eval_start_time
            logger.info(f"Evaluated in {eval_time:.2f}s")

        accepted_rollouts: list[Rollout] = []
        problem_requests, completion_requests, calls_to_generate = 0, 0, 0
        problems_per_batch = config.batch_size // config.rollouts_per_prompt
        problems_to_sample = problems_per_batch
        while True:
            # Get the batch
            problem_ids, problems = buffer.sample_problems(problems_to_sample)

            # Duplicate problems `rollouts_per_prompt` times
            problem_ids = [problem_id for problem_id in problem_ids for _ in range(config.rollouts_per_prompt)]
            problems = [problem for problem in problems for _ in range(config.rollouts_per_prompt)]

            # Prepare inputs for verifiers generation
            # TODO: Can we use `prime_rl.utils.utils.to_col_format` here?
            inputs = {
                "prompt": [problem["prompt"] for problem in problems],
                "info": [problem.get("info", {}) for problem in problems],
                "task": [problem["task"] for problem in problems],
                "answer": [problem.get("answer", "") for problem in problems],
            }

            # Generate completions + rewards with verifiers
            logger.info(f"Sending {len(problems)} requests to environments")
            generate_completions_start_time = time.time()
            sampling_args = dict(config.sampling)
            sampling_args["logprobs"] = True

            # Sanitize for vLLM OpenAI client
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
            generate_completions_time = time.time() - generate_completions_start_time
            problem_requests += problems_to_sample
            completion_requests += problems_to_sample * config.rollouts_per_prompt
            calls_to_generate += 1

            results = vf_env.process_env_results_vllm(
                prompts=outputs.prompt,
                completions=outputs.completion,
                states=outputs.state,
                rewards=outputs.reward,
                processing_class=tokenizer,
                max_seq_len=config.seq_len,
                mask_env_responses=config.mask_env_responses,
                zero_truncated_completions=config.zero_truncated_completions,
                mask_truncated_completions=config.mask_truncated_completions,
            )

            advantages = compute_advantages(
                rewards=outputs.reward,
                samples_per_problem=config.rollouts_per_prompt,
                advantage_type=config.advantage_type,
            )

            # Update pool
            rollouts = make_rollouts(
                problem_ids=problem_ids,
                prompt_tokens=results.prompt_ids,
                prompt_masks=results.prompt_mask,
                completion_tokens=results.completion_ids,
                completion_masks=results.completion_mask,
                completion_logprobs=results.completion_logprobs,
                rewards=outputs.reward,
                advantages=advantages,
            )
            buffer.update(rollouts)
            accepted_rollouts.extend(buffer.sample_rollouts(problems_to_sample))

            # Break if we have enough rollouts to fill the batch
            if len(accepted_rollouts) >= config.batch_size:
                accepted_rollouts = accepted_rollouts[: config.batch_size]
                break

            # On next iteration, sample the remaining problems to fill the batch
            problems_sampled = len(accepted_rollouts) // config.rollouts_per_prompt
            problems_to_sample = problems_per_batch - problems_sampled

        # Unpack accepted rollouts
        rewards = [rollout.reward for rollout in accepted_rollouts]
        advantages = [rollout.advantage for rollout in accepted_rollouts]
        problem_ids = [rollout.problem_id for rollout in accepted_rollouts]
        prompt_tokens = [rollout.prompt_tokens for rollout in accepted_rollouts]
        completion_tokens = [rollout.completion_tokens for rollout in accepted_rollouts]

        logger.debug(f"Computed rewards: {lt.lovely(torch.tensor(rewards))}")
        logger.debug(f"Computed advantages ({config.advantage_type}): {lt.lovely(torch.tensor(advantages))}")

        # compute batch metrics
        num_prompt_tokens = sum(len(prompt_tokens[i]) for i in range(len(prompt_tokens)))
        num_completion_tokens = sum(len(completion_tokens[i]) for i in range(len(completion_tokens)))
        num_tokens = num_prompt_tokens + num_completion_tokens

        progress.total_tokens += num_tokens
        progress.total_samples += config.batch_size
        progress.total_problems += config.batch_size // config.rollouts_per_prompt
        throughput = num_tokens / (generate_completions_time)

        seq_lens = np.array([len(p) + len(c) for p, c in zip(prompt_tokens, completion_tokens)])
        problem_seqlens = seq_lens.reshape(-1, config.rollouts_per_prompt).mean(axis=-1)

        # Compute solve all/ none and critical batch size
        grouped_rewards = [
            rewards[i : i + config.rollouts_per_prompt] for i in range(0, len(rewards), config.rollouts_per_prompt)
        ]
        assert len(grouped_rewards) == problems_per_batch
        solve_all = sum(1 for group in grouped_rewards if all(reward == 1 for reward in group)) / problems_per_batch
        solve_none = sum(1 for group in grouped_rewards if all(reward == 0 for reward in group)) / problems_per_batch
        effective_batch_size = 1 - solve_all - solve_none

        # Log samples to W&B table if enabled
        if monitor.wandb:
            monitor.wandb.log_samples(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                rewards=rewards,
                advantages=advantages,
                rollouts_per_problem=config.rollouts_per_prompt,
                step=progress.step,
            )
            monitor.wandb.log_distributions(
                rewards,
                advantages,
                rollouts_per_problem=config.rollouts_per_prompt,
                step=progress.step,
            )

        # Write serialized batch to disk for trainer workers to consume
        all_data_ranks_batches = prepare_batch(
            rollouts=rollouts,
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
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {np.mean(rewards):.2f} | Throughput: {throughput:.1f} tokens/s | Seq. Length: {float(problem_seqlens.mean()):.1f} tokens/sample"
        logger.success(step_message)

        # Log progress metrics to monitor
        progress_metrics = {
            "progress/total_tokens": progress.total_tokens,
            "progress/total_samples": progress.total_samples,
            "progress/total_problems": progress.total_problems,
            "progress/ckpt_step": ckpt_step,  # Shared W&B axis
            "step": progress.step,
        }
        monitor.log(progress_metrics)

        # Log sequence length metrics to monitor
        seq_metrics = {
            "seq/len": float(problem_seqlens.mean()),
            "seq/len/max": float(problem_seqlens.max()),
            "seq/len/min": float(problem_seqlens.min()),
            "seq/len/std": float(problem_seqlens.std()),
            "step": progress.step,
        }
        monitor.log(seq_metrics)

        # Log performance metrics to monitor
        perf_metrics = {
            "perf/infer/throughput": throughput,
            "perf/problem_requests": problem_requests,
            "perf/completion_requests": completion_requests,
            "perf/calls_to_generate": calls_to_generate,
            "step": progress.step,
        }
        monitor.log(perf_metrics)

        # Log rewards metrics to monitor
        reward_metrics = {
            "reward/reward": np.mean(rewards),
            "reward/solve_none": solve_none,
            "reward/solve_all": solve_all,
            "reward/effective_batch_size": effective_batch_size,
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
            "time/orchestrator/eval": eval_time,
            "step": progress.step,
        }
        monitor.log(time_metrics)

        # Increment progress
        progress.step += 1

    # Log final (immutable) samples and distributions to W&B table
    if monitor.wandb:
        logger.info("Logging final samples and distributions as W&B table")
        monitor.wandb.log_final_samples()
        monitor.wandb.log_final_distributions()

    # Write final checkpoint
    if config.ckpt and ckpt_manager:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(progress, step=progress.step)

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
