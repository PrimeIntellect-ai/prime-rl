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
from transformers import AutoTokenizer

from prime_rl.eval.utils import run_benchmark
from prime_rl.orchestrator.ckpt import CheckpointManager, Progress
from prime_rl.environments.registry import load_environment
from prime_rl.orchestrator.client import (
    check_has_model,
    check_health,
    reload_weights,
    reset_weights,
    setup_client,
)
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.orchestrator.data import DataPool, prepare_batch, GeneratedSample
from prime_rl.orchestrator.logger import setup_logger
from prime_rl.orchestrator.utils import (
    process_env_results,
    compute_advantages,
    wait_for_weight_checkpoint,
    print_benchmark,
    flatten_keep,
    create_generated_samples,
    unpack_generated_samples
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
    vf_env = load_environment(config.environment.id, config.environment.args)
    dataset = vf_env.get_dataset(seed=config.seed)
    datapool = DataPool(dataset, config.data_loading)
    
    # Load tokenizer -- placeholder until reworking verifiers to use vLLM tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Iterate over dataset in batches
    max_steps = config.max_steps or int(1e9)
    logger.info(f"Starting orchestrator loop ({max_steps=})")
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

        logger.info(f"Starting orchestrator step {progress.step} ({ckpt_step=})")
        step_start_time = time.time()

        # Optionally, wait for the next checkpoint to be available
        wait_for_weight_ckpt_time, reload_weights_time = 0, 0
        if progress.step - ckpt_step > config.async_level:
            logger.debug(
                f"Hit async barrier because step {progress.step} is {progress.step - ckpt_step} (>{config.async_level}) steps ahead of checkpoint step {ckpt_step}."
            )
            datapool.empty_buffer()

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
            and ((ckpt_step == 0 and config.eval.eval_base_model) or ckpt_step > 0)
        ):
            last_eval_step = ckpt_step
            logger.info(f"Running evals for checkpoint step {ckpt_step}")
            time_before_evals = time.time()
            await asyncio.gather(
                *[
                    run_benchmark(
                        client,
                        benchmark,
                        config.model,
                        config.sampling,
                        ckpt_step=ckpt_step,
                        monitor=monitor,
                        step=progress.step,
                    )
                    for benchmark in config.eval.benchmarks
                ]
            )
            time_eval = time.time() - time_before_evals
            logger.info(f"Evaluated in {time_eval:.2f}s")

        # Get the batch
        problems_per_batch = config.batch_size // config.rollouts_per_prompt

        generate_completions_start_time = time.time()
        
        all_generated_samples = datapool.get_buffered_samples()
        
        if len(all_generated_samples) >= config.batch_size:
            all_generated_samples, samples_to_buffer = all_generated_samples[:config.batch_size], all_generated_samples[config.batch_size:]
            datapool.add_to_buffer(samples_to_buffer)
        
        else:
            for sampling_iteration in range(config.data_loading.online_difficulty_filtering_strategy.max_sample_tries):
                
                if config.data_loading.online_difficulty_filtering_strategy.enabled:
                    num_problems_to_sample = int(problems_per_batch*config.data_loading.online_difficulty_filtering_strategy.oversampling_factor)
                else:
                    num_problems_to_sample = problems_per_batch
                    
                problems = [
                    problem for problem in datapool.sample_batch(num_problems_to_sample).to_list()
                    for _ in range(config.rollouts_per_prompt)
                ]
                
                # prepare inputs for verifiers generation
                inputs = {
                    "prompt": [problem["prompt"] for problem in problems],
                    "info": [problem.get("info", {}) for problem in problems],
                    "task": [problem["task"] for problem in problems],
                    "answer": [problem.get("answer", "") for problem in problems],
                }

                # generate completions + rewards with verifiers
                logger.info(f"Sending {len(problems)} requests to environments")
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

                
                # TODO: Switch parsing prompt+completion tokens/ masks to vf_env.process_env_results once it supports parsing directly from vLLM. For now, this only works for single-turn output results
                results = await process_env_results(outputs, client=client, config=config)
                prompt_tokens = results["prompt_tokens"]
                completion_tokens = results["completion_tokens"]
                completion_logprobs = results["completion_logprobs"]
                prompt_masks = results["prompt_masks"]
                completion_masks = results["completion_masks"]
                rewards = outputs["reward"]  # TODO: Align naming between prime-rl <> verifiers

                # Compute advantages
                per_problem_rewards = [rewards[i : i + config.rollouts_per_prompt] for i in range(0, len(rewards), config.rollouts_per_prompt)]
                advantages = compute_advantages(per_problem_rewards)
                per_problem_advantages = [advantages[i : i + config.rollouts_per_prompt] for i in range(0, len(advantages), config.rollouts_per_prompt)]
                
                # Remove samples that have been solved (all rollouts got reward 1)
                all_uids = [problem["prime_rl_data_uid"] for problem in problems]
                per_problem_uids = [all_uids[i] for i in range(0, len(all_uids), config.rollouts_per_prompt)]  # Take first UID from each group
                
                # postprocessing (difficulty re-prioritization etc.)
                datapool.maybe_postprocess(per_problem_uids, per_problem_rewards, per_problem_advantages)
                
                generated_samples = create_generated_samples(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    completion_logprobs=completion_logprobs,
                    prompt_masks=prompt_masks,
                    completion_masks=completion_masks,
                    rewards=rewards,
                    advantages=advantages
                )
                                    
                if not config.data_loading.online_difficulty_filtering_strategy.enabled:
                    all_generated_samples.extend(generated_samples)
                    break
                
                if sampling_iteration == config.data_loading.online_difficulty_filtering_strategy.max_sample_tries - 1:
                    all_generated_samples.extend(generated_samples)
                    all_generated_samples, remaining_generated_samples = all_generated_samples[:config.batch_size], all_generated_samples[config.batch_size:]
                    break
                
                keep_indices = [i for i, rewards_ in enumerate(per_problem_rewards) if config.data_loading.online_difficulty_filtering_strategy.min_avg_reward < np.mean(rewards_) < config.data_loading.online_difficulty_filtering_strategy.max_avg_reward]
                group_size = config.rollouts_per_prompt
                generated_samples = flatten_keep(generated_samples, keep_indices, group_size)

                if len(all_generated_samples) + len(keep_indices) >= problems_per_batch:
                    generated_samples, remaining_generated_samples = generated_samples[:config.batch_size], generated_samples[config.batch_size:]
                    all_generated_samples.extend(generated_samples)
                    datapool.add_to_buffer(remaining_generated_samples)  # TODO: Implement buffer functionality in DataPool
                    
                else:
                    all_generated_samples.extend(generated_samples)
                    
        unpacked_generated_samples = unpack_generated_samples(all_generated_samples)
        prompt_tokens = unpacked_generated_samples["prompt_tokens"]
        completion_tokens = unpacked_generated_samples["completion_tokens"]
        completion_logprobs = unpacked_generated_samples["completion_logprobs"]
        prompt_masks = unpacked_generated_samples["prompt_masks"]
        completion_masks = unpacked_generated_samples["completion_masks"]
        rewards = unpacked_generated_samples["reward"]
        advantages = unpacked_generated_samples["advantages"]

        generate_completions_time = time.time() - generate_completions_start_time

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
            prompt_tokens,
            prompt_masks,
            completion_tokens,
            completion_masks,
            completion_logprobs,
            advantages,
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
            "progress/orchestrator/step": ckpt_step,  # Shared W&B axis
            "progress/ckpt_step": ckpt_step,  # Shared W&B axis
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
            "reward/advantage_zero_ratio": np.mean(np.array(advantages) == 0.0),
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
