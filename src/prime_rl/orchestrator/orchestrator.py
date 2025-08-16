import asyncio
import time
from loguru import logger

# Import environment before any other imports
# ruff: noqa: I001,F401
from prime_rl.orchestrator import envs

import lovely_tensors as lt
import torch
from verifiers import load_environment
from verifiers.types import GenerateOutputs, ProcessedOutputs
from transformers import AutoTokenizer

from prime_rl.orchestrator.ckpt import RLProgress as Progress, setup_ckpt_manager
from prime_rl.eval.utils import run_eval
from prime_rl.orchestrator.client import (
    check_has_model,
    check_health,
    reload_weights,
    update_weights,
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
    parse_truncated_completions,
)
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import (
    clean_exit,
    format_num,
    get_rollout_dir,
    get_weights_dir,
    to_col_format,
)
import numpy as np


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
    logger.info(
        f"Initializing OpenAI client ({config.client.host}:{config.client.port})"
    )
    client = setup_client(config.client)

    # Load tokenizer
    logger.info(f"Initializing tokenizer for {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    # Setup monitor
    logger.info(f"Initializing monitor ({config.monitor})")
    monitor = setup_monitor(
        config.monitor,
        outputs_dir=config.outputs_dir,
        tokenizer=tokenizer,
        run_config=config,
    )

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(client)
    await check_has_model(client, config.model.name)
    logger.success("Inference pool ready")

    # Get checkpoint manager
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.outputs_dir, config.ckpt)

    # Reset weights to base model if starting from scratch
    progress = Progress()
    ckpt_step = 0
    if config.ckpt and ckpt_manager and config.ckpt.resume_step:
        logger.info(
            f"Resuming training from checkpoint step `{config.ckpt.resume_step}`"
        )
        ckpt_manager.load(progress, step=config.ckpt.resume_step)
        ckpt_step = max(progress.step - config.async_level, 0)
        await update_weights(client, get_weights_dir(config.outputs_dir), ckpt_step)
    else:
        logger.info("Training from scratch. Resetting weights to base model")
        await reload_weights(client)

    # Load environment and extract dataset
    logger.info(
        f"Loading environment {config.environment.id} with args {config.environment.args}"
    )
    vf_env = load_environment(config.environment.id, **config.environment.args)
    dataset = vf_env.get_dataset(seed=config.seed)

    # Setup buffer
    logger.info(f"Setting up buffer ({config.buffer})")
    buffer = setup_buffer(dataset, config.buffer)

    # Iterate over dataset in batches
    max_steps = config.max_steps or int(1e9)
    logger.info(f"Starting orchestrator loop ({max_steps=}")
    ckpt_step = 0
    last_eval_step = -1
    is_first_step = True
    while True:
        # Save checkpoint (if we are at an interval step and not at the first or last step)
        is_last_step = (
            config.max_steps is not None and progress.step == config.max_steps - 1
        )
        save_ckpt_time = 0
        if (
            ckpt_manager is not None
            and (config.ckpt and config.ckpt.interval)
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(progress, step=progress.step)
            save_ckpt_time = time.time() - save_ckpt_start_time

            # Maybe clean up old orchestrator checkpoints
            ckpt_manager.maybe_clean()

        # Break if we have reached the maximum number of steps
        if config.max_steps and progress.step >= config.max_steps:
            break

        logger.info(f"Starting orchestrator step {progress.step} ({ckpt_step=})")
        step_start_time = time.time()

        # Optionally, wait for the next checkpoint to be available
        wait_for_weight_ckpt_time, update_weights_time = 0, 0
        if progress.step - ckpt_step > config.async_level:
            logger.debug(
                f"Hit async barrier because step {progress.step} is {progress.step - ckpt_step} (>{config.async_level}) steps ahead of checkpoint step {ckpt_step}."
            )

            # Wait for the checkpoint to be available
            ckpt_step = progress.step - config.async_level
            logger.info(f"Waiting for weight checkpoint {ckpt_step}")
            wait_for_weight_ckpt_start_time = time.time()
            wait_for_weight_checkpoint(get_weights_dir(config.outputs_dir), ckpt_step)
            wait_for_weight_ckpt_time = time.time() - wait_for_weight_ckpt_start_time
            logger.debug(
                f"Waited {wait_for_weight_ckpt_time:.2f}s for weight checkpoint"
            )

            # Update the weights
            logger.info(f"Updating weights to weight checkpoint {ckpt_step}")
            update_weights_start_time = time.time()
            await update_weights(client, get_weights_dir(config.outputs_dir), ckpt_step)
            update_weights_time = time.time() - update_weights_start_time
            logger.debug(f"Updated weights in {update_weights_time:.2f}s")

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
                    run_eval(
                        client=client,
                        eval_id=eval_id,
                        env_args=config.eval.args.get(eval_id, {}),
                        model_config=config.model,
                        sampling_config=config.eval.sampling,
                        num_examples=num_examples,
                        rollouts_per_example=rollouts_per_example,
                        ckpt_step=ckpt_step,
                        monitor=monitor,
                        step=progress.step,
                    )
                    for eval_id, num_examples, rollouts_per_example in zip(
                        config.eval.ids,
                        config.eval.num_examples,
                        config.eval.rollouts_per_example,
                    )
                ]
            )
            eval_time = time.time() - eval_start_time
            logger.info(f"Evaluated in {eval_time:.2f}s")

        accepted_rollouts: list[Rollout] = []
        problem_requests, completion_requests, calls_to_generate = 0, 0, 0
        problems_per_batch = config.batch_size // config.rollouts_per_example
        problems_to_sample = problems_per_batch
        while True:
            # Get the batch
            problem_ids, problems = buffer.sample_problems(problems_to_sample)

            # Duplicate problems `rollouts_per_example` times
            problem_ids = [
                problem_id
                for problem_id in problem_ids
                for _ in range(config.rollouts_per_example)
            ]
            problems = [
                problem
                for problem in problems
                for _ in range(config.rollouts_per_example)
            ]

            # Prepare inputs for verifiers generation
            # TODO: Can we use `prime_rl.utils.utils.to_col_format` here?
            inputs = {
                "prompt": [problem["prompt"] for problem in problems],
                "info": [problem.get("info", {}) for problem in problems],
                "task": [problem["task"] for problem in problems],
                "answer": [problem.get("answer", "") for problem in problems],
            }

            # Convert SamplingConfig to vLLM OAI sampling args
            # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_2
            sampling_args = dict(config.sampling)
            sampling_args["top_p"] = 1.0
            sampling_args["logprobs"] = True
            sampling_args["extra_body"] = {
                "return_tokens_as_token_ids": True,
                "top_k": -1,
                "min_p": 0.0,
            }
            sampling_args["extra_body"]["min_tokens"] = sampling_args.pop("min_tokens")

            # Generate completions + rewards with verifiers
            logger.info(f"Sending {len(problems)} requests to environments")
            generate_completions_start_time = time.time()
            generate_outputs: GenerateOutputs = await vf_env.a_generate(
                inputs=inputs,
                client=client,
                model=config.model.name,
                sampling_args=sampling_args,
            )
            generate_completions_time = time.time() - generate_completions_start_time
            problem_requests += problems_to_sample
            completion_requests += problems_to_sample * config.rollouts_per_example
            calls_to_generate += 1

            processed_outputs: ProcessedOutputs = vf_env.process_env_results_vllm(
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

            # Compute advantages
            advantages = compute_advantages(
                rewards=processed_outputs.rewards,
                completion_lengths=list(map(len, processed_outputs.completion_ids)),
                samples_per_problem=config.rollouts_per_example,
                advantage_type=config.advantage_type,
            )

            # Parse whether the completions were truncated
            is_truncated = parse_truncated_completions(states=generate_outputs.state)

            # Update pool
            rollouts = make_rollouts(
                problem_ids=problem_ids,
                prompt_tokens=processed_outputs.prompt_ids,
                prompt_masks=processed_outputs.prompt_mask,
                completion_tokens=processed_outputs.completion_ids,
                completion_masks=processed_outputs.completion_mask,
                completion_logprobs=processed_outputs.completion_logprobs,
                is_truncated=is_truncated,
                rewards=processed_outputs.rewards,
                advantages=advantages,
            )
            buffer.update(rollouts)
            accepted_rollouts.extend(buffer.sample_rollouts(problems_to_sample))

            # Break if we have enough rollouts to fill the batch
            if len(accepted_rollouts) >= config.batch_size:
                accepted_rollouts = accepted_rollouts[: config.batch_size]
                break

            # On next iteration, sample the remaining problems to fill the batch
            problems_sampled = len(accepted_rollouts) // config.rollouts_per_example
            problems_to_sample = problems_per_batch - problems_sampled

        # Unpack accepted rollouts
        rewards = (
            torch.tensor([rollout.reward for rollout in accepted_rollouts])
            .reshape(-1, config.rollouts_per_example)
            .float()
        )
        advantages = (
            torch.tensor([rollout.advantage for rollout in accepted_rollouts])
            .reshape(-1, config.rollouts_per_example)
            .float()
        )
        is_truncated = (
            torch.tensor([rollout.is_truncated for rollout in accepted_rollouts])
            .reshape(-1, config.rollouts_per_example)
            .float()
        )
        assert (
            rewards.shape
            == advantages.shape
            == is_truncated.shape
            == (problems_per_batch, config.rollouts_per_example)
        )
        assert (
            rewards.numel()
            == advantages.numel()
            == is_truncated.numel()
            == config.batch_size
        )
        prompt_tokens = [rollout.prompt_tokens for rollout in accepted_rollouts]
        completion_tokens = [rollout.completion_tokens for rollout in accepted_rollouts]
        prompt_lens = (
            torch.tensor([len(p) for p in prompt_tokens])
            .float()
            .reshape(-1, config.rollouts_per_example)
        )
        completion_lens = (
            torch.tensor([len(c) for c in completion_tokens])
            .float()
            .reshape(-1, config.rollouts_per_example)
        )
        seq_lens = prompt_lens + completion_lens
        assert (
            seq_lens.shape
            == prompt_lens.shape
            == completion_lens.shape
            == (problems_per_batch, config.rollouts_per_example)
        )
        assert (
            seq_lens.numel()
            == prompt_lens.numel()
            == completion_lens.numel()
            == config.batch_size
        )
        assert is_truncated.shape == (problems_per_batch, config.rollouts_per_example)
        assert is_truncated.numel() == config.batch_size

        logger.debug(f"Got rewards: {lt.lovely(rewards)}")
        logger.debug(
            f"Got advantages ({config.advantage_type}): {lt.lovely(advantages)}"
        )

        # Compute progress metrics and throughput
        num_tokens = int(seq_lens.sum().item())
        progress.total_tokens += num_tokens
        progress.total_samples += config.batch_size
        progress.total_problems += config.batch_size // config.rollouts_per_example
        throughput = num_tokens / (generate_completions_time)

        # Compute solve all and none tensors
        solve_all = (
            rewards.sum(-1).eq(config.rollouts_per_example).float().mean().item()
        )
        solve_none = rewards.sum(-1).eq(0).float().mean().item()
        effective_batch_size = 1 - solve_none - solve_all

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

        step_path = get_rollout_dir(config.outputs_dir) / f"step_{progress.step}"
        step_path.mkdir(parents=True, exist_ok=True)
        for i, batches in enumerate(all_data_ranks_batches):
            batch_path = step_path / f"rank_{i}.pt"
            tmp_path = batch_path.with_suffix(".tmp")
            logger.debug(
                f"Saving rollouts for step {progress.step} for rank {i} to {batch_path}"
            )
            torch.save(batches, tmp_path)
            tmp_path.rename(batch_path)

        # Log step metrics
        step_time = time.time() - step_start_time
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {rewards.mean().item():.4f} | Throughput: {throughput:.1f} tokens/s | Seq. Length: {seq_lens.mean().item():.1f} tokens/sample"
        logger.success(step_message)

        # Log progress metrics to monitor
        progress_metrics = {
            "progress/tokens": num_tokens,
            "progress/samples": config.batch_size,
            "progress/problems": config.batch_size // config.rollouts_per_example,
            "progress/total_tokens": progress.total_tokens,
            "progress/total_samples": progress.total_samples,
            "progress/total_problems": progress.total_problems,
            "progress/ckpt_step": ckpt_step,  # Shared W&B axis
            "step": progress.step,
        }
        monitor.log(progress_metrics)

        # Log sequence lengths to monitor (first reduce over group dimension, then over problem dimension)
        seq_len_metrics = {
            "seq_len/mean": seq_lens.mean(-1).mean().item(),
            "seq_len/max": seq_lens.mean(-1).max().item(),
            "seq_len/min": seq_lens.mean(-1).min().item(),
            "step": progress.step,
        }
        monitor.log(seq_len_metrics)

        prompt_len_metrics = {
            "prompt_len/mean": prompt_lens.mean(-1).mean().item(),
            "prompt_len/max": prompt_lens.mean(-1).max().item(),
            "prompt_len/min": prompt_lens.mean(-1).min().item(),
            "step": progress.step,
        }
        monitor.log(prompt_len_metrics)

        completion_len_metrics = {
            "completion_len/mean": completion_lens.mean(-1).mean().item(),
            "completion_len/max": completion_lens.mean(-1).max().item(),
            "completion_len/min": completion_lens.mean(-1).min().item(),
            "step": progress.step,
        }
        monitor.log(completion_len_metrics)

        truncated_metrics = {
            "is_truncated/mean": is_truncated.mean(-1).mean().item(),
            "is_truncated/max": is_truncated.mean(-1).max().item(),
            "is_truncated/min": is_truncated.mean(-1).min().item(),
            "step": progress.step,
        }
        monitor.log(truncated_metrics)

        # Log performance metrics to monitor
        perf_metrics = {
            "perf/throughput": throughput,
            "perf/problem_requests": problem_requests,
            "perf/completion_requests": completion_requests,
            "perf/calls_to_generate": calls_to_generate,
            "step": progress.step,
        }
        monitor.log(perf_metrics)

        # Log reward metrics to monitor
        reward_metrics = {
            "reward/mean": rewards.mean().item(),
            "step": progress.step,
        }
        monitor.log(reward_metrics)

        # Log rewards metrics to monitor
        solve_metrics = {
            "batch/solve_none": solve_none,
            "batch/solve_all": solve_all,
            "batch/effective_batch_size": effective_batch_size,
            "step": progress.step,
        }
        monitor.log(solve_metrics)

        # Log time metrics to monitor
        time_metrics = {
            "time/step": step_time,
            "time/wait_for_weight_ckpt": wait_for_weight_ckpt_time,
            "time/generate_completions": generate_completions_time,
            "time/update_weights": update_weights_time,
            "time/save_ckpt": save_ckpt_time,
            "time/eval": eval_time,
            "step": progress.step,
        }
        monitor.log(time_metrics)

        # Log samples and distributions to W&B table if enabled
        if monitor.wandb:
            monitor.wandb.log_samples(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                rewards=rewards.flatten().tolist(),
                advantages=advantages.flatten().tolist(),
                rollouts_per_problem=config.rollouts_per_example,
                step=progress.step,
            )
            monitor.wandb.log_distributions(
                distributions={
                    "rewards": rewards.flatten().tolist(),
                    "advantages": advantages.flatten().tolist(),
                    "problem_rewards": rewards.mean(-1).tolist(),
                    "problem_advantages": advantages.mean(-1).tolist(),
                },
                step=progress.step,
            )

        # Increment progress
        progress.step += 1
        is_first_step = False

    # Log final (immutable) samples and distributions to W&B table
    if monitor.wandb:
        logger.info("Logging final samples and distributions as W&B table")
        monitor.wandb.log_final_samples()
        monitor.wandb.log_final_distributions()

    # Write final checkpoint
    if ckpt_manager is not None:
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
