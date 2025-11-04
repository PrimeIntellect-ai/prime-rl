import pandas as pd
import asyncio
import time
from loguru import logger

# Import environment before any other imports
# ruff: noqa: I001,F401
from prime_rl.orchestrator import envs
from prime_rl.orchestrator.scheduler import Scheduler
from prime_rl.orchestrator.utils import get_sampling_args, monkey_patch_chat_completion_logprobs, set_semaphore

# This monkey patch is necessary to avoid heavy CPU overhead from constructing the OAI ChatCompletion Pydantic model with logprobs
monkey_patch_chat_completion_logprobs()

import lovely_tensors as lt
import torch
import verifiers as vf
from verifiers.types import GenerateOutputs, ProcessedOutputs
from transformers import AutoTokenizer

from prime_rl.orchestrator.ckpt import Progress, setup_ckpt_manager
from prime_rl.utils.vf import make_rollouts
from prime_rl.eval.utils import run_evals
from prime_rl.utils.vf import generate_batch
from prime_rl.utils.client import (
    check_has_model,
    check_health,
    init_nccl_broadcast,
    reload_weights,
    setup_admin_clients,
    setup_clients,
    setup_evals_client,
    update_weights,
)
from prime_rl.orchestrator.config import OrchestratorConfig, SimpleBufferConfig
from prime_rl.orchestrator.buffer import setup_buffer, Rollout
from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.utils.logger import setup_logger
from prime_rl.orchestrator.utils import (
    monkey_patch_chat_completion_logprobs,
    print_benchmark,
)
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import (
    clean_exit,
    format_num,
    get_rollout_dir,
    get_step_path,
    get_weights_dir,
    to_col_format,
)
import numpy as np


@clean_exit
@logger.catch(reraise=True)
async def orchestrate(config: OrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level, log_file=config.output_dir / "logs" / "orchestrator.log" if config.log.file else None
    )
    vf.setup_logging(level=config.log.vf_level.upper())
    logger.info("Starting orchestrator")

    # Print warning if running in benchmark mode
    if config.bench:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Setup client
    assert config.client.server_type == "vllm", "Orchestrator only supports vLLM server type."
    logger.info(
        f"Initializing clients (base_url={config.client.base_url}, api_key_var={config.client.api_key_var}, server_type={config.client.server_type})"
    )
    clients = setup_clients(config.client)
    admin_clients = setup_admin_clients(config.client)
    evals_client = setup_evals_client()

    # Load tokenizer
    logger.info(f"Initializing tokenizer for {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)

    # Setup monitor
    logger.info(f"Initializing monitor ({config.wandb})")
    monitor = setup_monitor(
        config.wandb,
        output_dir=config.output_dir,
        tokenizer=tokenizer,
        run_config=config,
    )

    # Load environment and extract dataset
    logger.info(
        f"Loading {len(config.env)} training environment(s) ({', '.join(env.name or env.id for env in config.env)})"
    )
    env = vf.EnvGroup(
        envs=[vf.load_environment(env.id, **env.args) for env in config.env],
        env_names=[env.name or env.id for env in config.env],
    )
    dataset = env.get_dataset(seed=config.seed)
    val_dataset = env.get_eval_dataset(config.val.num_examples, seed=config.seed) if config.val else None

    # Setup buffer
    logger.info(f"Setting up buffer ({config.buffer})")
    buffer = setup_buffer(dataset, config.buffer)
    val_buffer = setup_buffer(val_dataset, SimpleBufferConfig()) if val_dataset else None

    # Setup scheduler
    scheduler = Scheduler(
        clients=clients,
        admin_clients=admin_clients,
        env=env,
        buffer=buffer,
        tokenizer=tokenizer,
        config=config,
        oversampling_factor=config.oversampling_factor,
        max_async_level=config.max_async_level,
        max_off_policy_steps=config.max_off_policy_steps,
        strict_async_level=config.strict_async_level,
    )

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(admin_clients)
    await check_has_model(clients, config.model.name)
    logger.success("Inference pool ready")

    # Set up weight broadcast backend
    if config.weight_broadcast.type == "nccl":
        logger.info(f"Initializing NCCL broadcast ({config.weight_broadcast})")
        await init_nccl_broadcast(
            admin_clients, config.weight_broadcast.host, config.weight_broadcast.port, config.weight_broadcast.timeout
        )
    else:
        logger.info("Using filesystem for broadcasting weights into the inference pool.")

    # Get checkpoint manager
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)

    # Reset weights to base model if starting from scratch
    progress = Progress()
    ckpt_step = 0
    if config.ckpt and ckpt_manager and config.ckpt.resume_step:
        ckpt_manager.load(progress, buffer, step=config.ckpt.resume_step)
        logger.info(f"Resuming training from checkpoint step `{config.ckpt.resume_step}`")
        ckpt_step = progress.step  # Always resume from the latest checkpoint
        await update_weights(admin_clients, get_step_path(get_weights_dir(config.output_dir), ckpt_step))
    else:
        logger.info("Training from scratch. Resetting weights to base model")
        await reload_weights(admin_clients)

    # Iterate over dataset in batches
    max_steps = config.max_steps or int(1e9)
    logger.info(f"Starting orchestrator loop ({max_steps=}")
    last_eval_step = -1
    is_first_step = True
    if config.max_concurrent is not None:
        semaphore = asyncio.Semaphore(config.max_concurrent)
        set_semaphore(semaphore)

    # Start update policy loop
    asyncio.create_task(scheduler.update_policy_loop())

    while True:
        # Save checkpoint (if we are at an interval step and not at the first or last step)
        is_last_step = config.max_steps is not None and progress.step == config.max_steps - 1
        save_ckpt_time = 0
        if (
            ckpt_manager is not None
            and (config.ckpt and config.ckpt.interval)
            and not (is_first_step or is_last_step)
            and progress.step % config.ckpt.interval == 0
        ):
            logger.info(f"Saving checkpoint at step {progress.step}")
            save_ckpt_start_time = time.time()
            ckpt_manager.save(progress, buffer, step=progress.step)
            save_ckpt_time = time.time() - save_ckpt_start_time

            # Maybe clean up old orchestrator checkpoints
            ckpt_manager.maybe_clean()

        # Break if we have reached the maximum number of steps
        if config.max_steps and progress.step >= config.max_steps:
            break

        logger.info(f"Starting orchestrator step {progress.step}")
        step_start_time = time.time()

        # Schedule generating the training batch
        generate_completions_start_time = time.time()
        train_task = asyncio.create_task(scheduler.generate_batch(step=progress.step))

        # Schedule running evals at the specified interval
        if val_buffer and config.val and progress.step % config.val.interval == 0:
            logger.info(f"Running validation for step {progress.step}")
            val_problems = val_buffer.sample_problems(config.val.num_examples)
            val_task = asyncio.create_task(
                generate_batch(
                    clients=clients,
                    env=env,
                    model_name=config.model.name,
                    problems=val_problems,
                    rollouts_per_example=config.val.rollouts_per_example,
                    sampling_args=get_sampling_args(config.sampling),
                    pbar_description="Generating rollouts (val)",
                )
            )
        else:
            val_task = asyncio.create_task(asyncio.sleep(0))  # Dummy task

        # Schedule running evals at the specified interval
        if (
            config.eval
            and ckpt_step % config.eval.interval == 0
            and ckpt_step > last_eval_step
            and ((ckpt_step == 0 and config.eval.eval_base_model) or ckpt_step > 0)
        ):
            last_eval_step = ckpt_step
            logger.info(f"Running evals for checkpoint step {ckpt_step}")
            eval_task = asyncio.create_task(
                run_evals(
                    clients=clients,
                    eval_config=config.eval,
                    model_config=config.model,
                    sampling_config=config.eval.sampling,
                    client_config=config.client,
                    evals_client=evals_client,
                    output_dir=config.output_dir,
                    ckpt_step=ckpt_step,
                    step=progress.step,
                )
            )
        else:
            eval_task = asyncio.create_task(asyncio.sleep(0))  # Dummy task

        # Await train rollouts, process results and write batch to disk to consume by trainer
        await train_task
        generate_completions_time = time.time() - generate_completions_start_time
        train_rollouts = train_task.result()
        all_data_ranks_batches = prepare_batch(
            rollouts=train_rollouts,
            temperature=config.sampling.temperature,
            tokenizer=tokenizer,
            num_train_workers=config.num_train_workers,
            seq_len=config.seq_len,
        )

        step_path = get_rollout_dir(config.output_dir) / f"step_{progress.step}"
        step_path.mkdir(parents=True, exist_ok=True)
        for i, batches in enumerate(all_data_ranks_batches):
            batch_path = step_path / f"rank_{i}.pt"
            tmp_path = batch_path.with_suffix(".tmp")
            logger.debug(f"Saving rollouts for step {progress.step} for rank {i} to {batch_path}")
            torch.save(batches, tmp_path)
            tmp_path.rename(batch_path)

        # Await and process val results
        await val_task
        val_outputs = val_task.result()

        # Await eval results
        await eval_task

        # Gather metrics in dataframes
        results_df = pd.DataFrame(
            {
                "example_id": [rollout["example_id"] for rollout in train_rollouts],
                "task": [rollout["task"] for rollout in train_rollouts],
                "reward": [rollout["reward"] for rollout in train_rollouts],
                "advantage": [rollout["advantage"] for rollout in train_rollouts],
                "is_truncated": [rollout["is_truncated"] for rollout in train_rollouts],
                "completion_len": [len(rollout["completion_ids"]) for rollout in train_rollouts],
                "prompt_len": [len(rollout["prompt_ids"]) for rollout in train_rollouts],
                "seq_len": [len(rollout["prompt_ids"]) + len(rollout["completion_ids"]) for rollout in train_rollouts],
            }
        )

        # Gather individual reward function metrics
        metrics_df = pd.DataFrame([rollout["metrics"] for rollout in train_rollouts])

        val_results_df = (
            pd.DataFrame(
                {
                    "example_id": val_outputs.example_id,
                    "task": val_outputs.task,
                    "reward": val_outputs.reward,
                }
            )
            if val_outputs is not None
            else None
        )

        # Update progress metrics and throughput
        num_tokens = int(results_df.seq_len.sum())
        progress.total_tokens += num_tokens
        progress.total_samples += config.batch_size
        progress.total_problems += config.batch_size // config.rollouts_per_example
        throughput = num_tokens / generate_completions_time

        # Compute solve all and none tensors
        solve_all = (
            results_df.groupby("example_id")
            .apply(lambda x: x.reward.sum() == config.rollouts_per_example, include_groups=False)
            .mean()
        )
        solve_none = results_df.groupby("example_id").apply(lambda x: x.reward.sum() == 0, include_groups=False).mean()
        effective_batch_size = 1 - solve_none - solve_all

        # Compute per-env reuslts
        num_envs_in_batch = results_df.task.nunique()
        per_env_reward = results_df.groupby("task").reward.mean().to_dict() if num_envs_in_batch > 1 else None
        per_env_count = results_df.task.value_counts().to_dict() if num_envs_in_batch > 1 else None

        step_time = time.time() - step_start_time
        to_log = {
            # Progress metrics
            "progress/tokens": num_tokens,
            "progress/samples": config.batch_size,
            "progress/problems": config.batch_size // config.rollouts_per_example,
            "progress/total_tokens": progress.total_tokens,
            "progress/total_samples": progress.total_samples,
            "progress/total_problems": progress.total_problems,
            "progress/ckpt_step": ckpt_step,  # Shared W&B axis
            # Sequence length metrics
            "seq_len/mean": results_df.groupby("example_id").seq_len.mean().mean(),
            "seq_len/max": results_df.groupby("example_id").seq_len.max().mean(),
            "seq_len/min": results_df.groupby("example_id").seq_len.min().mean(),
            "prompt_len/mean": results_df.groupby("example_id").prompt_len.mean().mean(),
            "prompt_len/max": results_df.groupby("example_id").prompt_len.max().mean(),
            "prompt_len/min": results_df.groupby("example_id").prompt_len.min().mean(),
            "completion_len/mean": results_df.groupby("example_id").completion_len.mean().mean(),
            "completion_len/max": results_df.groupby("example_id").completion_len.max().mean(),
            "completion_len/min": results_df.groupby("example_id").completion_len.min().mean(),
            "is_truncated/mean": results_df.groupby("example_id").is_truncated.mean().mean(),
            "is_truncated/max": results_df.groupby("example_id").is_truncated.max().mean(),
            "is_truncated/min": results_df.groupby("example_id").is_truncated.min().mean(),
            # Performance metrics
            "perf/throughput": throughput,
            # Train reward
            "reward/mean": results_df.reward.mean(),
            # Batch metrics
            "batch/solve_none": solve_none,
            "batch/solve_all": solve_all,
            "batch/effective_batch_size": effective_batch_size,
            # Env metrics
            **{f"metrics/{metric}": metrics_df[metric].mean() for metric in metrics_df.columns},
            # Time metrics
            "time/step": step_time,
            "time/generate_completions": generate_completions_time,
            "time/save_ckpt": save_ckpt_time,
            # Scheduler metrics
            **scheduler.metrics(),
            # W&B axis
            "step": progress.step,
        }

        # If more than one env, add per-env metrics
        if results_df.task.nunique() > 1:
            per_env_reward = results_df.groupby("task").reward.mean().to_dict()
            to_log.update({f"reward/{env}": reward for env, reward in per_env_reward.items()})

            per_env_count = results_df.task.value_counts().to_dict()
            to_log.update({f"batch/{env}": count for env, count in per_env_count.items()})

        # Optionally, add val metrics
        if val_results_df is not None:
            to_log.update({"val_reward/mean": val_results_df.reward.mean()})

            if val_results_df.task.nunique() > 1:
                per_env_reward = val_results_df.groupby("task").reward.mean().to_dict()
                to_log.update({f"val_reward/{env}": reward for env, reward in per_env_reward.items()})

                per_env_count = val_results_df.task.value_counts().to_dict()
                to_log.update({f"val_batch/{env}": count for env, count in per_env_count.items()})

        # Log metrics to W&B
        monitor.log(to_log)

        # Log samples and distributions to W&B table if enabled
        monitor.log_samples(
            input_tokens=[rollout["prompt_ids"] for rollout in train_rollouts],
            output_tokens=[rollout["completion_ids"] for rollout in train_rollouts],
            rewards=results_df.reward.tolist(),
            advantages=results_df.advantage.tolist(),
            rollouts_per_problem=config.rollouts_per_example,
            step=progress.step,
        )

        distributions = {
            "rewards": results_df.reward.tolist(),
            "advantages": results_df.advantage.tolist(),
            "problem_rewards": results_df.groupby("example_id").reward.mean().tolist(),
            "problem_advantages": results_df.groupby("example_id").advantage.mean().tolist(),
        }

        # Log distributions to W&B table
        monitor.log_distributions(distributions=distributions, step=progress.step)

        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {results_df.reward.mean():.4f} |{f' Val. Reward: {val_results_df.reward.mean():.4f} |' if val_results_df is not None else ''} Throughput: {throughput:.1f} tokens/s | Seq. Length: {results_df.seq_len.mean():.1f} tokens/sample | Async Level: {scheduler.async_level} | Max. Off-Policy Level: {scheduler.max_off_policy_level}"
        logger.success(step_message)

        # Increment step
        progress.step += 1
        is_first_step = False

    if config.eval:
        logger.info("Running final evals")
        await run_evals(
            clients=clients,
            eval_config=config.eval,
            model_config=config.model,
            sampling_config=config.eval.sampling,
            client_config=config.client,
            evals_client=evals_client,
            output_dir=config.output_dir,
            ckpt_step=ckpt_step,
            step=progress.step,
        )

    # Log final (immutable) samples and distributions to W&B table
    monitor.log_final_samples()
    monitor.log_final_distributions()
    monitor.save_final_summary()

    # Write final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(progress, buffer, step=progress.step)

    logger.success("Orchestrator finished.")

    # Optionally, print benchmark table
    if config.bench:
        print_benchmark(to_col_format(monitor.history))


def main():
    """Main entry-point for orchestrator. Run using `uv run orchestrator`"""

    asyncio.run(orchestrate(parse_argv(OrchestratorConfig)))


if __name__ == "__main__":
    main()
