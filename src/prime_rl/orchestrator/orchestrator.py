import asyncio
import random
import time

import tomli_w

from prime_rl.orchestrator.advantage import compute_advantages, compute_advantages_multi_reward
from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor
from prime_rl.orchestrator.patches import monkey_patch_chat_completion_logprobs, monkey_patch_oai_iterable_types
from prime_rl.orchestrator.trajectories import branch_rollout, interleave_rollout
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender

# This monkey patch is necessary to avoid Pydantic validating fields using typing.Iterable (e.g. in multimodal or tool call messages) lazily which leads to tokenization errors, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1249
monkey_patch_oai_iterable_types()


# This monkey patch is necessary to avoid heavy CPU overhead from constructing the OAI ChatCompletion Pydantic model with logprobs, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1189
monkey_patch_chat_completion_logprobs()

# Import environment before any other imports
import pandas as pd
import verifiers as vf
from loguru import logger
from transformers import AutoTokenizer

from prime_rl.eval.utils import run_evals_subprocess
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.ckpt import Progress, setup_ckpt_manager
from prime_rl.orchestrator.config import BufferConfig, OrchestratorConfig
from prime_rl.orchestrator.scheduler import Scheduler
from prime_rl.orchestrator.utils import (
    compute_teacher_logprobs,
    get_sampling_args,
    print_benchmark,
    set_semaphore,
)
from prime_rl.utils.client import (
    check_has_model,
    check_health,
    init_nccl_broadcast,
    reload_weights,
    setup_admin_clients,
    setup_clients,
    update_weights,
)
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import intercept_verifiers_logging, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import (
    clean_exit,
    get_broadcast_dir,
    get_env_ids_to_install,
    get_step_path,
    install_env,
    resolve_latest_ckpt_step,
    to_col_format,
)
from prime_rl.utils.vf import generate_batch, get_completion_len, get_prompt_len, get_seq_len


@clean_exit
@logger.catch(reraise=True)
async def orchestrate(config: OrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level, log_file=config.output_dir / "logs" / "orchestrator.log" if config.log.file else None
    )
    intercept_verifiers_logging(level=config.log.vf_level)
    logger.info("Starting orchestrator")

    event_loop_lag_monitor = EventLoopLagMonitor()
    event_loop_lag_monitor_task = asyncio.create_task(event_loop_lag_monitor.run())

    # Print warning if running in benchmark mode
    if config.bench:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Save configs to output directory
    config_dir = config.output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), f)

    # Install environments
    env_ids_to_install = set()
    env_ids_to_install.update(get_env_ids_to_install(config.env))
    if config.eval is not None:
        env_ids_to_install.update(get_env_ids_to_install(config.eval.env))

    for env_id in env_ids_to_install:
        install_env(env_id)

    # Setup client
    logger.info(
        f"Initializing OpenAI client (base_url={', '.join(config.client.base_url)}, api_key_var={config.client.api_key_var}, headers={config.client.headers})"
    )
    clients = setup_clients(config.client)
    admin_clients = setup_admin_clients(config.client)

    # Setup teacher model client if configured
    teacher_clients = None
    teacher_admin_clients = None
    teacher_model_name = None
    if config.teacher_model:
        logger.info(
            f"Initializing teacher OpenAI client (base_url={', '.join(config.teacher_model.client.base_url)}, "
            f"model={config.teacher_model.model.name})"
        )
        teacher_clients = setup_clients(config.teacher_model.client)
        teacher_admin_clients = setup_admin_clients(config.teacher_model.client)
        teacher_model_name = config.teacher_model.model.name

    # Load tokenizer
    logger.info(f"Initializing tokenizer for {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)

    # Setup monitor
    logger.info(f"Initializing monitor (wandb={config.wandb}, prime_monitor={config.prime_monitor})")
    monitor = setup_monitor(
        wandb_config=config.wandb,
        prime_config=config.prime_monitor,
        output_dir=config.output_dir,
        tokenizer=tokenizer,
        run_config=config,
    )

    # Setup heartbeat (only on rank 0, orchestrator is single process)
    heart = None
    if config.heartbeat is not None:
        logger.info("Initializing heartbeat")
        heart = Heartbeat(config.heartbeat.url)

    # Load environment and extract dataset
    logger.info(
        f"Loading {len(config.env)} training environment(s) ({', '.join(env.name or env.id for env in config.env)})"
    )
    env = vf.EnvGroup(
        envs=[vf.load_environment(env.id, **env.args) for env in config.env],
        env_names=[env.name or env.id for env in config.env],
        map_kwargs=dict(writer_batch_size=1),  # Set defensively to not error on map operations on large datasets
    )
    env.set_max_seq_len(config.seq_len)
    if config.trajectory_strategy == "interleaved":
        logger.info("Using token prompts in environment to avoid retokenization discrepancies in multi-turn rollouts")
        env.set_interleaved_rollouts(True)
    if config.buffer.skip_verification:
        logger.info("Skipping verification (rewards will be set to 0)")
        env.set_score_rollouts(False)

    # Setup buffer
    logger.info(f"Setting up buffer ({config.buffer})")
    train_dataset = env.get_dataset(seed=config.buffer.seed)
    buffer = Buffer(train_dataset, env.env_names, config.buffer)
    if config.val is not None:
        val_buffer_config = BufferConfig(env_ratios=config.buffer.env_ratios)
        val_dataset = env.get_eval_dataset(seed=val_buffer_config.seed)
        val_buffer = Buffer(val_dataset, env.env_names, val_buffer_config)
    else:
        val_buffer = None

    # Get checkpoint manager
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)

    checkpoint_step = None
    if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
        if config.ckpt.resume_step == -1:
            checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
        else:
            checkpoint_step = config.ckpt.resume_step

    # Setup scheduler (uses subprocess workers for env execution)
    scheduler = Scheduler(
        admin_clients=admin_clients,
        client_config=config.client,
        env_configs=config.env,
        buffer=buffer,
        config=config,
        oversampling_factor=config.oversampling_factor,
        max_async_level=config.max_async_level,
        max_off_policy_steps=config.max_off_policy_steps,
        strict_async_level=config.strict_async_level,
        lora_name=config.model.lora.name if config.model.lora else None,
        output_dir=config.output_dir,
    )

    if checkpoint_step is not None and config.model.lora is not None:
        scheduler.model_name = config.model.lora.name
        for workers in scheduler.workers.values():
            for worker in workers:
                worker.model_name = config.model.lora.name

    await scheduler.start()

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(admin_clients)
    await check_has_model(clients, config.model.name)
    logger.success("Inference pool ready")

    # Check health of teacher inference server if configured
    if teacher_admin_clients is not None:
        logger.info("Waiting for teacher inference pool to be ready")
        await check_health(teacher_admin_clients)
        await check_has_model(teacher_clients, teacher_model_name)
        logger.success("Teacher inference pool ready")

    # Set up weight broadcast backend
    logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
    if config.weight_broadcast.type == "nccl":
        await init_nccl_broadcast(
            admin_clients, config.weight_broadcast.host, config.weight_broadcast.port, config.weight_broadcast.timeout
        )

    # Setup training batch sender for sending training examples to trainer
    logger.info(f"Initializing training batch sender ({config.rollout_transport})")
    training_batch_sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

    # Track last online eval checkpoint step for this process
    last_eval_step = -1

    # Reset weights to base model if starting from scratch
    progress = Progress()

    if checkpoint_step is not None and ckpt_manager is not None:
        ckpt_manager.load(progress, buffer, step=checkpoint_step)
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")
        scheduler.ckpt_step = progress.step  # Always resume from the latest checkpoint
        if config.eval and config.eval.skip_eval_on_resume:
            last_eval_step = scheduler.ckpt_step
            logger.info(f"Skipping online eval on resume (ckpt_step={scheduler.ckpt_step})")
        await update_weights(
            admin_clients,
            get_step_path(get_broadcast_dir(config.output_dir), scheduler.ckpt_step),
            lora_name=config.model.lora.name if config.model.lora else None,
        )
    else:
        logger.info("Training from scratch. Resetting weights to base model")
        if config.model.lora is None:
            await reload_weights(admin_clients)

    # Iterate over dataset in batches
    max_steps = config.max_steps or int(1e9)
    logger.info(f"Starting orchestrator loop (max_steps={max_steps or 'infinite'})")
    is_first_step = True
    await set_semaphore(config.max_concurrent or -1)

    # Start update policy loop
    update_policy_task = asyncio.create_task(scheduler.update_policy_loop())

    while True:
        # Check if update_policy_task has failed and propagate the exception
        if update_policy_task.done():
            # End all other tasks
            for task in asyncio.all_tasks():
                task.cancel()
            update_policy_task.result()  # Raises if the task failed
        # Capture ckpt_step once for consistency (it's updated by update_policy_loop concurrently)
        ckpt_step = scheduler.ckpt_step

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
            save_ckpt_start_time = time.perf_counter()
            ckpt_manager.save(progress, buffer, step=progress.step)
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time

        # Break if we have reached the maximum number of steps
        if config.max_steps and progress.step >= config.max_steps:
            break

        logger.info(f"Starting orchestrator step {progress.step}")
        step_start_time = time.perf_counter()

        # Run evals BEFORE training (blocking, in subprocess to isolate event loop)
        # This ensures weights don't change during eval and eval doesn't cause event loop lag
        if (
            config.eval
            and ckpt_step % config.eval.interval == 0
            and ckpt_step > last_eval_step
            and ((ckpt_step == 0 and config.eval.eval_base_model) or ckpt_step > 0)
        ):
            last_eval_step = ckpt_step
            logger.info(f"Running evals for checkpoint step {ckpt_step} (blocking, subprocess)")

            # Pause weight updates during eval
            scheduler.checkpoint_ready.clear()

            await run_evals_subprocess(
                client_config=config.client,
                eval_config=config.eval,
                model_config=config.model,
                sampling_config=config.eval.sampling,
                reasoning_field=config.eval.reasoning_field,
                output_dir=config.output_dir,
                ckpt_step=ckpt_step,
                step=progress.step,
                max_concurrent=config.max_concurrent or -1,
            )

            # Resume weight updates
            scheduler.checkpoint_ready.set()

        # Schedule generating the training batch
        generate_completions_start_time = time.perf_counter()
        train_task = asyncio.create_task(scheduler.generate_batch(step=progress.step))

        # Schedule running validation at the specified interval
        if val_buffer and config.val and progress.step % config.val.interval == 0:
            logger.info(f"Running validation for step {progress.step}")
            val_examples = val_buffer.sample_examples(config.val.num_examples)
            val_task = asyncio.create_task(
                generate_batch(
                    clients=clients,
                    env=env,
                    model_name=config.model.name,
                    examples=val_examples,
                    rollouts_per_example=config.val.rollouts_per_example,
                    sampling_args=get_sampling_args(config.sampling),
                    pbar_description="Generating rollouts (val)",
                )
            )
        else:
            val_task = asyncio.create_task(asyncio.sleep(0))  # Dummy task

        # Await train rollouts, process results and write batch to disk to consume by trainer
        await train_task
        generate_completions_time = time.perf_counter() - generate_completions_start_time
        train_rollouts = train_task.result()

        # Compute advantages (per-env, supporting mixed single/multi-reward configurations)
        env_config_map = {(env_cfg.name or env_cfg.id): env_cfg for env_cfg in config.env}
        advantages = [0.0] * len(train_rollouts)
        rewards = [rollout["reward"] for rollout in train_rollouts]

        # Group rollout indices by environment
        env_indices: dict[str, list[int]] = {}
        for idx, rollout in enumerate(train_rollouts):
            env_name = rollout["task"]
            if env_name not in env_indices:
                env_indices[env_name] = []
            env_indices[env_name].append(idx)

        # Compute advantages for each environment separately
        for env_name, indices in env_indices.items():
            env_cfg = env_config_map[env_name]
            env_rollouts = [train_rollouts[i] for i in indices]

            if env_cfg.reward_keys is not None:
                # Multi-reward path
                if env_cfg.reward_weights is not None and len(env_cfg.reward_weights) != len(env_cfg.reward_keys):
                    raise ValueError(
                        f"reward_weights length ({len(env_cfg.reward_weights)}) must match "
                        f"reward_keys length ({len(env_cfg.reward_keys)}) for env '{env_name}'"
                    )
                metrics = [r["metrics"] for r in env_rollouts]
                env_advantages = compute_advantages_multi_reward(
                    metrics,
                    env_cfg.reward_keys,
                    config.rollouts_per_example,
                    config.advantage,
                    env_cfg.reward_weights,
                )
            else:
                # Single-reward path
                env_rewards = [r["reward"] for r in env_rollouts]
                env_completion_lens = [get_completion_len(r) for r in env_rollouts]
                env_advantages = compute_advantages(
                    env_rewards,
                    env_completion_lens,
                    config.rollouts_per_example,
                    config.advantage,
                )

            # Place advantages back in original order
            for i, adv in zip(indices, env_advantages):
                advantages[i] = adv

        # Update and sample rollouts from the buffer
        make_train_example = interleave_rollout if config.trajectory_strategy == "interleaved" else branch_rollout
        train_examples: list[TrainingSample] = []
        for train_rollout, advantage in zip(train_rollouts, advantages):
            train_example = make_train_example(train_rollout)
            if train_example is not None:
                for te in train_example:
                    te.advantage = advantage
                    te.reward = train_rollout["reward"]
                train_examples.extend(train_example)
        logger.debug(
            f"Converted {len(train_rollouts)} training rollouts to {len(train_examples)} training examples using {config.trajectory_strategy} strategy"
        )

        # Compute teacher logprobs if teacher model is configured
        teacher_logprobs_time = 0
        if config.teacher_model is not None:
            logger.info(f"Computing teacher logprobs for {len(train_examples)} training examples")
            teacher_logprobs_start_time = time.perf_counter()
            teacher_logprobs_list = await compute_teacher_logprobs(
                clients=teacher_clients,
                model_name=teacher_model_name,
                samples=train_examples,
            )
            for train_example, teacher_logprobs in zip(train_examples, teacher_logprobs_list):
                train_example.teacher_logprobs = teacher_logprobs
            teacher_logprobs_time = time.perf_counter() - teacher_logprobs_start_time
            logger.debug(f"Computed teacher logprobs in {teacher_logprobs_time:.2f}s")

        training_batch = TrainingBatch(
            examples=train_examples,
            temperature=config.sampling.temperature,
            step=progress.step,
        )
        assert len(training_batch.examples) != 0, "Step with no samples is not allowed"
        training_batch_sender.send(training_batch)

        # Await and process val results
        await val_task
        val_outputs = val_task.result()

        # Gather metrics in dataframes
        results_df = pd.DataFrame(
            {
                "example_id": [rollout["example_id"] for rollout in train_rollouts],
                "task": [rollout["task"] for rollout in train_rollouts],
                "reward": [rollout["reward"] for rollout in train_rollouts],
                "is_truncated": [rollout["is_truncated"] for rollout in train_rollouts],
                "error": [rollout["error"] for rollout in train_rollouts],
                "completion_len": [get_completion_len(rollout) for rollout in train_rollouts],
                "prompt_len": [get_prompt_len(rollout) for rollout in train_rollouts],
                "seq_len": [get_seq_len(rollout) for rollout in train_rollouts],
                "num_turns": [len(rollout["trajectory"]) for rollout in train_rollouts],
                "generation_ms": [rollout["timing"]["generation_ms"] for rollout in train_rollouts],
                "scoring_ms": [rollout["timing"]["scoring_ms"] for rollout in train_rollouts],
            }
        )

        # Gather individual reward function metrics
        metrics_df = pd.DataFrame([rollout["metrics"] for rollout in train_rollouts])

        val_results_df = (
            pd.DataFrame(
                {
                    "example_id": [rollout["input"]["example_id"] for rollout in val_outputs],
                    "task": [rollout["input"]["task"] for rollout in val_outputs],
                    "reward": [rollout["reward"] for rollout in val_outputs],
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

        step_time = time.perf_counter() - step_start_time
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
            "seq_len/max": results_df.groupby("example_id").seq_len.mean().max(),
            "seq_len/min": results_df.groupby("example_id").seq_len.mean().min(),
            "prompt_len/mean": results_df.groupby("example_id").prompt_len.mean().mean(),
            "prompt_len/max": results_df.groupby("example_id").prompt_len.mean().max(),
            "prompt_len/min": results_df.groupby("example_id").prompt_len.mean().min(),
            "completion_len/mean": results_df.groupby("example_id").completion_len.mean().mean(),
            "completion_len/max": results_df.groupby("example_id").completion_len.mean().max(),
            "completion_len/min": results_df.groupby("example_id").completion_len.mean().min(),
            "is_truncated/mean": results_df.groupby("example_id").is_truncated.mean().mean(),
            "is_truncated/max": results_df.groupby("example_id").is_truncated.mean().max(),
            "is_truncated/min": results_df.groupby("example_id").is_truncated.mean().min(),
            # Turn metrics
            "num_turns/mean": results_df.groupby("example_id").num_turns.mean().mean(),
            "num_turns/max": results_df.groupby("example_id").num_turns.mean().max(),
            "num_turns/min": results_df.groupby("example_id").num_turns.mean().min(),
            # Verifier timing metrics
            "generation_ms/mean": results_df.groupby("example_id").generation_ms.mean().mean(),
            "generation_ms/max": results_df.groupby("example_id").generation_ms.mean().max(),
            "generation_ms/min": results_df.groupby("example_id").generation_ms.mean().min(),
            "scoring_ms/mean": results_df.groupby("example_id").scoring_ms.mean().mean(),
            "scoring_ms/max": results_df.groupby("example_id").scoring_ms.mean().max(),
            "scoring_ms/min": results_df.groupby("example_id").scoring_ms.mean().min(),
            # Performance metrics
            "perf/throughput": throughput,
            # Train reward
            "reward/mean": results_df.reward.mean(),
            # Batch metrics
            "batch/solve_none": solve_none,
            "batch/solve_all": solve_all,
            "batch/effective_batch_size": effective_batch_size,
            # Error metrics
            "error/mean": (~results_df.error.isna()).mean(),
            **{
                f"error/{error}": error_rate
                for error, error_rate in results_df.error.dropna().value_counts(normalize=True).items()
            },
            # Env metrics
            **{f"metrics/{metric}": metrics_df[metric].mean() for metric in metrics_df.columns},
            # Time metrics
            "time/step": step_time,
            "time/generate_completions": generate_completions_time,
            "time/teacher_logprobs": teacher_logprobs_time,
            "time/save_ckpt": save_ckpt_time,
            # Scheduler metrics
            **scheduler.get_metrics(),
            # Buffer metrics
            **buffer.get_metrics(),
            # Event loop lag metrics
            **event_loop_lag_monitor.get_metrics(),
            # W&B axis
            "step": progress.step,
        }

        # If more than one env, add per-env metrics
        if results_df.task.nunique() > 1:
            per_env_reward = results_df.groupby("task").reward.mean().to_dict()
            to_log.update({f"reward/{env}": reward for env, reward in per_env_reward.items()})

            per_env_ratio = results_df.task.value_counts(normalize=True).to_dict()
            to_log.update({f"batch/{env}": ratio for env, ratio in per_env_ratio.items()})

        # Optionally, add val metrics
        if val_results_df is not None:
            to_log.update({"val_reward/mean": val_results_df.reward.mean()})

            if val_results_df.task.nunique() > 1:
                per_env_reward = val_results_df.groupby("task").reward.mean().to_dict()
                to_log.update({f"val_reward/{env}": reward for env, reward in per_env_reward.items()})

                per_env_ratio = val_results_df.task.value_counts(normalize=True).to_dict()
                to_log.update({f"val_batch/{env}": ratio for env, ratio in per_env_ratio.items()})

        # Log metrics to monitor(s)
        monitor.log(to_log)

        # Log samples to monitor(s) if enabled
        subset_train_rollouts = random.sample(train_rollouts, min(8, len(train_rollouts)))
        monitor.log_samples(subset_train_rollouts, step=progress.step)

        # Log distributions (rewards, advantages) if enabled
        monitor.log_distributions(
            distributions={
                "rewards": rewards,
                "advantages": advantages,
            },
            step=progress.step,
        )

        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {results_df.reward.mean():.4f} |{f' Val. Reward: {val_results_df.reward.mean():.4f} |' if val_results_df is not None else ''} Throughput: {throughput:.1f} tokens/s | Seq. Length: {results_df.groupby('example_id').seq_len.mean().mean():.1f} tokens/sample | Async Level: {scheduler.async_level} | Max. Off-Policy Level: {scheduler.max_off_policy_level}"
        logger.success(step_message)

        # Increment step
        progress.step += 1
        is_first_step = False

        event_loop_lag_monitor.reset()

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

    if config.eval:
        logger.info("Running final evals (subprocess)")
        await run_evals_subprocess(
            client_config=config.client,
            eval_config=config.eval,
            model_config=config.model,
            sampling_config=config.eval.sampling,
            reasoning_field=config.eval.reasoning_field,
            output_dir=config.output_dir,
            ckpt_step=scheduler.ckpt_step,
            step=progress.step,
            max_concurrent=config.max_concurrent or -1,
        )

    # Log final (immutable) samples and distributions to monitor(s)
    monitor.log_final_samples()
    monitor.save_final_summary()

    # Write final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(progress, buffer, step=progress.step)

    # Close training batch sender
    training_batch_sender.close()

    # Stop env workers
    await scheduler.stop()

    # Cancel event loop lag monitor task
    event_loop_lag_monitor_task.cancel()

    logger.success("Orchestrator finished.")

    # Optionally, print benchmark table
    if config.bench:
        print_benchmark(to_col_format(monitor.history))


def main():
    """Main entry-point for orchestrator. Run using `uv run orchestrator`"""

    asyncio.run(orchestrate(parse_argv(OrchestratorConfig)))


if __name__ == "__main__":
    main()
