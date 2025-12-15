import asyncio
import random
import time

import tomli_w

from prime_rl.orchestrator.advantage import compute_advantages
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

from prime_rl.eval.utils import run_evals
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.ckpt import Progress, setup_ckpt_manager
from prime_rl.orchestrator.config import BufferConfig, OrchestratorConfig
from prime_rl.orchestrator.scheduler import Scheduler
from prime_rl.orchestrator.utils import (
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
    setup_evals_client,
    update_weights,
)
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import setup_logger
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
from prime_rl.utils.vf import generate_batch, get_completion_len, get_is_truncated, get_prompt_len, get_seq_len


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
        env_mix_strategy=config.env_mix.strategy,
        env_mix_kwargs=dict(
            probabilities=config.env_mix.probabilities,
            stopping_strategy=config.env_mix.stopping_strategy,
            seed=config.env_mix.seed,
        ),
    )
    env.set_max_seq_len(config.seq_len)
    if config.use_token_prompts:
        logger.info("Using token prompts in environment to avoid retokenization discrepancies in multi-turn rollouts")
        env.set_use_token_prompts(config.use_token_prompts)
        if config.tokenize_method is not None:
            env.set_tokenize_method(config.tokenize_method)
        if config.exact_tokenization is not None:
            env.set_exact_tokenization(config.exact_tokenization)
    dataset = env.get_dataset(seed=config.seed)
    val_dataset = env.get_eval_dataset(seed=config.seed) if config.val else None

    # Setup buffer
    logger.info(f"Setting up buffer ({config.buffer})")
    buffer = Buffer(dataset, config.buffer)
    val_buffer = Buffer(val_dataset, BufferConfig()) if val_dataset else None

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
        lora_name=config.lora_name,
    )

    # Check health of the client
    logger.info("Waiting for inference pool to be ready")
    await check_health(admin_clients)
    await check_has_model(clients, config.model.name)
    logger.success("Inference pool ready")

    # Set up weight broadcast backend
    logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
    if config.weight_broadcast.type == "nccl":
        await init_nccl_broadcast(
            admin_clients, config.weight_broadcast.host, config.weight_broadcast.port, config.weight_broadcast.timeout
        )

    # Get checkpoint manager
    logger.info(f"Initializing checkpoint manager ({config.ckpt})")
    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)

    # Setup training batch sender for sending training examples to trainer
    logger.info(f"Initializing training batch sender ({config.rollout_transport})")
    training_batch_sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

    # Reset weights to base model if starting from scratch
    progress = Progress()

    checkpoint_step = None
    if config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None:
        if config.ckpt.resume_step == -1:
            checkpoint_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
        else:
            checkpoint_step = config.ckpt.resume_step

    if checkpoint_step is not None:
        ckpt_manager.load(progress, buffer, step=checkpoint_step)
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")
        scheduler.ckpt_step = progress.step  # Always resume from the latest checkpoint
        await update_weights(
            admin_clients,
            get_step_path(get_broadcast_dir(config.output_dir), scheduler.ckpt_step),
            lora_name=config.lora_name,
        )
        if config.lora_name is not None:
            scheduler.model_name = config.lora_name
    else:
        logger.info("Training from scratch. Resetting weights to base model")
        if config.lora_name is None:
            await reload_weights(admin_clients)

    # Iterate over dataset in batches
    max_steps = config.max_steps or int(1e9)
    logger.info(f"Starting orchestrator loop (max_steps={max_steps or 'infinite'})")
    last_eval_step = -1
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

        # Schedule generating the training batch
        generate_completions_start_time = time.perf_counter()
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
                    examples=val_problems,
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
                    evals_client=evals_client,
                    reasoning_field=config.eval.reasoning_field,
                    output_dir=config.output_dir,
                    ckpt_step=ckpt_step,
                    step=progress.step,
                )
            )
        else:
            eval_task = asyncio.create_task(asyncio.sleep(0))  # Dummy task

        # Await train rollouts, process results and write batch to disk to consume by trainer
        await train_task
        generate_completions_time = time.perf_counter() - generate_completions_start_time
        train_rollouts = train_task.result()

        # Compute advantages
        rewards = [rollout["reward"] for rollout in train_rollouts]
        completion_lens = [get_completion_len(rollout) for rollout in train_rollouts]
        advantages = compute_advantages(
            rewards,
            completion_lens,
            config.rollouts_per_example,
            config.advantage,
        )

        # Update and sample rollouts from the buffer
        make_train_example = interleave_rollout if config.trajectory_strategy == "interleaved" else branch_rollout
        train_examples: list[TrainingSample] = []
        for train_rollout, advantage in zip(train_rollouts, advantages):
            train_example = make_train_example(train_rollout)
            for te in train_example:
                te.advantage = advantage
            train_examples.extend(train_example)
        logger.debug(
            f"Converted {len(train_rollouts)} training rollouts to {len(train_examples)} training examples using {config.trajectory_strategy} strategy"
        )

        training_batch = TrainingBatch(
            examples=train_examples,
            temperature=config.sampling.temperature,
            step=progress.step,
        )
        training_batch_sender.send(training_batch)

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
                "is_truncated": [get_is_truncated(rollout) for rollout in train_rollouts],
                "completion_len": [get_completion_len(rollout) for rollout in train_rollouts],
                "prompt_len": [get_prompt_len(rollout) for rollout in train_rollouts],
                "seq_len": [get_seq_len(rollout) for rollout in train_rollouts],
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

        # Compute per-env reuslts
        num_envs_in_batch = results_df.task.nunique()
        per_env_reward = results_df.groupby("task").reward.mean().to_dict() if num_envs_in_batch > 1 else None
        per_env_count = results_df.task.value_counts().to_dict() if num_envs_in_batch > 1 else None

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
            **scheduler.get_metrics(),
            # Buffer metrics
            **buffer.get_metrics(),
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

        # Log samples to W&B table if enabled
        subset_train_rollouts = random.sample(train_rollouts, min(8, len(train_rollouts)))
        monitor.log_samples(subset_train_rollouts, step=progress.step)

        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {results_df.reward.mean():.4f} |{f' Val. Reward: {val_results_df.reward.mean():.4f} |' if val_results_df is not None else ''} Throughput: {throughput:.1f} tokens/s | Seq. Length: {results_df.groupby('example_id').seq_len.mean().mean():.1f} tokens/sample | Async Level: {scheduler.async_level} | Max. Off-Policy Level: {scheduler.max_off_policy_level}"
        logger.success(step_message)

        # Increment step
        progress.step += 1
        is_first_step = False

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

    if config.eval:
        logger.info("Running final evals")
        await run_evals(
            clients=clients,
            eval_config=config.eval,
            model_config=config.model,
            sampling_config=config.eval.sampling,
            evals_client=evals_client,
            reasoning_field=config.eval.reasoning_field,
            output_dir=config.output_dir,
            ckpt_step=scheduler.ckpt_step,
            step=progress.step,
        )

    # Log final (immutable) samples to W&B table
    monitor.log_final_samples()
    monitor.save_final_summary()

    # Write final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(progress, buffer, step=progress.step)

    # Close training batch sender
    training_batch_sender.close()

    logger.success("Orchestrator finished.")

    # Optionally, print benchmark table
    if config.bench:
        print_benchmark(to_col_format(monitor.history))


def main():
    """Main entry-point for orchestrator. Run using `uv run orchestrator`"""

    asyncio.run(orchestrate(parse_argv(OrchestratorConfig)))


if __name__ == "__main__":
    main()
