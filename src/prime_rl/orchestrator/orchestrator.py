import asyncio
import atexit
import gc
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor

import tomli_w

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.eval_utils import compute_eval_ckpt_step, get_eval_sampling_args
from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor
from prime_rl.orchestrator.patches import monkey_patch_chat_completion_logprobs, monkey_patch_oai_iterable_types
from prime_rl.orchestrator.trajectories import (
    build_vlm_image_cache,
    interleave_rollout,
    offload_images_to_disk,
    pretokenize_rollout_trajectory,
)
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.utils.pathing import get_log_dir

# This monkey patch is necessary to avoid Pydantic validating fields using typing.Iterable (e.g. in multimodal or tool call messages) lazily which leads to tokenization errors, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1249
monkey_patch_oai_iterable_types()


# This monkey patch is necessary to avoid heavy CPU overhead from constructing the OAI ChatCompletion Pydantic model with logprobs, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1189
monkey_patch_chat_completion_logprobs()

# Import environment before any other imports

import pandas as pd
import verifiers as vf
from transformers import AutoProcessor, AutoTokenizer

from prime_rl.configs.orchestrator import BufferConfig, OrchestratorConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.ckpt import Progress, setup_ckpt_manager
from prime_rl.orchestrator.eval_utils import evaluate_env
from prime_rl.orchestrator.filters import apply_filters, setup_filters
from prime_rl.orchestrator.scheduler import Scheduler
from prime_rl.orchestrator.utils import (
    compute_teacher_logprobs,
    get_sampling_args,
    get_weight_dir,
    print_benchmark,
    set_semaphore,
    setup_external_rollout_model,
)
from prime_rl.orchestrator.vf_utils import (
    generate,
    get_completion_len,
    get_seq_len,
    intercept_vf_logging,
    resolve_num_workers,
    setup_env_client,
    spawn_env_server,
    task_uses_group_scoring,
    wait_for_env_servers,
)
from prime_rl.utils.client import (
    init_nccl_broadcast,
    setup_inference_pool,
)
from prime_rl.utils.config import cli
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import (
    clean_exit,
    get_env_ids_to_install,
    install_env,
    resolve_latest_ckpt_step,
    strip_env_version,
    to_col_format,
)


@clean_exit
async def orchestrate(config: OrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "orchestrator.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )
    intercept_vf_logging(logger="verifiers.serve", level=config.log.vf_level)  # show logs from env clients
    logger.info("Starting orchestrator")

    event_loop_lag_monitor = EventLoopLagMonitor()
    event_loop_lag_monitor_task = asyncio.create_task(event_loop_lag_monitor.run())

    # Print warning if running in benchmark mode
    if config.bench:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    # Save configs to output directory
    config_dir = config.output_dir / "control"
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

    # Setup rollout inference pool (handles both static and elastic modes)
    rollout_client_config, rollout_model_name, enable_policy_updates = setup_external_rollout_model(config, logger)

    client_type = "openai_chat_completions_token" if config.use_token_client else "openai_chat_completions"
    if config.use_token_client:
        logger.warning(
            "Token-in-token-out (TITO) client is enabled. Only use this if your environment has a linear "
            "history and the chat template has the extension property."
        )
    inference_pool = await setup_inference_pool(
        rollout_client_config, model_name=rollout_model_name, client_type=client_type
    )

    # Setup teacher inference pool if configured
    if config.teacher_model:
        logger.info(
            f"Initializing teacher inference pool (base_url={', '.join(config.teacher_model.client.base_url)}, "
            f"model={config.teacher_model.model.name})"
        )
        teacher_inference_pool = await setup_inference_pool(
            config.teacher_model.client, model_name=config.teacher_model.model.name
        )
    else:
        teacher_inference_pool = None

    # Check if this is a vision-language model (used throughout for VLM-specific paths)
    is_vlm = config.model.vlm is not None

    # Load tokenizer and processor (processor only for VLM models)
    logger.info(f"Initializing tokenizer for {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=config.model.trust_remote_code)

    processor = None
    if is_vlm:
        logger.info(f"Loading VLM processor for {config.model.name}")
        processor = AutoProcessor.from_pretrained(
            config.model.name, trust_remote_code=config.model.trust_remote_code, use_fast=True
        )

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

    # Build rollout filters
    rollout_filters = setup_filters(config.filters, vocab_size=tokenizer.vocab_size)
    if rollout_filters:
        logger.info(f"Initialized {len(rollout_filters)} rollout filter(s): {[f.name for f in rollout_filters]}")

    # Load environment and extract dataset
    logger.info(
        f"Loading {len(config.env)} training environment(s) ({', '.join(env.resolved_name for env in config.env)})"
    )
    env_ids = [strip_env_version(env.id) for env in config.env]
    train_env_names = [env.resolved_name for env in config.env]
    train_env_group = vf.EnvGroup(
        envs=[vf.load_environment(env_id, **env.args) for env_id, env in zip(env_ids, config.env)],
        env_names=train_env_names,
        map_kwargs=dict(writer_batch_size=1),  # set defensively to not error on map operations on large datasets
    )
    verification_enabled = config.verification.enabled

    train_env_deferred_group_scoring_tasks = (
        {env_name for env_name in train_env_names if task_uses_group_scoring(train_env_group, env_name)}
        if verification_enabled
        else set()
    )
    for train_env_name, env_cfg in zip(train_env_names, config.env):
        env_cfg.extra_env_kwargs["score_rollouts"] = (
            verification_enabled and train_env_name not in train_env_deferred_group_scoring_tasks
        )
    if not verification_enabled:
        logger.info("Verification disabled; all training envs will skip scoring.")
    elif train_env_deferred_group_scoring_tasks:
        deferred_tasks = ", ".join(sorted(train_env_deferred_group_scoring_tasks))
        logger.info(
            f"Deferred group scoring enabled for training tasks: {deferred_tasks}. "
            "Rollouts run individually and are scored once each group completes."
        )

    train_env_addresses = []
    env_processes: list[mp.Process] = []

    def _cleanup_env_processes():
        for proc in env_processes:
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)

    atexit.register(_cleanup_env_processes)

    for env_id, env, env_name in zip(env_ids, config.env, train_env_names):
        if env.address is None:
            num_workers = resolve_num_workers(env.num_workers, config.max_inflight_rollouts)
            log_dir = (get_log_dir(config.output_dir.parent) / "envs" / "train" / env_name).as_posix()
            address, process = spawn_env_server(
                env_id=env_id,
                env_args=env.args,
                extra_env_kwargs=env.extra_env_kwargs,
                num_workers=num_workers,
                log_level=config.log.vf_level,
                log_dir=log_dir,
                json_logging=config.log.json_logging,
            )
            logger.info(f"Spawned env server for {env_name} with {num_workers} worker(s)")
            env_processes.append(process)
        else:
            if env_name in train_env_deferred_group_scoring_tasks:
                logger.warning(
                    f"Training env {env_name} uses external server at {env.address}. "
                    "Ensure that server was started with score_rollouts=False."
                )
            address = env.address
        logger.info(f"Connecting train environment {env_name} to server at {address}")
        train_env_addresses.append(address)
    train_env_clients = [
        setup_env_client(address=address, name=name) for name, address in zip(train_env_names, train_env_addresses)
    ]

    logger.info("Waiting for train environment servers to be ready")
    await wait_for_env_servers(train_env_clients)
    logger.success("Train environment servers ready")

    # this puts all train envs into server model
    # all calls to run_rollout and run_group will be routed to the server via the env client
    for env, env_client in zip(train_env_group.envs, train_env_clients):
        env.env_client = env_client

    if config.eval:
        env_ids = [strip_env_version(env.id) for env in config.eval.env]
        eval_envs = [vf.load_environment(env_id, **env.args) for env_id, env in zip(env_ids, config.eval.env)]
        eval_env_names = [env.resolved_name for env in config.eval.env]
        eval_sampling_args = get_eval_sampling_args(config.eval.sampling)
        eval_env_addresses = []

        for env_id, env, eval_env_name in zip(env_ids, config.eval.env, eval_env_names):
            if env.address is None:
                num_examples = env.num_examples or config.eval.num_examples
                rollouts_per_example = env.rollouts_per_example or config.eval.rollouts_per_example
                if num_examples == -1:
                    max_concurrent = 1024
                    logger.warning(
                        f"Eval env '{eval_env_name}' uses all examples (num_examples=-1). "
                        f"Defaulting max_concurrent={max_concurrent} for worker scaling."
                    )
                else:
                    max_concurrent = num_examples * rollouts_per_example
                num_workers = resolve_num_workers(env.num_workers, max_concurrent)
                log_dir = (get_log_dir(config.output_dir.parent) / "envs" / "eval" / eval_env_name).as_posix()
                address, process = spawn_env_server(
                    env_id=env_id,
                    env_args=env.args,
                    extra_env_kwargs=env.extra_env_kwargs,
                    num_workers=num_workers,
                    log_level=config.log.vf_level,
                    log_dir=log_dir,
                    json_logging=config.log.json_logging,
                )
                logger.info(f"Spawned eval env server for {eval_env_name} with {num_workers} worker(s)")
                env_processes.append(process)
            else:
                address = env.address
            logger.info(f"Connecting eval environment {eval_env_name} to server at {address}")
            eval_env_addresses.append(address)

        eval_env_clients = [
            setup_env_client(address=address, name=name) for name, address in zip(eval_env_names, eval_env_addresses)
        ]

        logger.info("Waiting for eval environment servers to be ready")
        await wait_for_env_servers(eval_env_clients)
        logger.success("Eval environment servers ready")

        # this puts all eval envs into server mode
        # all calls to run_rollout and run_group will be routed to the server via the env client
        for eval_env, eval_env_client in zip(eval_envs, eval_env_clients):
            eval_env.env_client = eval_env_client
    else:
        eval_envs: list[vf.Environment] = []
        eval_env_names: list[str] = []
        eval_sampling_args = {}

    # Setup buffer
    logger.info(f"Setting up buffer ({config.buffer})")
    train_dataset = train_env_group.get_dataset(seed=config.buffer.seed)
    buffer = Buffer(train_dataset, train_env_group.env_names, config.buffer)
    if config.val is not None:
        val_buffer_config = BufferConfig(env_ratios=config.buffer.env_ratios)
        val_dataset = train_env_group.get_eval_dataset(seed=val_buffer_config.seed)
        val_buffer = Buffer(val_dataset, train_env_group.env_names, val_buffer_config)
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

    scheduler = Scheduler(
        env=train_env_group,
        buffer=buffer,
        inference_pool=inference_pool,
        max_inflight_rollouts=config.max_inflight_rollouts,
        max_async_level=config.max_async_level,
        max_off_policy_steps=config.max_off_policy_steps,
        strict_async_level=config.strict_async_level,
        tasks_per_minute=config.tasks_per_minute,
        enable_policy_updates=enable_policy_updates,
        lora_name=config.model.lora.name if config.model.lora else None,
        deferred_group_scoring_tasks=train_env_deferred_group_scoring_tasks,
        config=config,
    )
    scheduler.model_name = rollout_model_name

    if checkpoint_step is not None and config.model.lora is not None and enable_policy_updates:
        assert config.model.lora.name is not None
        scheduler.model_name = config.model.lora.name

    # Check health of the inference pool
    logger.info("Waiting for inference pool to be ready")
    await inference_pool.wait_for_ready(rollout_model_name)

    logger.success("Inference pool ready")

    # Check health of teacher inference server if configured
    if config.teacher_model and teacher_inference_pool:
        logger.info("Waiting for teacher inference pool to be ready")
        await teacher_inference_pool.wait_for_ready(config.teacher_model.model.name)
        logger.success("Teacher inference pool ready")

    # Set up weight broadcast backend
    if enable_policy_updates:
        logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
        if config.weight_broadcast.type == "nccl":
            await init_nccl_broadcast(
                inference_pool.admin_clients,
                config.weight_broadcast.host,
                config.weight_broadcast.port,
                config.weight_broadcast.timeout,
                inference_world_size=config.weight_broadcast.inference_world_size,
                quantize_in_weight_transfer=config.weight_broadcast.quantize_in_weight_transfer,
            )
    else:
        logger.info("Skipping weight broadcast initialization (SFT distillation mode)")

    # Setup training batch sender for sending training examples to trainer
    logger.info(f"Initializing training batch sender ({config.rollout_transport})")
    training_batch_sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

    # Track last online eval checkpoint step for this process
    last_eval_step = -1
    # Track previous ckpt_step to detect when ckpt_step jumps over eval interval boundaries
    prev_ckpt_step = -1

    # Reset weights to base model if starting from scratch
    progress = Progress()

    if checkpoint_step is not None and ckpt_manager is not None:
        ckpt_manager.load(progress, buffer, step=checkpoint_step)
        logger.info(f"Resuming training from checkpoint step {checkpoint_step}")
        scheduler.ckpt_step = progress.step  # Always resume from the latest checkpoint
        if config.eval and config.eval.skip_eval_on_resume:
            prev_ckpt_step = scheduler.ckpt_step
            last_eval_step = scheduler.ckpt_step
            logger.info(f"Skipping online eval on resume (ckpt_step={scheduler.ckpt_step})")
        else:
            # Allow eval at resumed step by setting prev_ckpt_step one behind
            prev_ckpt_step = scheduler.ckpt_step - 1

        if enable_policy_updates:
            # In NCCL mode, skip existence check - weights are broadcasted, not stored on disk
            check_exists = config.weight_broadcast.type != "nccl"
            wait_timeout = config.ckpt.wait_for_weights_timeout if config.ckpt else None
            weights_path = get_weight_dir(
                config.output_dir, scheduler.ckpt_step, check_exists=check_exists, wait_timeout=wait_timeout
            )
            lora_name = config.model.lora.name if config.model.lora else None
            await inference_pool.update_weights(weights_path, lora_name=lora_name, step=scheduler.ckpt_step)
    else:
        logger.info("Training from scratch")

    # Iterate over dataset in batches
    max_steps = config.max_steps or int(1e9)
    logger.info(f"Starting orchestrator loop (max_steps={max_steps or 'infinite'})")
    is_first_step = True
    await set_semaphore(config.max_concurrent or -1)

    # Persistent ThreadPoolExecutor for parallel rollout processing
    rollout_executor = ThreadPoolExecutor(max_workers=64)

    while True:
        # Check if this run has been evicted by the trainer
        evicted_path = config.output_dir / "control" / "evicted.txt"
        if evicted_path.exists():
            reason = evicted_path.read_text().strip()
            raise RuntimeError(f"Run evicted by trainer: {reason}")

        # Capture ckpt_step once for consistency (it's updated inside the scheduler)
        ckpt_step = scheduler.ckpt_step if enable_policy_updates else progress.step
        scheduler.ckpt_step = ckpt_step

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

        # Run evals BEFORE training (blocking). Weight updates are paused via
        # scheduler.checkpoint_ready during eval to ensure consistent weights.
        # Use range check to handle ckpt_step jumping over interval boundaries.
        eval_ckpt_step = None
        if config.eval:
            eval_ckpt_step = compute_eval_ckpt_step(
                ckpt_step=ckpt_step,
                prev_ckpt_step=prev_ckpt_step,
                last_eval_step=last_eval_step,
                interval=config.eval.interval,
                eval_base_model=config.eval.eval_base_model,
            )

        if eval_ckpt_step is not None:
            last_eval_step = ckpt_step
            if eval_ckpt_step != ckpt_step:
                logger.info(f"Running evals for interval step {eval_ckpt_step} (current ckpt_step={ckpt_step})")
            else:
                logger.info(f"Running evals for checkpoint step {ckpt_step}")

            # Pause weight updates and re-scheduling of training rollouts during eval
            # to avoid evaluating across different checkpoints and avoid congestion
            scheduler.checkpoint_ready.clear()

            # For heavy eval workloads, it might be necessary additionally cancel in-flight training rollouts
            if config.eval.cancel_inflight_rollouts_on_eval:
                logger.info("Cancelling in-flight training rollouts before starting evals to avoid congestion.")
                await scheduler.cancel_inflight_rollouts()

            results = await asyncio.gather(
                *[
                    evaluate_env(
                        env=eval_env,
                        env_name=eval_env_name,
                        get_client=inference_pool.get_next_client,
                        model_name=scheduler.model_name,
                        sampling_args=eval_sampling_args,
                        num_examples=eval_env_config.num_examples or config.eval.num_examples,
                        rollouts_per_example=eval_env_config.rollouts_per_example or config.eval.rollouts_per_example,
                        max_retries=eval_env_config.max_retries,
                        ckpt_step=ckpt_step,
                        step=progress.step,
                    )
                    for eval_env, eval_env_name, eval_env_config in zip(eval_envs, eval_env_names, config.eval.env)
                ]
            )

            # Resume weight updates
            scheduler.checkpoint_ready.set()

        # Update prev_ckpt_step for next iteration
        prev_ckpt_step = ckpt_step

        # Schedule generating the training batch
        temperature = compute_temperature(progress.step, config.sampling, config.max_steps)
        is_vllm = config.teacher_rollout_model is None
        sampling_args = get_sampling_args(config.sampling, temperature=temperature, is_vllm=is_vllm)
        scheduler.set_sampling_args(sampling_args)
        train_task = asyncio.create_task(scheduler.generate_batch(step=progress.step))

        # Schedule running validation at the specified interval
        if val_buffer and config.val and progress.step % config.val.interval == 0:
            logger.info(f"Running validation for step {progress.step}")
            val_examples = val_buffer.sample_examples(config.val.num_examples)
            val_task = asyncio.create_task(
                generate(
                    env=train_env_group,
                    model_name=scheduler.model_name,
                    examples=val_examples,
                    rollouts_per_example=config.val.rollouts_per_example,
                    sampling_args=sampling_args,
                    clients=inference_pool.clients,
                    pbar_description="Generating rollouts (val)",
                )
            )
        else:
            val_task = asyncio.create_task(asyncio.sleep(0))  # Dummy task

        # Await train rollouts
        await train_task
        generate_completions_time = scheduler.last_batch_generation_time
        train_rollouts = train_task.result()

        # VLM: offload base64 images to disk immediately to free memory
        if is_vlm:
            offload_start = time.perf_counter()
            num_offloaded = offload_images_to_disk(train_rollouts, config.output_dir)
            if num_offloaded:
                logger.info(
                    f"VLM offloaded {num_offloaded} unique images to disk in {time.perf_counter() - offload_start:.2f}s"
                )

        # Apply rollout filters (zeros reward/mask for degenerate generations)
        filter_metrics = apply_filters(rollout_filters, train_rollouts)

        # Compute advantages
        example_ids = [r["example_id"] for r in train_rollouts]
        num_rollouts = len(train_rollouts)
        num_unique_examples = len(set(example_ids))
        rewards = [r["reward"] for r in train_rollouts]
        completion_lens = [get_completion_len(r) for r in train_rollouts]
        advantages = compute_advantages(
            rewards,
            completion_lens,
            config.rollouts_per_example,
            config.advantage,
        )

        # Convert rollouts to training samples
        parallel_preprocess_start = time.perf_counter()

        # Pretokenize before VLM image cache build (which strips image data from messages)
        for rollout in train_rollouts:
            pretokenize_rollout_trajectory(rollout, tokenizer, processor=processor)

        # VLM: build image cache in a thread so it doesn't block the event loop.
        # This lets the scheduler continue servicing inflight rollout requests
        # and — with max_async_level >= 2 — overlap with the next batch's inference.
        if is_vlm:
            vlm_cache = await asyncio.get_event_loop().run_in_executor(
                rollout_executor, build_vlm_image_cache, train_rollouts, processor
            )
            logger.info(
                f"VLM timing: extract={vlm_cache.extract_time:.2f}s, preprocess={vlm_cache.preprocess_time:.2f}s "
                f"({vlm_cache.num_unique_images} unique images from {vlm_cache.num_unique_examples} examples)"
            )
        else:
            vlm_cache = None

        # Process rollouts in parallel
        def process_rollout(rollout: vf.RolloutOutput, rollout_idx: int) -> list[TrainingSample] | None:
            return interleave_rollout(rollout, vlm_cache=vlm_cache, cache_key=rollout_idx)

        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(rollout_executor, process_rollout, r, rollout_idx)
            for rollout_idx, r in enumerate(train_rollouts)
        ]
        results = await asyncio.gather(*futures)

        # Collect results and assign advantages
        train_examples: list[TrainingSample] = []
        rollout_prefill_lens: list[int] = []
        rollout_decode_lens: list[int] = []
        rollout_samples_per_rollout: list[int] = []
        num_prefill_tokens = 0
        num_decode_tokens = 0
        for rollout, advantage, samples in zip(train_rollouts, advantages, results):
            rollout_prefill_tokens = 0
            rollout_decode_tokens = 0
            if samples is not None:
                rollout_samples_per_rollout.append(len(samples))
                for sample in samples:
                    sample.advantage = advantage
                    sample.reward = rollout["reward"]
                    sample_decode_tokens = sum(sample.completion_mask)
                    sample_prefill_tokens = len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode_tokens
                    rollout_decode_tokens += sample_decode_tokens
                    rollout_prefill_tokens += sample_prefill_tokens
                    train_examples.append(sample)
            else:
                rollout_samples_per_rollout.append(0)
            rollout_prefill_lens.append(rollout_prefill_tokens)
            rollout_decode_lens.append(rollout_decode_tokens)
            num_prefill_tokens += rollout_prefill_tokens
            num_decode_tokens += rollout_decode_tokens

        parallel_preprocess_time = time.perf_counter() - parallel_preprocess_start
        logger.debug(
            f"Converted {len(train_rollouts)} rollouts ({num_unique_examples} unique examples) "
            f"to {len(train_examples)} training examples"
        )

        # Compute teacher logprobs if teacher model is configured
        teacher_logprobs_time = 0
        if config.teacher_model and teacher_inference_pool:
            logger.info(f"Computing teacher logprobs for {len(train_examples)} training examples")
            teacher_logprobs_start_time = time.perf_counter()
            teacher_logprobs_list = await compute_teacher_logprobs(
                clients=teacher_inference_pool.clients,
                model_name=config.teacher_model.model.name,
                samples=train_examples,
            )
            for train_example, teacher_logprobs in zip(train_examples, teacher_logprobs_list):
                train_example.teacher_logprobs = teacher_logprobs
            teacher_logprobs_time = time.perf_counter() - teacher_logprobs_start_time
            logger.debug(f"Computed teacher logprobs in {teacher_logprobs_time:.2f}s")

        training_batch = TrainingBatch(
            examples=train_examples,
            step=progress.step,
        )

        training_batch_sender.send(training_batch)

        # Await and process val results
        await val_task
        val_outputs = val_task.result()

        step_time = time.perf_counter() - step_start_time

        # Gather metrics in dataframes
        results_df = pd.DataFrame(
            {
                "example_id": [rollout["example_id"] for rollout in train_rollouts],
                "task": [rollout["task"] for rollout in train_rollouts],
                "reward": [rollout["reward"] for rollout in train_rollouts],
                "is_truncated": [rollout["is_truncated"] for rollout in train_rollouts],
                "stop_condition": [rollout.get("stop_condition") for rollout in train_rollouts],
                "seq_len": [get_seq_len(rollout) for rollout in train_rollouts],
                "prefill_len": rollout_prefill_lens,
                "decode_len": rollout_decode_lens,
                "samples_per_rollout": rollout_samples_per_rollout,
                "num_turns": [len(rollout["trajectory"]) for rollout in train_rollouts],
                "generation_ms": [rollout["timing"]["generation_ms"] for rollout in train_rollouts],
                "scoring_ms": [rollout["timing"]["scoring_ms"] for rollout in train_rollouts],
            }
        )

        # Separate DataFrame for env reward function metrics to avoid column name collisions
        metrics_df = pd.DataFrame([rollout["metrics"] for rollout in train_rollouts])

        val_results_df = (
            pd.DataFrame(
                {
                    "example_id": [rollout["example_id"] for rollout in val_outputs],
                    "task": [rollout["task"] for rollout in val_outputs],
                    "reward": [rollout["reward"] for rollout in val_outputs],
                }
            )
            if val_outputs is not None
            else None
        )

        # Update progress metrics
        num_tokens = int(results_df.seq_len.sum())
        progress.total_tokens += num_tokens
        progress.total_samples += num_rollouts
        progress.total_problems += num_unique_examples

        def compute_solve_rates(df):
            """Compute solve_none, solve_all, effective_batch_size for a set of rollouts."""
            reward_per_problem = df.groupby("example_id").reward.sum()
            solve_none = (reward_per_problem == 0).mean()
            solve_all = (reward_per_problem == config.rollouts_per_example).mean()
            return solve_none, solve_all, 1 - solve_none - solve_all

        # Group by example_id to average across rollouts within each problem
        by_example = results_df.groupby("example_id")

        solve_none, solve_all, effective_batch_size = compute_solve_rates(results_df)
        to_log = {
            # Progress metrics
            "progress/tokens": num_tokens,
            "progress/prefill_tokens": num_prefill_tokens,
            "progress/decode_tokens": num_decode_tokens,
            "progress/samples": num_rollouts,
            "progress/problems": num_unique_examples,
            "progress/total_tokens": progress.total_tokens,
            "progress/total_samples": progress.total_samples,
            "progress/total_problems": progress.total_problems,
            "progress/ckpt_step": ckpt_step,  # Shared W&B axis
            # Sequence length metrics
            "seq_len/all/mean": by_example.seq_len.mean().mean(),
            "seq_len/all/max": by_example.seq_len.mean().max(),
            "seq_len/all/min": by_example.seq_len.mean().min(),
            "prefill_len/all/mean": by_example.prefill_len.mean().mean(),
            "prefill_len/all/max": by_example.prefill_len.mean().max(),
            "prefill_len/all/min": by_example.prefill_len.mean().min(),
            "decode_len/all/mean": by_example.decode_len.mean().mean(),
            "decode_len/all/max": by_example.decode_len.mean().max(),
            "decode_len/all/min": by_example.decode_len.mean().min(),
            "is_truncated/all/mean": by_example.is_truncated.mean().mean(),
            "is_truncated/all/max": by_example.is_truncated.mean().max(),
            "is_truncated/all/min": by_example.is_truncated.mean().min(),
            "stop_condition/all/generation_truncated": (
                results_df.is_truncated & (results_df.stop_condition != "prompt_too_long")
            ).mean(),
            **{
                f"stop_condition/all/{sc}": rate
                for sc, rate in results_df.stop_condition.dropna().value_counts(normalize=True).items()
            },
            "samples_per_rollout/all/mean": by_example.samples_per_rollout.mean().mean(),
            "samples_per_rollout/all/max": by_example.samples_per_rollout.mean().max(),
            "samples_per_rollout/all/min": by_example.samples_per_rollout.mean().min(),
            "num_turns/all/mean": by_example.num_turns.mean().mean(),
            "num_turns/all/max": by_example.num_turns.mean().max(),
            "num_turns/all/min": by_example.num_turns.mean().min(),
            "generation_ms/all/mean": by_example.generation_ms.mean().mean(),
            "generation_ms/all/max": by_example.generation_ms.mean().max(),
            "generation_ms/all/min": by_example.generation_ms.mean().min(),
            "scoring_ms/all/mean": by_example.scoring_ms.mean().mean(),
            "scoring_ms/all/max": by_example.scoring_ms.mean().max(),
            "scoring_ms/all/min": by_example.scoring_ms.mean().min(),
            # Train reward
            "reward/all/mean": by_example.reward.mean().mean(),
            "reward/all/max": by_example.reward.mean().max(),
            "reward/all/min": by_example.reward.mean().min(),
            "sampling/temperature": temperature,
            # Solve / batch metrics
            "solve_none/all": solve_none,
            "solve_all/all": solve_all,
            "effective_batch_size/all": effective_batch_size,
            **{f"batch/{env}": r for env, r in results_df.task.value_counts(normalize=True).items()},
            # Time metrics
            "time/step": step_time,
            "time/generate_completions": generate_completions_time,
            "time/teacher_logprobs": teacher_logprobs_time,
            "time/save_ckpt": save_ckpt_time,
            "time/parallel_preprocess": parallel_preprocess_time,
            # Scheduler metrics
            **scheduler.get_metrics(),
            # Buffer metrics
            **buffer.get_metrics(),
            # Event loop lag metrics
            **event_loop_lag_monitor.get_metrics(),
            # Rollout filter metrics
            **filter_metrics,
            # W&B axis
            "step": progress.step,
        }

        # Per-env metrics
        per_env_columns = [
            "seq_len",
            "prefill_len",
            "decode_len",
            "is_truncated",
            "samples_per_rollout",
            "num_turns",
            "generation_ms",
            "scoring_ms",
        ]

        for env, env_df in results_df.groupby("task"):
            env_by_example = env_df.groupby("example_id")
            for col in per_env_columns:
                to_log[f"{col}/{env}/mean"] = env_by_example[col].mean().mean()
                to_log[f"{col}/{env}/max"] = env_by_example[col].mean().max()
                to_log[f"{col}/{env}/min"] = env_by_example[col].mean().min()
            to_log[f"reward/{env}/mean"] = env_by_example.reward.mean().mean()
            to_log[f"reward/{env}/max"] = env_by_example.reward.mean().max()
            to_log[f"reward/{env}/min"] = env_by_example.reward.mean().min()
            solve_none, solve_all, effective_batch_size = compute_solve_rates(env_df)
            to_log[f"solve_none/{env}"] = solve_none
            to_log[f"solve_all/{env}"] = solve_all
            to_log[f"effective_batch_size/{env}"] = effective_batch_size
            to_log[f"stop_condition/{env}/generation_truncated"] = (
                env_df.is_truncated & (env_df.stop_condition != "prompt_too_long")
            ).mean()
            for sc, rate in env_df.stop_condition.dropna().value_counts(normalize=True).items():
                to_log[f"stop_condition/{env}/{sc}"] = rate
            env_metrics_df = metrics_df.loc[env_df.index]
            for metric in metrics_df.columns:
                to_log[f"metrics/{env}/{metric}"] = env_metrics_df.groupby(env_df["example_id"])[metric].mean().mean()

        # Optionally, add val metrics
        if val_results_df is not None:
            val_by_example = val_results_df.groupby("example_id")
            to_log["val/reward/all/mean"] = val_by_example.reward.mean().mean()
            to_log["val/reward/all/max"] = val_by_example.reward.mean().max()
            to_log["val/reward/all/min"] = val_by_example.reward.mean().min()
            for env, env_df in val_results_df.groupby("task"):
                env_by_example = env_df.groupby("example_id")
                to_log[f"val/reward/{env}/mean"] = env_by_example.reward.mean().mean()
                to_log[f"val/reward/{env}/max"] = env_by_example.reward.mean().max()
                to_log[f"val/reward/{env}/min"] = env_by_example.reward.mean().min()

        # Log metrics to monitor(s)
        monitor.log(to_log, step=progress.step)

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

        reward_mean = by_example.reward.mean().mean()
        val_reward_str = ""
        if val_results_df is not None:
            val_reward_mean = val_results_df.groupby("example_id").reward.mean().mean()
            val_reward_str = f" Val. Reward: {val_reward_mean:.4f} |"
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {reward_mean:.4f} |{val_reward_str} Seq. Length: {by_example.seq_len.mean().mean():.1f} tokens/sample | Async Level: {scheduler.async_level} | Max. Off-Policy Level: {scheduler.max_off_policy_level}"
        logger.success(step_message)

        # Increment step
        progress.step += 1
        is_first_step = False

        # Free large per-step objects to prevent memory accumulation
        del train_rollouts, train_examples, training_batch, vlm_cache
        del results_df, metrics_df, val_results_df
        gc.collect()

        event_loop_lag_monitor.reset()

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

    if config.eval:
        logger.info("Running final evals")
        results = await asyncio.gather(
            *[
                evaluate_env(
                    env=eval_env,
                    env_name=eval_env_name,
                    get_client=inference_pool.get_next_client,
                    model_name=scheduler.model_name,
                    sampling_args=eval_sampling_args,
                    num_examples=eval_env_config.num_examples or config.eval.num_examples,
                    rollouts_per_example=eval_env_config.rollouts_per_example or config.eval.rollouts_per_example,
                    max_retries=eval_env_config.max_retries,
                    ckpt_step=ckpt_step,
                    step=progress.step,
                )
                for eval_env, eval_env_name, eval_env_config in zip(eval_envs, eval_env_names, config.eval.env)
            ]
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

    # Shutdown rollout executor
    rollout_executor.shutdown(wait=False)

    # Stop scheduler
    await scheduler.stop()

    # Stop inference pool
    await inference_pool.stop()

    if teacher_inference_pool is not None:
        await teacher_inference_pool.stop()

    # Cancel event loop lag monitor task
    event_loop_lag_monitor_task.cancel()

    # Shutdown env processes (also registered as atexit handler for crash safety)
    atexit.unregister(_cleanup_env_processes)
    _cleanup_env_processes()

    logger.success("Orchestrator finished.")

    # Optionally, print benchmark table
    if config.bench:
        print_benchmark(to_col_format(monitor.history))


def main():
    """Main entry-point for orchestrator. Run using `uv run orchestrator`"""

    asyncio.run(orchestrate(cli(OrchestratorConfig)))


if __name__ == "__main__":
    main()
