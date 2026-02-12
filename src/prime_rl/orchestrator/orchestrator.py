import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor

import tomli_w

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.eval_utils import get_eval_sampling_args
from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor
from prime_rl.orchestrator.patches import monkey_patch_chat_completion_logprobs, monkey_patch_oai_iterable_types
from prime_rl.orchestrator.trajectories import build_vlm_image_cache, interleave_rollout
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.utils.pathing import get_log_dir

# This monkey patch is necessary to avoid Pydantic validating fields using typing.Iterable (e.g. in multimodal or tool call messages) lazily which leads to tokenization errors, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1249
monkey_patch_oai_iterable_types()


# This monkey patch is necessary to avoid heavy CPU overhead from constructing the OAI ChatCompletion Pydantic model with logprobs, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1189
monkey_patch_chat_completion_logprobs()

# Import environment before any other imports

import pandas as pd
import verifiers as vf
from loguru import logger
from transformers import AutoProcessor, AutoTokenizer

from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.ckpt import Progress, setup_ckpt_manager
from prime_rl.orchestrator.config import BufferConfig, OrchestratorConfig
from prime_rl.orchestrator.eval_utils import evaluate_env
from prime_rl.orchestrator.scheduler import Scheduler
from prime_rl.orchestrator.utils import (
    compute_teacher_logprobs,
    get_sampling_args,
    get_weight_dir,
    print_benchmark,
    set_semaphore,
)
from prime_rl.orchestrator.vf_utils import (
    generate,
    get_completion_len,
    get_prompt_len,
    get_seq_len,
    setup_env_client,
    spawn_env_server,
    wait_for_env_servers,
)
from prime_rl.utils.client import (
    init_nccl_broadcast,
    reload_weights,
    setup_inference_pool,
)
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.temp_scheduling import compute_temperature
from prime_rl.utils.utils import (
    clean_exit,
    count_chinese_chars,
    get_env_ids_to_install,
    install_env,
    resolve_latest_ckpt_step,
    strip_env_version,
    to_col_format,
)
from prime_rl.utils.vlm import is_vlm_model


@clean_exit
@logger.catch(reraise=True)
async def orchestrate(config: OrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "orchestrator.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )
    vf.setup_logging(level="CRITICAL")
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

    # Setup inference pool (handles both static and elastic modes)
    inference_pool = await setup_inference_pool(config.client, model_name=config.model.name)

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
    is_vlm = is_vlm_model(config.model.name)

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

    # Load environment and extract dataset
    logger.info(
        f"Loading {len(config.env)} training environment(s) ({', '.join(env.name or env.id for env in config.env)})"
    )
    env_ids = [strip_env_version(env.id) for env in config.env]
    train_env_group = vf.EnvGroup(
        envs=[vf.load_environment(env_id, **env.args) for env_id, env in zip(env_ids, config.env)],
        env_names=[env.name or env_id for env_id, env in zip(env_ids, config.env)],
        map_kwargs=dict(writer_batch_size=1),  # set defensively to not error on map operations on large datasets
    )

    train_env_addresses = []
    for env_id, env, env_name in zip(env_ids, config.env, train_env_group.env_names):
        if env.address is None:
            address = spawn_env_server(
                env_id=env_id,
                env_args=env.args,
                extra_env_kwargs=env.extra_env_kwargs,
                log_level="CRITICAL",
                log_file=(get_log_dir(config.output_dir) / "train" / f"{env_name}.log").as_posix(),
                log_file_level=config.log.vf_level,
            )
        else:
            address = env.address
        logger.info(f"Connecting train environment {env_name} to server at {address}")
        train_env_addresses.append(address)
    train_env_clients = [setup_env_client(address) for address in train_env_addresses]

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
        eval_env_names = [env.name or env_id for env_id, env in zip(env_ids, config.eval.env)]
        eval_sampling_args = get_eval_sampling_args(config.eval.sampling)
        eval_env_addresses = []

        for env_id, env, eval_env_name in zip(env_ids, config.eval.env, eval_env_names):
            if env.address is None:
                address = spawn_env_server(
                    env_id=env_id,
                    env_args=env.args,
                    extra_env_kwargs=env.extra_env_kwargs,
                    log_level="CRITICAL",
                    log_file=(get_log_dir(config.output_dir) / "eval" / f"{eval_env_name}.log").as_posix(),
                    log_file_level=config.log.vf_level,
                )
            else:
                address = env.address
            logger.info(f"Connecting eval environment {eval_env_name} to server at {address}")
            eval_env_addresses.append(address)

        eval_env_clients = [setup_env_client(address) for address in eval_env_addresses]

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
        oversampling_factor=config.oversampling_factor,
        max_async_level=config.max_async_level,
        max_off_policy_steps=config.max_off_policy_steps,
        strict_async_level=config.strict_async_level,
        tasks_per_minute=config.tasks_per_minute,
        lora_name=config.model.lora.name if config.model.lora else None,
        output_dir=config.output_dir,
        config=config,
    )

    if checkpoint_step is not None and config.model.lora is not None:
        assert config.model.lora.name is not None
        scheduler.model_name = config.model.lora.name

    # Check health of the inference pool
    logger.info("Waiting for inference pool to be ready")
    await inference_pool.wait_for_ready(config.model.name)
    logger.success("Inference pool ready")

    # Check health of teacher inference server if configured
    if config.teacher_model and teacher_inference_pool:
        logger.info("Waiting for teacher inference pool to be ready")
        await teacher_inference_pool.wait_for_ready(config.teacher_model.model.name)
        logger.success("Teacher inference pool ready")

    # Set up weight broadcast backend
    logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
    if config.weight_broadcast.type == "nccl":
        await init_nccl_broadcast(
            inference_pool.admin_clients,
            config.weight_broadcast.host,
            config.weight_broadcast.port,
            config.weight_broadcast.timeout,
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

        # In NCCL mode, skip existence check - weights are broadcasted, not stored on disk
        check_exists = config.weight_broadcast.type != "nccl"
        wait_timeout = config.ckpt.wait_for_weights_timeout if config.ckpt else None
        weights_path = get_weight_dir(
            config.output_dir, scheduler.ckpt_step, check_exists=check_exists, wait_timeout=wait_timeout
        )
        lora_name = config.model.lora.name if config.model.lora else None
        await inference_pool.update_weights(weights_path, lora_name=lora_name, step=scheduler.ckpt_step)
    else:
        if config.reload_weights_on_start:
            if config.model.lora is None:
                logger.info("Training from scratch. Resetting weights to base model")
                await reload_weights(inference_pool.admin_clients)
            else:
                logger.info("Training from scratch. Skipping base weight reload because LoRA is enabled")
        else:
            logger.info("Training from scratch. Skipping base weight reload")

    # Iterate over dataset in batches
    max_steps = config.max_steps or int(1e9)
    logger.info(f"Starting orchestrator loop (max_steps={max_steps or 'infinite'})")
    is_first_step = True
    await set_semaphore(config.max_concurrent or -1)

    # Start update policy loop
    update_policy_task = asyncio.create_task(scheduler.update_policy_loop())

    # Track consecutive empty batches for retry logic
    empty_batch_retries = 0
    max_empty_batch_retries = 5

    # Persistent ThreadPoolExecutor for parallel rollout processing
    rollout_executor = ThreadPoolExecutor(max_workers=64)

    while True:
        # Check if this run has been evicted by the trainer
        evicted_path = config.output_dir / "control" / "evicted.txt"
        if evicted_path.exists():
            reason = evicted_path.read_text().strip()
            raise RuntimeError(f"Run evicted by trainer: {reason}")

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
            logger.info(f"Running evals for checkpoint step {ckpt_step}")

            # Pause weight updates during eval and cancel inflight rollouts
            # this avoid doing eval across different checkpoints and avoid congestion
            scheduler.checkpoint_ready.clear()
            scheduler.cancel_all_inflight_rollouts()

            results = await asyncio.gather(
                *[
                    evaluate_env(
                        env=eval_env,
                        env_name=eval_env_name,
                        clients=inference_pool.clients,
                        model_name=scheduler.model_name,
                        sampling_args=eval_sampling_args,
                        num_examples=eval_env_config.num_examples or config.eval.num_examples,
                        rollouts_per_example=eval_env_config.rollouts_per_example or config.eval.rollouts_per_example,
                        ckpt_step=ckpt_step,
                        step=progress.step,
                    )
                    for eval_env, eval_env_name, eval_env_config in zip(eval_envs, eval_env_names, config.eval.env)
                ]
            )

            # Resume weight updates
            scheduler.checkpoint_ready.set()

        # Schedule generating the training batch
        temperature = compute_temperature(progress.step, config.sampling, config.max_steps)
        sampling_args = get_sampling_args(config.sampling, temperature=temperature)
        scheduler.set_sampling_args(sampling_args)
        train_task = asyncio.create_task(scheduler.generate_batch(step=progress.step))

        # Schedule running validation at the specified interval
        if val_buffer and config.val and progress.step % config.val.interval == 0:
            logger.info(f"Running validation for step {progress.step}")
            val_examples = val_buffer.sample_examples(config.val.num_examples)
            val_task = asyncio.create_task(
                generate(
                    env=train_env_group,
                    clients=inference_pool.clients,
                    model_name=scheduler.model_name,
                    examples=val_examples,
                    rollouts_per_example=config.val.rollouts_per_example,
                    sampling_args=sampling_args,
                    pbar_description="Generating rollouts (val)",
                )
            )
        else:
            val_task = asyncio.create_task(asyncio.sleep(0))  # Dummy task

        # Await train rollouts, process results and write batch to disk to consume by trainer
        await train_task
        generate_completions_time = scheduler.last_batch_generation_time
        train_rollouts = train_task.result()

        # Compute advantages
        example_ids = [r["example_id"] for r in train_rollouts]
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
        num_unique_examples = len(set(example_ids))

        # VLM: build image cache for efficient batched preprocessing
        if is_vlm:
            vlm_cache = build_vlm_image_cache(train_rollouts, processor)
            logger.info(
                f"VLM timing: extract={vlm_cache.extract_time:.2f}s, preprocess={vlm_cache.preprocess_time:.2f}s"
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
        for rollout, advantage, samples in zip(train_rollouts, advantages, results):
            if samples is not None:
                for sample in samples:
                    sample.advantage = advantage
                    sample.reward = rollout["reward"]
                train_examples.extend(samples)

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

        # Retry with exponential backoff if batch is empty (e.g., inference temporarily unavailable)
        if len(training_batch.examples) == 0:
            empty_batch_retries += 1
            if empty_batch_retries >= max_empty_batch_retries:
                raise RuntimeError(
                    f"Step {progress.step} failed after {max_empty_batch_retries} consecutive empty batches"
                )
            backoff = min(30 * (2 ** (empty_batch_retries - 1)), 300)  # 30s, 60s, 120s, 240s, 300s cap
            logger.warning(
                f"Step {progress.step} produced 0 training samples "
                f"(attempt {empty_batch_retries}/{max_empty_batch_retries}). Retrying in {backoff}s..."
            )
            # Cancel validation task to avoid accumulating background tasks
            val_task.cancel()
            await asyncio.sleep(backoff)
            continue

        # Reset retry counter on successful batch
        empty_batch_retries = 0
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

        # Count Chinese characters in completions
        chinese_stats = []
        for rollout in train_rollouts:
            trajectory = rollout["trajectory"]
            if not trajectory:
                continue
            last_step = trajectory[-1]
            tokens = last_step["tokens"]
            completion_text = tokenizer.decode(tokens["completion_ids"])
            chinese_count, total_count = count_chinese_chars(completion_text)
            chinese_stats.append(
                {"chinese_chars": chinese_count, "total_chars": total_count, "has_chinese": chinese_count > 0}
            )
        chinese_df = pd.DataFrame(chinese_stats)

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
            "sampling/temperature": temperature,
            # Batch metrics
            "batch/solve_none": solve_none,
            "batch/solve_all": solve_all,
            "batch/effective_batch_size": effective_batch_size,
            # Error metrics
            "error/mean": (~results_df.error.isna()).mean(),
            **{
                f"error/{error}": error_rate
                for error, error_rate in results_df.error.dropna()
                .apply(lambda e: e.get("error") if isinstance(e, dict) else e)
                .value_counts(normalize=True)
                .items()
            },
            # Env metrics
            **{f"metrics/{metric}": metrics_df[metric].mean() for metric in metrics_df.columns},
            # Chinese character metrics (cast to native Python types for JSON serialization)
            "chinese/char_count": int(chinese_df.chinese_chars.sum()),
            "chinese/char_ratio": (
                float(chinese_df.chinese_chars.sum() / chinese_df.total_chars.sum())
                if chinese_df.total_chars.sum() > 0
                else 0.0
            ),
            "chinese/rollout_ratio": float(chinese_df.has_chinese.mean()),
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
        logger.info("Running final evals")
        results = await asyncio.gather(
            *[
                evaluate_env(
                    env=eval_env,
                    env_name=eval_env_name,
                    clients=inference_pool.clients,
                    model_name=scheduler.model_name,
                    sampling_args=eval_sampling_args,
                    num_examples=eval_env_config.num_examples or config.eval.num_examples,
                    rollouts_per_example=eval_env_config.rollouts_per_example or config.eval.rollouts_per_example,
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
    scheduler.cancel_all_inflight_rollouts()
    update_policy_task.cancel()

    # Stop inference pool
    await inference_pool.stop()

    if teacher_inference_pool is not None:
        await teacher_inference_pool.stop()

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
