import asyncio
import gc
import os
import time

import tomli_w

from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.eval_utils import compute_eval_ckpt_step
from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.patches import monkey_patch_chat_completion_logprobs, monkey_patch_oai_iterable_types
from prime_rl.orchestrator.trajectories import (
    build_vlm_image_cache,
    interleave_rollout,
    offload_images_to_disk,
    pretokenize_rollout_trajectory,
)
from prime_rl.transport import TrainingBatch, TrainingSample, setup_training_batch_sender
from prime_rl.utils.pathing import get_log_dir, get_rollout_dir, get_step_path
from prime_rl.utils.usage_reporter import UsageReporter

# This monkey patch is necessary to avoid Pydantic validating fields using typing.Iterable (e.g. in multimodal or tool call messages) lazily which leads to tokenization errors, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1249
monkey_patch_oai_iterable_types()


# This monkey patch is necessary to avoid heavy CPU overhead from constructing the OAI ChatCompletion Pydantic model with logprobs, for more info see https://github.com/PrimeIntellect-ai/prime-rl/pull/1189
monkey_patch_chat_completion_logprobs()

# Import environment before any other imports

import pandas as pd
import verifiers as vf
from transformers import AutoProcessor

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.buffer import Buffer
from prime_rl.orchestrator.ckpt import Progress, setup_ckpt_manager
from prime_rl.orchestrator.envs import EvalEnv, EvalEnvs, TrainEnvs
from prime_rl.orchestrator.filters import apply_filters, setup_filters
from prime_rl.orchestrator.scheduler import Scheduler
from prime_rl.orchestrator.utils import (
    compute_teacher_logprobs,
    get_weight_dir,
    print_benchmark,
    set_default_executor,
    setup_external_rollout_model,
)
from prime_rl.orchestrator.vf_utils import (
    get_seq_len,
    intercept_vf_logging,
    save_rollouts,
)
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.utils.client import (
    init_nccl_broadcast,
    setup_inference_pool,
)
from prime_rl.utils.config import cli
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import (
    clean_exit,
    get_env_ids_to_install,
    install_env,
    resolve_latest_ckpt_step,
    to_col_format,
)

# Hard wall-clock budget for the orchestrator's post-training cleanup. If the
# graceful shutdown sequence (scheduler / inference pool / env teardown) is
# still running after this many seconds, we force-exit the process so the run
# pod terminates instead of sitting wedged forever. The training checkpoint
# and artifacts are persisted *before* this point, so a forced exit is safe.
SHUTDOWN_TIMEOUT_S = 300

# Maximum number of times to attempt generating a training batch when all
# rollouts are filtered out. After this many attempts, the orchestrator crashes
# rather than silently skipping training steps.
MAX_EMPTY_BATCH_ATTEMPTS = 3


@clean_exit
async def orchestrate(config: OrchestratorConfig):
    # Initialize the logger
    logger = setup_logger(
        config.log.level,
        json_logging=config.log.json_logging,
    )
    intercept_vf_logging(logger="verifiers.serve", level="WARN")  # show logs from env clients
    logger.info("Starting orchestrator")
    set_default_executor()

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
    env_ids_to_install.update(get_env_ids_to_install(config.train.env))
    if config.eval is not None:
        env_ids_to_install.update(get_env_ids_to_install(config.eval.env))

    for env_id in env_ids_to_install:
        install_env(env_id, prerelease=config.env_install_prerelease)

    # Setup rollout inference pool (handles both static and elastic modes)
    rollout_client_config, rollout_model_name, enable_policy_updates = setup_external_rollout_model(config, logger)

    train_client_type = "openai_chat_completions_token" if config.use_token_client else "openai_chat_completions"
    inference_pool = await setup_inference_pool(
        rollout_client_config,
        model_name=rollout_model_name,
        train_client_type=train_client_type,
        eval_client_type="openai_chat_completions",
    )

    # Setup teacher inference pool if configured
    if config.teacher_model:
        logger.info(
            f"Initializing teacher inference pool (base_url={', '.join(config.teacher_model.client.base_url)}, "
            f"model={config.teacher_model.model.name})"
        )
        teacher_inference_pool = await setup_inference_pool(
            config.teacher_model.client,
            model_name=config.teacher_model.model.name,
            train_client_type="openai_chat_completions",
        )
    else:
        teacher_inference_pool = None

    # Check if this is a vision-language model (used throughout for VLM-specific paths)
    is_vlm = config.model.vlm is not None

    # Load tokenizer and processor (processor only for VLM models)
    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    processor = None
    if is_vlm:
        logger.info(f"Loading VLM processor for {config.model.name}")
        processor = AutoProcessor.from_pretrained(
            config.model.name, trust_remote_code=config.model.trust_remote_code, use_fast=True
        )

    # Setup monitor (may register the run and set RUN_ID in the environment)
    logger.info(f"Initializing monitor (wandb={config.wandb}, prime_monitor={config.prime_monitor})")
    monitor = setup_monitor(
        wandb_config=config.wandb,
        prime_config=config.prime_monitor,
        output_dir=config.output_dir,
        tokenizer=tokenizer,
        run_config=config,
    )

    # Read run_id AFTER setup_monitor so that newly registered runs are captured
    run_id = os.getenv("RUN_ID", "")

    # Usage reporter requires BOTH the base URL and the API key. Activating
    # with only one set used to crash every POST inside httpx (None header
    # value), so we now gate construction on both being present and log a
    # clear warning when half-configured.
    usage_base_url = os.environ.get("PI_USAGE_BASE_URL")
    usage_api_key = os.environ.get("PI_USAGE_API_KEY")
    if usage_base_url and usage_api_key:
        usage_reporter = UsageReporter()
    else:
        if usage_base_url and not usage_api_key:
            logger.warning("PI_USAGE_BASE_URL is set but PI_USAGE_API_KEY is missing; usage reporting disabled.")
        usage_reporter = None

    # Setup heartbeat (only on rank 0, orchestrator is single process)
    heart = None
    if config.heartbeat is not None:
        logger.info("Initializing heartbeat")
        heart = Heartbeat(config.heartbeat.url)

    # Build rollout filters
    rollout_filters = setup_filters(config.filters, vocab_size=tokenizer.vocab_size)

    # Load environments
    logger.info("Loading training environments")
    train_envs = TrainEnvs(config.train.env)
    logger.info(f"Loaded {len(train_envs)} training environment(s) ({', '.join(train_envs.names)})")

    await train_envs.start(
        log_dir=get_log_dir(config.output_dir.parent) / "envs" / "train",
        log_level=config.log.vf_level,
        json_logging=config.log.json_logging,
    )
    logger.success("Train environment(s) ready")

    eval_envs: EvalEnvs | None = None
    if config.eval:
        logger.info("Loading eval environment(s)")
        eval_envs = EvalEnvs(config.eval.env)
        logger.info(f"Loaded {len(eval_envs)} eval environment(s) ({', '.join(eval_envs.names)})")

        await eval_envs.start(
            log_dir=get_log_dir(config.output_dir.parent) / "envs" / "eval",
            log_level=config.log.vf_level,
            json_logging=config.log.json_logging,
        )
        logger.success("Eval environment(s) ready")

    # Setup buffer
    logger.info(f"Setting up buffer ({config.buffer})")
    buffer = Buffer(train_envs, config.buffer)

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
        train_envs=train_envs,
        buffer=buffer,
        inference_pool=inference_pool,
        max_inflight_rollouts=config.max_inflight_rollouts,
        max_async_level=config.max_async_level,
        max_off_policy_steps=config.max_off_policy_steps,
        strict_async_level=config.strict_async_level,
        tasks_per_minute=config.tasks_per_minute,
        enable_policy_updates=enable_policy_updates,
        lora_name=config.model.lora.name if config.model.lora else None,
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

    # Start inference metrics collector (requires W&B)
    inference_metrics_collector = None
    if config.wandb is not None and config.collect_inference_metrics:
        inference_metrics_collector = InferenceMetricsCollector(inference_pool.admin_clients)
        await inference_metrics_collector.start()

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

    # Track last online eval checkpoint step per eval env
    last_eval_steps: dict[str, int] = {env.name: -1 for env in eval_envs} if eval_envs else {}
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
            last_eval_steps = {name: scheduler.ckpt_step for name in last_eval_steps}
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
    logger.info(f"Starting orchestrator loop (max_steps={config.max_steps or 'infinite'})")
    is_first_step = True

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
        # Each eval env has its own interval, so we check each independently.
        envs_to_eval: list[EvalEnv] = []
        if config.eval:
            assert eval_envs is not None
            for eval_env in eval_envs:
                eval_ckpt_step = compute_eval_ckpt_step(
                    ckpt_step=ckpt_step,
                    prev_ckpt_step=prev_ckpt_step,
                    last_eval_step=last_eval_steps[eval_env.name],
                    interval=eval_env.config.interval,
                    eval_base_model=config.eval.eval_base_model,
                )
                if eval_ckpt_step is not None:
                    last_eval_steps[eval_env.name] = ckpt_step
                    envs_to_eval.append(eval_env)

        if envs_to_eval:
            env_names = ", ".join(e.name for e in envs_to_eval)
            logger.info(f"Running evals at {ckpt_step=} for {env_names}")

            # Pause weight updates and re-scheduling of training rollouts during eval
            # to avoid evaluating across different checkpoints and avoid congestion
            scheduler.checkpoint_ready.clear()

            # For heavy eval workloads, it might be necessary additionally cancel in-flight training rollouts
            if config.eval.cancel_inflight_rollouts_on_eval:
                logger.info("Cancelling in-flight training rollouts before starting evals to avoid congestion.")
                await scheduler.cancel_inflight_rollouts()

            eval_results = await asyncio.gather(
                *[
                    eval_env.evaluate(
                        model_name=scheduler.model_name,
                        get_client=inference_pool.get_eval_client,
                        ckpt_step=ckpt_step,
                        step=progress.step,
                        cache_salt=str(ckpt_step),
                    )
                    for eval_env in envs_to_eval
                ]
            )

            # Save eval rollouts to disk (fire-and-forget background thread)
            eval_rollouts = [o for outputs in eval_results for o in outputs]
            if eval_rollouts:
                step_path = get_step_path(get_rollout_dir(config.output_dir), progress.step)
                await asyncio.to_thread(
                    save_rollouts, eval_rollouts, step_path / "eval_rollouts.jsonl", exclude_keys={"trajectory"}
                )

            # Resume weight updates
            scheduler.checkpoint_ready.set()

        # Update prev_ckpt_step for next iteration
        prev_ckpt_step = ckpt_step

        # Schedule generating the training batch. Retry on empty-after-filter
        # batches so the trainer never receives an empty batch.
        generate_completions_time = 0.0
        train_rollouts: list[vf.RolloutOutput] = []
        num_rollouts = 0
        num_unique_examples = 0
        n_trainable = 0
        for attempt in range(MAX_EMPTY_BATCH_ATTEMPTS):
            train_rollouts = await scheduler.generate_batch(step=progress.step)
            generate_completions_time += scheduler.last_batch_generation_time

            # Compute advantages (in-place)
            num_rollouts = len(train_rollouts)
            num_unique_examples = len({r["example_id"] for r in train_rollouts})
            compute_advantages(train_rollouts, config.rollouts_per_example, config.advantage)

            # Apply rollout filters — sets rollout["filters"] and rollout["is_filtered"]
            apply_filters(rollout_filters, train_rollouts)

            n_trainable = sum(1 for r in train_rollouts if not r["is_filtered"])
            if n_trainable > 0:
                break

            if attempt == MAX_EMPTY_BATCH_ATTEMPTS - 1:
                logger.error(
                    f"Attempt {attempt + 1}/{MAX_EMPTY_BATCH_ATTEMPTS} at step {progress.step} "
                    f"filtered out all {num_rollouts} rollouts - crashing orchestrator"
                )
                reason = (
                    f"All {num_rollouts} rollouts were filtered out on "
                    f"{MAX_EMPTY_BATCH_ATTEMPTS} consecutive attempts at step {progress.step}"
                )
                evicted_path = config.output_dir / "control" / "evicted.txt"
                evicted_path.parent.mkdir(parents=True, exist_ok=True)
                evicted_path.write_text(reason)
                raise RuntimeError(reason)

            logger.warning(
                f"Attempt {attempt + 1}/{MAX_EMPTY_BATCH_ATTEMPTS} at step {progress.step} "
                f"filtered out all {num_rollouts} rollouts - retrying batch generation"
            )

        trainable_ratio = n_trainable / num_rollouts
        if trainable_ratio <= 0.1:
            logger.warning(
                f"Only {n_trainable}/{num_rollouts} rollouts in the batch are trainable "
                f"({trainable_ratio:.1%}) - this can mean the tasks are too easy or too hard for the "
                "model, consider reviewing the task difficulty of your environment(s)"
            )

        # Save train rollouts to disk (fire-and-forget background thread)
        step_path = get_step_path(get_rollout_dir(config.output_dir), progress.step)
        await asyncio.to_thread(
            save_rollouts, train_rollouts, step_path / "train_rollouts.jsonl", exclude_keys={"trajectory"}
        )

        # VLM: offload base64 images to disk immediately to free memory
        if is_vlm:
            offload_start = time.perf_counter()
            num_offloaded = offload_images_to_disk(train_rollouts, config.output_dir)
            if num_offloaded:
                logger.info(
                    f"VLM offloaded {num_offloaded} unique images to disk in {time.perf_counter() - offload_start:.2f}s"
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
            vlm_cache = await asyncio.to_thread(build_vlm_image_cache, train_rollouts, processor)
            mm_token_type_ids_mapping = {}
            if hasattr(processor, "image_token_id") and processor.image_token_id is not None:
                mm_token_type_ids_mapping[processor.image_token_id] = 1
            if hasattr(processor, "video_token_id") and processor.video_token_id is not None:
                mm_token_type_ids_mapping[processor.video_token_id] = 2

            logger.info(
                f"VLM timing: extract={vlm_cache.extract_time:.2f}s, preprocess={vlm_cache.preprocess_time:.2f}s "
                f"({vlm_cache.num_unique_images} unique images from {vlm_cache.num_unique_examples} examples)"
            )
        else:
            vlm_cache = None
            mm_token_type_ids_mapping = None

        # Process rollouts in parallel
        def process_rollout(rollout: vf.RolloutOutput, rollout_idx: int) -> list[TrainingSample] | None:
            return interleave_rollout(
                rollout, vlm_cache=vlm_cache, cache_key=rollout_idx, mm_token_type_ids_mapping=mm_token_type_ids_mapping
            )

        results = await asyncio.gather(
            *(asyncio.to_thread(process_rollout, r, rollout_idx) for rollout_idx, r in enumerate(train_rollouts))
        )

        # Collect results and assign advantages. Metrics are computed over all
        # rollouts; only non-filtered samples are sent to the trainer.
        train_examples: list[TrainingSample] = []
        rollout_prefill_lens: list[int] = []
        rollout_decode_lens: list[int] = []
        rollout_samples_per_rollout: list[int] = []
        num_prefill_tokens = 0
        num_decode_tokens = 0
        for rollout, samples in zip(train_rollouts, results):
            rollout_prefill_tokens = 0
            rollout_decode_tokens = 0
            if samples is None:
                samples = []
            rollout_samples_per_rollout.append(len(samples))
            for sample in samples:
                sample.advantage = rollout["advantage"]
                sample.reward = rollout["reward"]
                sample_decode_tokens = sum(sample.completion_mask)
                sample_prefill_tokens = len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode_tokens
                rollout_decode_tokens += sample_decode_tokens
                rollout_prefill_tokens += sample_prefill_tokens
                if not rollout["is_filtered"]:
                    train_examples.append(sample)
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
                clients=teacher_inference_pool.train_clients,
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

        step_time = time.perf_counter() - step_start_time

        # Gather metrics in dataframes
        results_df = pd.DataFrame(
            {
                "example_id": [rollout["example_id"] for rollout in train_rollouts],
                "env_name": [rollout["env_name"] for rollout in train_rollouts],
                "reward": [rollout["reward"] for rollout in train_rollouts],
                "is_truncated": [rollout["is_truncated"] for rollout in train_rollouts],
                "is_filtered": [rollout["is_filtered"] for rollout in train_rollouts],
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

        # Separate DataFrames for env reward function metrics and filter flags to avoid column name collisions
        metrics_df = pd.DataFrame([rollout["metrics"] for rollout in train_rollouts])
        filter_df = pd.DataFrame([rollout["filters"] for rollout in train_rollouts])

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
            # Solve / batch metrics
            "solve_none/all": solve_none,
            "solve_all/all": solve_all,
            "effective_batch_size/all": effective_batch_size,
            **{f"batch/{env}": r for env, r in results_df.env_name.value_counts(normalize=True).items()},
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
            # Rollout filter metrics (detection rate per filter + overall drop rate)
            "filters/all/is_filtered": results_df.is_filtered.astype(float).mean(),
            **{f"filters/all/{name}": filter_df[name].astype(float).mean() for name in filter_df.columns},
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

        for env, env_df in results_df.groupby("env_name"):
            env_by_example = env_df.groupby("example_id")
            for col in per_env_columns:
                to_log[f"{col}/{env}/mean"] = env_by_example[col].mean().mean()
                to_log[f"{col}/{env}/max"] = env_by_example[col].mean().max()
                if col != "is_truncated":
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
            to_log[f"filters/{env}/is_filtered"] = env_df.is_filtered.astype(float).mean()
            env_filter_df = filter_df.loc[env_df.index]
            for name in filter_df.columns:
                to_log[f"filters/{env}/{name}"] = env_filter_df[name].astype(float).mean()

        # Log metrics to monitor(s)
        monitor.log(to_log, step=progress.step)

        # Log samples to monitor(s) if enabled.
        monitor.log_samples(train_rollouts, step=progress.step)

        # Log distributions (rewards, advantages) if enabled
        monitor.log_distributions(
            distributions={
                "rewards": [r["reward"] for r in train_rollouts],
                "advantages": [r["advantage"] for r in train_rollouts],
            },
            step=progress.step,
        )

        if usage_reporter and run_id:
            usage_reporter.report_training_usage(
                run_id=run_id,
                step=progress.step,
                tokens=num_prefill_tokens + num_decode_tokens,
            )

        reward_mean = by_example.reward.mean().mean()
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {reward_mean:.4f} | Seq. Length: {by_example.seq_len.mean().mean():.1f} tokens/sample | Async Level: {scheduler.async_level} | Max. Off-Policy Level: {scheduler.max_off_policy_level}"
        logger.success(step_message)

        # Increment step
        progress.step += 1
        is_first_step = False

        # Free large per-step objects to prevent memory accumulation
        del train_rollouts, train_examples, training_batch, vlm_cache
        del results_df, metrics_df
        gc.collect()

        event_loop_lag_monitor.reset()

        # Send heartbeat if configured
        if heart is not None:
            heart.beat()

    if config.eval and eval_envs is not None:
        logger.info("Running final evals")
        eval_results = await asyncio.gather(
            *[
                eval_env.evaluate(
                    model_name=scheduler.model_name,
                    get_client=inference_pool.get_eval_client,
                    ckpt_step=ckpt_step,
                    step=progress.step,
                    cache_salt=str(ckpt_step),
                )
                for eval_env in eval_envs
            ]
        )

        # Save final eval rollouts to disk
        eval_rollouts = [o for outputs in eval_results for o in outputs]
        if eval_rollouts:
            step_path = get_step_path(get_rollout_dir(config.output_dir), progress.step)
            await asyncio.to_thread(
                save_rollouts, eval_rollouts, step_path / "eval_rollouts.jsonl", exclude_keys={"trajectory"}
            )

    # Log final (immutable) samples and distributions to monitor(s)
    monitor.log_final_samples()
    monitor.save_final_summary()

    # Write final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(progress, buffer, step=progress.step)

    # Bounded best-effort cleanup. Each await below may block on a remote peer
    # (env-server ZMQ recv, inference admin httpx aclose, etc.). The outer
    # asyncio.wait gives the whole sequence a single deadline; if anything
    # wedges past SHUTDOWN_TIMEOUT_S we force-exit the process. Individual
    # awaits intentionally do NOT have their own timeouts — asyncio.wait_for
    # would itself hang on an uncancellable await, which is exactly the
    # failure mode we're guarding against.
    async def _graceful_shutdown() -> None:
        training_batch_sender.close()
        await scheduler.stop()
        if inference_metrics_collector is not None:
            await inference_metrics_collector.stop()
        await inference_pool.stop()
        if teacher_inference_pool is not None:
            await teacher_inference_pool.stop()
        event_loop_lag_monitor_task.cancel()
        # Shutdown env processes (also registered as atexit handler for crash safety)
        train_envs.shutdown()
        if eval_envs is not None:
            eval_envs.shutdown()

    shutdown_task = asyncio.create_task(_graceful_shutdown())
    _, pending = await asyncio.wait({shutdown_task}, timeout=SHUTDOWN_TIMEOUT_S)

    if pending:
        logger.warning(
            f"Orchestrator shutdown did not complete within {SHUTDOWN_TIMEOUT_S}s; "
            "forcing process exit. Training artifacts are already persisted."
        )
        os._exit(0)

    # asyncio.wait swallows task exceptions; re-raise so a fast cleanup
    # failure surfaces the same way as it did when each step was awaited
    # directly.
    await shutdown_task

    if usage_reporter:
        usage_reporter.close()

    logger.success("Orchestrator finished.")

    # Optionally, print benchmark table
    if config.bench:
        print_benchmark(to_col_format(monitor.history))


def main():
    """Main entry-point for orchestrator. Run using `uv run orchestrator`"""
    set_proc_title("Orchestrator")
    asyncio.run(orchestrate(cli(OrchestratorConfig)))


if __name__ == "__main__":
    main()
