import asyncio
import gc
import json
import os
import time
from pathlib import Path

import tomli_w

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive import
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.eval_utils import compute_eval_ckpt_step
from prime_rl.orchestrator.event_loop_lag import EventLoopLagMonitor
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.member_generation import is_trainable_member as is_bound_trainable_member
from prime_rl.orchestrator.multi_agent_advantage import (
    RAEState,
    compute_rae_advantages,
    fan_out_for_multi_agent,
)
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
from renderers.base import create_renderer
from transformers import AutoProcessor

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.metrics.debate import write_step_metrics as write_debate_step_metrics
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


def _write_scalar_metrics(path: Path, metrics: dict[str, object]) -> None:
    normalized = {}
    for key, value in metrics.items():
        try:
            normalized[key] = float(value)
        except (TypeError, ValueError):
            continue
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(normalized, f, indent=2, sort_keys=True)


async def _persist_rollouts_and_metrics(
    rollouts: list[vf.RolloutOutput],
    step_path: Path,
    kind: str,  # "train" or "eval"
    *,
    step: int,
    dump_trajectory: bool,
    monitor,
) -> None:
    """Fire-and-forget disk write for rollouts + tier-2 debate metrics.

    Bundles the two side-effects that every save-site performs: the
    full ``*_rollouts.jsonl`` (trajectory optionally included) and the
    sidecar ``*_debate_metrics.json`` scalar aggregate. Both run on
    background threads so the main orchestration loop isn't blocked.
    """
    await asyncio.to_thread(
        save_rollouts,
        rollouts,
        step_path / f"{kind}_rollouts.jsonl",
        exclude_keys=None if dump_trajectory else {"trajectory"},
    )
    await asyncio.to_thread(
        write_debate_step_metrics,
        rollouts,
        path=step_path / f"{kind}_debate_metrics.json",
        step=step,
        monitor=monitor,
        prefix=f"debate_{kind}",
    )


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

    renderer, inference_pool = await setup_rollout_inference_pool(
        config=config,
        rollout_client_config=rollout_client_config,
        rollout_model_name=rollout_model_name,
        tokenizer=tokenizer,
        logger=logger,
    )

    # Setup monitor (may register the run and set RUN_ID in the environment)
    logger.info(f"Initializing monitor (wandb={config.wandb}, prime_monitor={config.prime_monitor})")
    monitor = setup_monitor(
        wandb_config=config.wandb,
        prime_config=config.prime_monitor,
        output_dir=config.output_dir,
        tokenizer=tokenizer,
        run_config=config,
        keep_full_history=config.bench,
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

    ma_env_names = train_envs.multi_agent_names
    non_ma_env_names = set(train_envs.names) - ma_env_names
    if ma_env_names and non_ma_env_names:
        raise NotImplementedError(
            f"Mixed multi-agent and single-agent envs in the same group are not supported. "
            f"MA: {sorted(ma_env_names)}, single-agent: {sorted(non_ma_env_names)}. "
            "Run them in separate orchestrator processes."
        )
    is_ma = bool(ma_env_names)

    advantage_type = config.advantage.type if config.advantage else None
    if is_ma and advantage_type != "ema_per_member":
        raise ValueError(
            f"advantage.type={advantage_type!r} is not supported for "
            "multi-agent envs; fan-out interleaves members per episode so "
            "compute_advantages' samples_per_problem reshape would mix "
            "seats/episodes and silently corrupt gradients. Use "
            "type='ema_per_member' for multi-agent training."
        )
    if (not is_ma) and advantage_type == "ema_per_member":
        raise ValueError(
            "advantage.type='ema_per_member' requires a multi-agent env. "
            "Use type='default' or type='custom' for single-agent training."
        )

    if is_ma:
        logger.info(
            f"Multi-agent envs={sorted(ma_env_names)}. Runtime member generation={config.multi_agent}; "
            f"advantage estimator={advantage_type}."
        )
        if is_vlm:
            raise NotImplementedError(
                "VLM + multi-agent training is not yet supported; image cache fan-out "
                "across per-member rollouts is unimplemented."
            )

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

    async def get_eval_client_config() -> vf.ClientConfig:
        return await inference_pool.get_eval_client()

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

    # Reset weights to base model if starting from scratch. RAE is stateful and
    # must resume with progress/buffer for multi-agent runs.
    progress = Progress()
    advantage_state: RAEState | None = (
        RAEState(momentum=config.advantage.momentum) if advantage_type == "ema_per_member" else None
    )

    if checkpoint_step is not None and ckpt_manager is not None:
        ckpt_manager.load(progress, buffer, step=checkpoint_step, rae_state=advantage_state)
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
            ckpt_manager.save(progress, buffer, step=progress.step, rae_state=advantage_state)
            save_ckpt_time = time.perf_counter() - save_ckpt_start_time

        # Break if we have reached the maximum number of steps
        if config.max_steps and progress.step >= config.max_steps:
            break

        logger.info(f"Starting orchestrator step {progress.step}")
        step_start_time = time.perf_counter()

        # Bring scheduler.ckpt_step up to the checkpoint required for this
        # step before deciding whether eval is due. Otherwise checkpoint-bound
        # evals can be noticed one loop late, after the next train batch has
        # already filled the inference queue.
        await scheduler.sync_policy_for_step(progress.step)
        ckpt_step = scheduler.ckpt_step if enable_policy_updates else progress.step

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

            # Pause policy-update polling during eval so a newly saved trainer
            # checkpoint cannot swap weights mid-evaluation.
            await scheduler.pause_policy_updates()

            # Pause re-scheduling of training rollouts during eval to avoid
            # congestion.
            scheduler.checkpoint_ready.clear()

            # For heavy eval workloads, it might be necessary additionally cancel in-flight training rollouts
            if config.eval.cancel_inflight_rollouts_on_eval:
                logger.info("Cancelling in-flight training rollouts before starting evals to avoid congestion.")
                await scheduler.cancel_inflight_rollouts()

            eval_results = await asyncio.gather(
                *[
                    eval_env.evaluate(
                        model_name=scheduler.model_name,
                        get_client=get_eval_client_config,
                        ckpt_step=ckpt_step,
                        step=progress.step,
                        cache_salt=str(ckpt_step),
                        multi_agent=config.multi_agent if eval_env.is_multi_agent else None,
                        eval_clients=inference_pool.eval_clients,
                    )
                    for eval_env in envs_to_eval
                ]
            )

            # Save eval rollouts to disk (fire-and-forget background thread)
            eval_rollouts = [o for outputs in eval_results for o in outputs]
            if eval_rollouts:
                step_path = get_step_path(get_rollout_dir(config.output_dir), progress.step)
                await _persist_rollouts_and_metrics(
                    eval_rollouts,
                    step_path,
                    "eval",
                    step=progress.step,
                    dump_trajectory=config.dump_trajectory,
                    monitor=monitor,
                )

            # Resume weight updates
            scheduler.checkpoint_ready.set()

        # Update prev_ckpt_step for next iteration
        prev_ckpt_step = ckpt_step

        # Schedule generating the training batch. With online difficulty
        # filtering enabled, zero-std groups are removed before buffer
        # admission; retry here only if later rollout-level filters remove
        # every training unit in a generated batch.
        generate_completions_time = 0.0

        def prepare_candidate_batch(
            candidate_rollouts: list[vf.RolloutOutput],
        ) -> tuple[list[vf.RolloutOutput], list[list[int]]]:
            num_candidate_rollouts = len(candidate_rollouts)
            if is_ma:
                candidate_training_units, candidate_rollout_to_unit_idxs = fan_out_for_multi_agent(
                    candidate_rollouts,
                    is_trainable_member=lambda rollout, member_id: is_bound_trainable_member(
                        config.multi_agent, rollout, member_id
                    ),
                )
                assert advantage_state is not None  # gated by MA validation above
                advantages = compute_rae_advantages(candidate_training_units, advantage_state)
                for unit, advantage in zip(candidate_training_units, advantages):
                    unit["advantage"] = advantage

                apply_filters(rollout_filters, candidate_training_units)
                for rollout, unit_idxs in zip(candidate_rollouts, candidate_rollout_to_unit_idxs):
                    unit_filters = [candidate_training_units[i]["filters"] for i in unit_idxs]
                    rollout["filters"] = {
                        name: any(flags.get(name, False) for flags in unit_filters)
                        for name in (unit_filters[0].keys() if unit_filters else [f.name for f in rollout_filters])
                    }
                    rollout["is_filtered"] = bool(unit_idxs) and all(
                        candidate_training_units[i]["is_filtered"] for i in unit_idxs
                    )
                    unit_advantages = [candidate_training_units[i]["advantage"] for i in unit_idxs]
                    rollout["advantage"] = sum(unit_advantages) / len(unit_advantages) if unit_advantages else 0.0
            else:
                candidate_training_units = list(candidate_rollouts)
                candidate_rollout_to_unit_idxs = [[i] for i in range(num_candidate_rollouts)]
                compute_advantages(candidate_rollouts, config.rollouts_per_example, config.advantage)
                apply_filters(rollout_filters, candidate_rollouts)
            return candidate_training_units, candidate_rollout_to_unit_idxs

        for attempt in range(MAX_EMPTY_BATCH_ATTEMPTS):
            train_rollouts = await scheduler.generate_batch(step=progress.step)
            generate_completions_time += scheduler.last_batch_generation_time
            training_units, rollout_to_unit_idxs = prepare_candidate_batch(train_rollouts)

            num_rollouts = len(train_rollouts)
            n_trainable = sum(1 for u in training_units if not u["is_filtered"])
            if n_trainable > 0:
                break

            if attempt == MAX_EMPTY_BATCH_ATTEMPTS - 1:
                logger.error(
                    f"Attempt {attempt + 1}/{MAX_EMPTY_BATCH_ATTEMPTS} at step {progress.step} "
                    f"filtered out all {len(train_rollouts)} rollouts - crashing orchestrator"
                )
                reason = (
                    f"All rollouts were filtered out on "
                    f"{MAX_EMPTY_BATCH_ATTEMPTS} consecutive attempts at step {progress.step}"
                )
                evicted_path = config.output_dir / "control" / "evicted.txt"
                evicted_path.parent.mkdir(parents=True, exist_ok=True)
                evicted_path.write_text(reason)
                raise RuntimeError(reason)

            logger.warning(
                f"Attempt {attempt + 1}/{MAX_EMPTY_BATCH_ATTEMPTS} at step {progress.step} "
                f"filtered out all {len(train_rollouts)} rollouts - retrying batch generation"
            )

        num_unique_examples = len({(r["env_name"], r["example_id"]) for r in train_rollouts})
        trainable_denominator = len(training_units) if is_ma else num_rollouts
        trainable_ratio = n_trainable / max(trainable_denominator, 1)
        if trainable_ratio <= 0.1:
            logger.warning(
                f"Only {n_trainable}/{trainable_denominator} training units in the batch are trainable "
                f"({trainable_ratio:.1%}) - this can mean the tasks are too easy or too hard for the "
                "model, consider reviewing the task difficulty of your environment(s)"
            )

        # Save train rollouts + tier-2 metrics to disk (background threads)
        step_path = get_step_path(get_rollout_dir(config.output_dir), progress.step)
        await _persist_rollouts_and_metrics(
            train_rollouts,
            step_path,
            "train",
            step=progress.step,
            dump_trajectory=config.dump_trajectory,
            monitor=monitor,
        )

        # VLM: offload base64 images to disk immediately to free memory
        if is_vlm:
            offload_start = time.perf_counter()
            num_offloaded = offload_images_to_disk(train_rollouts, config.output_dir)
            if num_offloaded:
                logger.info(
                    f"VLM offloaded {num_offloaded} unique images to disk in {time.perf_counter() - offload_start:.2f}s"
                )

        # Convert training units to training samples
        parallel_preprocess_start = time.perf_counter()

        # Stage 1: pretokenize + (for VLM) build image cache concurrently.
        # Pretokenize is a no-op when the renderer client already populated
        # `tokens` on each trajectory step, but the fallback-tokenizer path
        # and image-cache build are both CPU-heavy. Running them on threads
        # and awaiting a single gather lets whichever finishes first free
        # the event loop immediately and, with max_async_level >= 2, overlaps
        # this whole stage with inference for the next batch.
        async def _pretokenize_all() -> None:
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        pretokenize_rollout_trajectory,
                        unit,
                        tokenizer,
                        processor=processor,
                        renderer=renderer,
                    )
                    for unit in training_units
                )
            )

        if is_vlm:
            mm_token_type_ids_mapping = {}
            if hasattr(processor, "image_token_id") and processor.image_token_id is not None:
                mm_token_type_ids_mapping[processor.image_token_id] = 1
            if hasattr(processor, "video_token_id") and processor.video_token_id is not None:
                mm_token_type_ids_mapping[processor.video_token_id] = 2
            _, vlm_cache = await asyncio.gather(
                _pretokenize_all(),
                asyncio.to_thread(build_vlm_image_cache, train_rollouts, processor),
            )
            logger.info(
                f"VLM timing: extract={vlm_cache.extract_time:.2f}s, preprocess={vlm_cache.preprocess_time:.2f}s "
                f"({vlm_cache.num_unique_images} unique images from {vlm_cache.num_unique_examples} examples)"
            )
        else:
            await _pretokenize_all()
            vlm_cache = None
            mm_token_type_ids_mapping = None

        # Process training units in parallel
        def process_unit(unit: vf.RolloutOutput, unit_idx: int) -> list[TrainingSample] | None:
            return interleave_rollout(
                unit,
                vlm_cache=vlm_cache,
                cache_key=unit_idx,
                mm_token_type_ids_mapping=mm_token_type_ids_mapping,
            )

        results = await asyncio.gather(
            *(asyncio.to_thread(process_unit, unit, unit_idx) for unit_idx, unit in enumerate(training_units))
        )

        # Collect results and assign advantages. Metrics stay per rollout;
        # training examples are per unit after optional multi-agent fan-out.
        train_examples: list[TrainingSample] = []
        rollout_prefill_lens: list[int] = []
        rollout_decode_lens: list[int] = []
        rollout_samples_per_rollout: list[int] = []
        num_prefill_tokens = 0
        num_decode_tokens = 0
        for unit_idxs in rollout_to_unit_idxs:
            rollout_prefill_tokens = 0
            rollout_decode_tokens = 0
            rollout_total_samples = 0
            for unit_idx in unit_idxs:
                samples = results[unit_idx] or []
                unit = training_units[unit_idx]
                rollout_total_samples += len(samples)
                if unit["is_filtered"]:
                    # Filtered units never reach the trainer, so their tokens
                    # don't belong in training-usage accounting -- otherwise
                    # report_training_usage (usage_type="training") over-bills
                    # by the filtered slice.
                    continue
                for sample in samples:
                    sample.advantage = unit["advantage"]
                    sample.reward = unit["reward"]
                    if config.use_sft_loss:
                        sample.sft_loss = True
                    sample_decode_tokens = sum(sample.completion_mask)
                    sample_prefill_tokens = len(sample.prompt_ids) + len(sample.completion_mask) - sample_decode_tokens
                    rollout_decode_tokens += sample_decode_tokens
                    rollout_prefill_tokens += sample_prefill_tokens
                    train_examples.append(sample)
            rollout_samples_per_rollout.append(rollout_total_samples)
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
            }
        )

        # Separate DataFrames for env reward function metrics, filter flags, and per-rollout timings
        # to avoid column name collisions
        metrics_df = pd.DataFrame([rollout["metrics"] for rollout in train_rollouts])
        filter_df = pd.DataFrame([rollout["filters"] for rollout in train_rollouts])
        timing_df = pd.DataFrame(
            [
                {
                    "total": rollout["timing"]["total"],
                    "setup": rollout["timing"]["setup"]["duration"],
                    "generation": rollout["timing"]["generation"]["duration"],
                    "model": rollout["timing"]["model"]["duration"],
                    "env": rollout["timing"]["env"]["duration"],
                    "scoring": rollout["timing"]["scoring"]["duration"],
                    "overhead": rollout["timing"]["overhead"],
                }
                for rollout in train_rollouts
            ]
        )

        # Update progress metrics
        num_tokens = int(results_df.seq_len.sum())
        progress.total_tokens += num_tokens
        progress.total_samples += num_rollouts
        progress.total_problems += num_unique_examples

        def compute_solve_rates(df):
            """Compute solve_none, solve_all, effective_batch_size for a set of rollouts."""
            reward_per_problem = df.groupby(["env_name", "example_id"]).reward.sum()
            solve_none = (reward_per_problem == 0).mean()
            solve_all = (reward_per_problem == config.rollouts_per_example).mean()
            return solve_none, solve_all, 1 - solve_none - solve_all

        # Group by (env_name, example_id) to average across rollouts within each problem
        by_example = results_df.groupby(["env_name", "example_id"])

        solve_none, solve_all, effective_batch_size = compute_solve_rates(results_df)
        filtered_mask = results_df.is_filtered.astype(bool)
        unfiltered_df = results_df.loc[~filtered_mask]
        filtered_df = results_df.loc[filtered_mask]
        train_batch_filter_metrics = {
            "train_batch/rollouts_unconditioned_on_filtering": float(len(results_df)),
            "train_batch/rollouts_conditioned_on_filtering": float(len(unfiltered_df)),
            "train_batch/rollouts_filtered_out": float(len(filtered_df)),
            "train_batch/reward_unconditioned_on_filtering/mean": results_df.reward.mean(),
        }
        if not unfiltered_df.empty:
            train_batch_filter_metrics["train_batch/reward_conditioned_on_filtering/mean"] = unfiltered_df.reward.mean()
        if not filtered_df.empty:
            train_batch_filter_metrics["train_batch/reward_filtered_out/mean"] = filtered_df.reward.mean()
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
            **{
                f"timing/all/{key}/{stat}": getattr(
                    timing_df[key].groupby([results_df.env_name, results_df.example_id]).mean(),
                    stat,
                )()
                for key in timing_df.columns
                for stat in ("mean", "max", "min")
            },
            # Train reward
            "reward/all/mean": by_example.reward.mean().mean(),
            "reward/all/max": by_example.reward.mean().max(),
            "reward/all/min": by_example.reward.mean().min(),
            **train_batch_filter_metrics,
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
        ]

        for env, env_df in results_df.groupby("env_name"):
            env_by_example = env_df.groupby("example_id")
            for col in per_env_columns:
                to_log[f"{col}/{env}/mean"] = env_by_example[col].mean().mean()
                to_log[f"{col}/{env}/max"] = env_by_example[col].mean().max()
                if col != "is_truncated":
                    to_log[f"{col}/{env}/min"] = env_by_example[col].mean().min()
            env_timing_df = timing_df.loc[env_df.index]
            for key in timing_df.columns:
                per_example = env_timing_df.groupby(env_df["example_id"])[key].mean()
                to_log[f"timing/{env}/{key}/mean"] = per_example.mean()
                to_log[f"timing/{env}/{key}/max"] = per_example.max()
                to_log[f"timing/{env}/{key}/min"] = per_example.min()
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

        train_comparison_metrics = {key: value for key, value in to_log.items() if key.startswith("train_batch/")}
        train_comparison_metrics["step"] = progress.step
        await asyncio.to_thread(
            _write_scalar_metrics,
            step_path / "train_filter_metrics.json",
            train_comparison_metrics,
        )

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
        step_message = f"Step {progress.step} | Time: {step_time:.2f}s | Reward: {reward_mean:.4f} | Seq. Length: {by_example.seq_len.mean().mean():.1f} tokens/sample | Max. Off-Policy Level: {scheduler.max_off_policy_level}"
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
                    get_client=get_eval_client_config,
                    ckpt_step=ckpt_step,
                    step=progress.step,
                    cache_salt=str(ckpt_step),
                    multi_agent=config.multi_agent if eval_env.is_multi_agent else None,
                    eval_clients=inference_pool.eval_clients,
                )
                for eval_env in eval_envs
            ]
        )

        # Save final eval rollouts to disk
        eval_rollouts = [o for outputs in eval_results for o in outputs]
        if eval_rollouts:
            step_path = get_step_path(get_rollout_dir(config.output_dir), progress.step)
            await _persist_rollouts_and_metrics(
                eval_rollouts,
                step_path,
                "eval",
                step=progress.step,
                dump_trajectory=config.dump_trajectory,
                monitor=monitor,
            )

    monitor.save_final_summary()

    # Write final checkpoint
    if ckpt_manager is not None:
        logger.info("Writing final checkpoint")
        ckpt_manager.save(progress, buffer, step=progress.step, rae_state=advantage_state)

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


async def setup_rollout_inference_pool(
    *,
    config: OrchestratorConfig,
    rollout_client_config,
    rollout_model_name: str,
    tokenizer,
    logger,
):
    """Set up rollout inference.

    Routing policy is driven by ``config.use_token_client`` and
    ``config.use_renderer`` (mutually exclusive — config-level validators
    block both being True):

      - external teacher rollout → MITO (``openai_chat_completions``),
        forced regardless of the toggles (config-level validator
        rejects ``use_token_client`` / ``use_renderer`` in that case)
      - ``use_renderer=True``  → renderer client (``/v1/generate``).
        Not allowed for VLMs (validated at config time).
      - ``use_token_client=True`` → TITO
        (``openai_chat_completions_token``, ``/v1/chat/completions/tokens``).
        Default. VLMs land here too.
      - both False → MITO (``openai_chat_completions``).
    """
    if config.teacher_rollout_model is not None:
        logger.info("Using external rollout model (MITO) without renderer client")
        inference_pool = await setup_inference_pool(
            rollout_client_config,
            model_name=rollout_model_name,
            train_client_type="openai_chat_completions",
            eval_client_type="openai_chat_completions",
        )
        return None, inference_pool

    if config.use_renderer:
        renderer = create_renderer(
            tokenizer,
            renderer=config.renderer.name,
            tool_parser=config.renderer.tool_parser,
            reasoning_parser=config.renderer.reasoning_parser,
        )
        logger.info(f"Initialized {type(renderer).__name__} for {config.model.name}")
        inference_pool = await setup_inference_pool(
            rollout_client_config,
            model_name=rollout_model_name,
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_name=config.renderer.name,
            tool_parser=config.renderer.tool_parser,
            reasoning_parser=config.renderer.reasoning_parser,
            renderer_pool_size=config.renderer.pool_size,
        )
        logger.info("Using direct renderer rollout client")
        return renderer, inference_pool

    train_client_type = "openai_chat_completions_token" if config.use_token_client else "openai_chat_completions"
    if config.use_token_client:
        logger.info("Using token client (TITO) for rollouts — server-side templating, /v1/chat/completions/tokens")
    else:
        logger.info("Using MITO (openai_chat_completions) for rollouts")
    inference_pool = await setup_inference_pool(
        rollout_client_config,
        model_name=rollout_model_name,
        train_client_type=train_client_type,
        eval_client_type="openai_chat_completions",
    )
    return None, inference_pool


if __name__ == "__main__":
    main()
