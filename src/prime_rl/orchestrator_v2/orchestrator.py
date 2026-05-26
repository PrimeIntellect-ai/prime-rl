"""Async-pipelined RL orchestrator v2.

``orchestrate(config)`` wires four long-lived tasks under an ``asyncio.TaskGroup``:

* ``WeightWatcher`` — polls broadcast dir, advances ``Policy.version``, notifies observers.
* ``RolloutDispatcher`` — the only thing that schedules rollouts. Shares a single
  ``asyncio.Semaphore(max_inflight_rollouts)`` and ``AsyncLimiter(tasks_per_minute)``
  across train and eval. Phase-based priority (PREFER_TRAIN / PREFER_EVAL) gives
  drain-switch overlap around eval boundaries. Owns off-policy cancellation.
* ``TrainBatcher`` — drains the rollout queue, applies pre_batch_filters →
  compute_advantages → batch buffer → post_batch_filters → tokenize+send. Routes
  ``kind="eval"`` trajectories to a per-eval-step aggregator that flushes
  metrics on the trigger step.
* ``IntervalLogger`` — async task that wakes on a fixed cadence and logs
  dispatcher gauges + event-loop lag + vLLM Prometheus snapshots on the wandb
  time axis (independent from step semantics).

The single ``OrchestratorConfig`` is shared between the legacy orchestrator and
this one; only ``orchestrator.experimental.use_orch_v2`` flips the
``rl`` entrypoint to launch this binary.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import time

import tomli_w

import prime_rl._compat  # noqa: F401 — patch ring_flash_attn compat before transitive imports
from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.filters import setup_filters
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.patches import (
    monkey_patch_chat_completion_logprobs,
    monkey_patch_oai_iterable_types,
)
from prime_rl.orchestrator.utils import (
    get_weight_dir,
    set_default_executor,
)
from prime_rl.orchestrator.vf_utils import intercept_vf_logging
from prime_rl.orchestrator_v2.batcher import Done, TrainBatcher
from prime_rl.orchestrator_v2.ckpt import Progress, setup_ckpt_manager
from prime_rl.orchestrator_v2.dispatcher import RolloutDispatcher
from prime_rl.orchestrator_v2.log_loop import IntervalLogger
from prime_rl.orchestrator_v2.policy import Policy
from prime_rl.orchestrator_v2.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import setup_training_batch_sender
from prime_rl.utils.client import init_nccl_broadcast, setup_inference_pool
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import get_log_dir
from prime_rl.utils.utils import (
    clean_exit,
    get_env_ids_to_install,
    install_env,
    resolve_latest_ckpt_step,
)

monkey_patch_oai_iterable_types()
monkey_patch_chat_completion_logprobs()


# Hard wall-clock budget for the v2 orchestrator's post-training cleanup.
# Mirrors the legacy orchestrator's safety: persist artifacts before this point,
# then force-exit if graceful shutdown wedges (env-server ZMQ recv, vLLM admin
# aclose, etc).
SHUTDOWN_TIMEOUT_S = 300


@clean_exit
async def orchestrate(config: OrchestratorConfig) -> None:
    logger = setup_logger(
        config.log.level,
        json_logging=config.log.json_logging,
    )
    intercept_vf_logging(logger="verifiers.serve", level="WARN")
    logger.info(f"Starting orchestrator v2 ({config.training_mode})")

    set_default_executor()

    if config.bench:
        logger.warning(f"Running in benchmark mode (max_steps={config.max_steps})")

    config_dir = config.output_dir / "control"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), f)

    # Install envs (train + eval) up-front so the env-server processes can spawn.
    env_ids_to_install = set(get_env_ids_to_install(config.train.env))
    if config.eval is not None:
        env_ids_to_install.update(get_env_ids_to_install(config.eval.env))
    for env_id in env_ids_to_install:
        install_env(env_id, prerelease=config.env_install_prerelease)

    logger.info(f"Initializing tokenizer ({config.tokenizer})")
    tokenizer = setup_tokenizer(config.tokenizer)

    logger.info(
        f"Initializing student inference pool (base_url={', '.join(config.student.client.base_url)}, "
        f"model={config.student.model.name})"
    )
    renderer, student_inference = await _setup_student_inference_pool(config=config, tokenizer=tokenizer)
    mm_token_type_ids_mapping = getattr(renderer, "mm_token_type_id_map", None) if renderer is not None else None
    if mm_token_type_ids_mapping == {}:
        mm_token_type_ids_mapping = None

    teacher_inference = None
    if config.teacher is not None:
        logger.info(
            f"Initializing teacher inference pool (base_url={', '.join(config.teacher.client.base_url)}, "
            f"model={config.teacher.model.name})"
        )
        teacher_inference = await setup_inference_pool(
            config.teacher.client,
            model_name=config.teacher.model.name,
            train_client_type="openai_chat_completions",
        )

    logger.info(f"Initializing monitor (wandb={config.wandb}, prime_monitor={config.prime_monitor})")
    monitor = setup_monitor(
        wandb_config=config.wandb,
        prime_config=config.prime_monitor,
        output_dir=config.output_dir,
        tokenizer=tokenizer,
        run_config=config,
        keep_full_history=config.bench,
    )

    heart = Heartbeat(config.heartbeat.url) if config.heartbeat is not None else None

    pre_filters = setup_filters(config.pre_batch_filters, vocab_size=tokenizer.vocab_size)
    post_filters = setup_filters(config.post_batch_filters, vocab_size=tokenizer.vocab_size)

    logger.info("Loading training environments")
    train_envs = TrainEnvs(config.train.env)
    if config.training_mode == "sft":
        for env in train_envs:
            env.sampling_args.pop("logprobs", None)
    logger.info(f"Loaded {len(train_envs)} training environment(s) ({', '.join(train_envs.names)})")
    await train_envs.start(
        log_dir=get_log_dir(config.output_dir.parent) / "envs" / "train",
        log_level=config.log.vf_level,
        json_logging=config.log.json_logging,
    )
    logger.success("Train environment(s) ready")

    eval_envs: EvalEnvs | None = None
    if config.eval is not None:
        logger.info("Loading eval environment(s)")
        eval_envs = EvalEnvs(config.eval.env)
        logger.info(f"Loaded {len(eval_envs)} eval environment(s) ({', '.join(eval_envs.names)})")
        await eval_envs.start(
            log_dir=get_log_dir(config.output_dir.parent) / "envs" / "eval",
            log_level=config.log.vf_level,
            json_logging=config.log.json_logging,
        )
        logger.success("Eval environment(s) ready")

    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
    progress = Progress()
    resume_step: int | None = None
    if config.ckpt is not None and config.ckpt.resume_step is not None and ckpt_manager is not None:
        if config.ckpt.resume_step == -1:
            resume_step = resolve_latest_ckpt_step(ckpt_manager.ckpt_dir)
        else:
            resume_step = config.ckpt.resume_step

    # Shared mutable state: watcher mutates ``policy.version``, dispatcher reads it.
    policy = Policy(
        version=0,
        model_name=student_inference.model_name,
    )

    # Wait for inference pools to come up before bootstrapping the dispatcher.
    logger.info("Waiting for student inference pool to be ready")
    await student_inference.wait_for_ready(config.student.model.name)
    logger.success("Student inference pool ready")
    if teacher_inference is not None:
        assert config.teacher is not None
        logger.info("Waiting for teacher inference pool to be ready")
        await teacher_inference.wait_for_ready(config.teacher.model.name)
        logger.success("Teacher inference pool ready")

    inference_metrics_collector: InferenceMetricsCollector | None = None
    if config.wandb is not None and config.collect_inference_metrics:
        inference_metrics_collector = InferenceMetricsCollector(student_inference.admin_clients)
        await inference_metrics_collector.start()

    logger.info(f"Initializing weight broadcast ({config.weight_broadcast})")
    if config.weight_broadcast.type == "nccl":
        await init_nccl_broadcast(
            student_inference.admin_clients,
            config.weight_broadcast.host,
            config.weight_broadcast.port,
            config.weight_broadcast.timeout,
            inference_world_size=config.weight_broadcast.inference_world_size,
            quantize_in_weight_transfer=config.weight_broadcast.quantize_in_weight_transfer,
        )

    logger.info(f"Initializing training batch sender ({config.rollout_transport})")
    training_batch_sender = setup_training_batch_sender(config.output_dir, config.rollout_transport)

    lora_name = config.student.model.lora.name if config.student.model.lora else None

    # Apply resume weights to vLLM before the dispatcher starts spinning.
    if resume_step is not None and ckpt_manager is not None:
        ckpt_manager.load(progress, step=resume_step)
        logger.info(f"Resuming v2 orchestrator from checkpoint step {resume_step}")
        check_exists = config.weight_broadcast.type != "nccl"
        wait_timeout = config.ckpt.wait_for_weights_timeout if config.ckpt else None
        weights_path = get_weight_dir(
            config.output_dir, progress.step, check_exists=check_exists, wait_timeout=wait_timeout
        )
        await student_inference.update_weights(weights_path, lora_name=lora_name, step=progress.step)
        if lora_name is not None:
            student_inference.update_model_name(lora_name)
            policy.model_name = lora_name
        policy.version = progress.step
    else:
        logger.info("Training from scratch")

    dispatcher = RolloutDispatcher(
        config=config,
        train_envs=train_envs,
        eval_envs=eval_envs,
        student_inference=student_inference,
        teacher_inference=teacher_inference,
        policy=policy,
        resume_step=resume_step,
    )

    batcher = TrainBatcher(
        config=config,
        dispatcher=dispatcher,
        tokenizer=tokenizer,
        renderer=renderer,
        mm_token_type_ids_mapping=mm_token_type_ids_mapping,
        student_inference=student_inference,
        teacher_inference=teacher_inference,
        pre_filters=pre_filters,
        post_filters=post_filters,
        sender=training_batch_sender,
        ckpt_manager=ckpt_manager,
        progress=progress,
        policy=policy,
        heart=heart,
        monitor=monitor,
    )

    watcher = WeightWatcher(
        config=config,
        policy=policy,
        student_inference=student_inference,
        observers=[dispatcher],
        lora_name=lora_name,
        ckpt_step=progress.step,
    )

    log_loop = IntervalLogger(
        config=config,
        dispatcher=dispatcher,
        batcher=batcher,
        policy=policy,
        inference_metrics=inference_metrics_collector,
        monitor=monitor,
    )

    logger.info(f"Starting orchestrator v2 loop (max_steps={config.max_steps or 'infinite'})")
    start_time = time.perf_counter()

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(dispatcher.run(), name="dispatcher")
            tg.create_task(batcher.run(), name="batcher")
            tg.create_task(watcher.run(), name="watcher")
            tg.create_task(log_loop.run(), name="log_loop")
    except* Done as eg:
        # ``Done`` is raised by the batcher once max_steps is reached. Treat as
        # graceful completion; let the rest of the task group unwind.
        logger.success(f"Orchestrator finished: reached max_steps={config.max_steps} ({eg!s})")

    logger.success(f"Orchestrator v2 step loop done in {time.perf_counter() - start_time:.1f}s")

    # Final evals before shutdown. After ``Done`` the dispatcher/batcher tasks
    # are cancelled, so we don't route through the pipeline; instead we call
    # the legacy ``EvalEnv.evaluate`` directly (same pattern as the legacy
    # orchestrator's tail). It uses ``monitor.log`` / ``monitor.log_eval_samples``
    # directly and consumes the existing student inference pool's eval client
    # round-robin.
    if eval_envs is not None and config.eval is not None:
        logger.info("Running final evals")
        await asyncio.gather(
            *(
                eval_env.evaluate(
                    model_name=student_inference.model_name,
                    get_client=student_inference.get_eval_client,
                    step=progress.step,
                    cache_salt=str(progress.step),
                )
                for eval_env in eval_envs
            ),
            return_exceptions=True,
        )

    monitor.save_final_summary()

    if ckpt_manager is not None:
        logger.info("Writing final v2 checkpoint")
        ckpt_manager.save(progress, step=progress.step)

    async def _graceful_shutdown() -> None:
        training_batch_sender.close()
        await dispatcher.stop()
        await batcher.stop()
        await watcher.stop()
        await log_loop.stop()
        if inference_metrics_collector is not None:
            await inference_metrics_collector.stop()
        await student_inference.stop()
        if teacher_inference is not None:
            await teacher_inference.stop()
        train_envs.shutdown()
        if eval_envs is not None:
            eval_envs.shutdown()

    shutdown_task = asyncio.create_task(_graceful_shutdown())
    _, pending = await asyncio.wait({shutdown_task}, timeout=SHUTDOWN_TIMEOUT_S)
    if pending:
        logger.warning(
            f"Orchestrator v2 shutdown did not complete within {SHUTDOWN_TIMEOUT_S}s; "
            "forcing process exit. Training artifacts are already persisted."
        )
        os._exit(0)
    await shutdown_task

    logger.success("Orchestrator v2 finished.")

    # Return free glibc heap pages to the OS so the launcher's exit isn't held
    # by malloc bookkeeping from numpy/pandas allocations.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as e:
        get_logger().debug(f"malloc_trim(0) failed: {e}")


async def _setup_student_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Mirror of ``prime_rl.orchestrator.orchestrator.setup_student_inference_pool``.

    Kept inline so the v2 orchestrator can evolve client setup independently
    (e.g. drop the teacher_inference for sft mode in a follow-up) without
    touching the legacy code path.
    """
    from renderers.base import create_renderer

    client_config = config.student.client
    model_name = config.student.model.name

    if config.use_renderer:
        renderer = create_renderer(
            tokenizer,
            renderer=config.renderer.name,
            tool_parser=config.renderer.tool_parser,
            reasoning_parser=config.renderer.reasoning_parser,
            preserve_all_thinking=config.renderer.preserve_all_thinking,
            preserve_thinking_between_tool_calls=config.renderer.preserve_thinking_between_tool_calls,
        )
        get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
        inference_pool = await setup_inference_pool(
            client_config,
            model_name=model_name,
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_name=config.renderer.name,
            tool_parser=config.renderer.tool_parser,
            reasoning_parser=config.renderer.reasoning_parser,
            renderer_pool_size=config.renderer.pool_size,
            preserve_all_thinking=config.renderer.preserve_all_thinking,
            preserve_thinking_between_tool_calls=config.renderer.preserve_thinking_between_tool_calls,
        )
        get_logger().info("Using direct renderer rollout client")
        return renderer, inference_pool

    get_logger().info("Using MITO (openai_chat_completions) for rollouts")
    inference_pool = await setup_inference_pool(
        client_config,
        model_name=model_name,
        train_client_type="openai_chat_completions",
        eval_client_type="openai_chat_completions",
    )
    return None, inference_pool


def main():
    """Entrypoint for direct invocation via ``python -m prime_rl.orchestrator_v2.orchestrator``."""
    from prime_rl.utils.config import cli
    from prime_rl.utils.process import set_proc_title

    set_proc_title("OrchestratorV2")
    import uvloop

    uvloop.install()
    asyncio.run(orchestrate(cli(OrchestratorConfig)))


if __name__ == "__main__":
    main()
