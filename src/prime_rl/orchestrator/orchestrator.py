import asyncio
import os
import time

import tomli_w
import verifiers as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.batcher import Done, TrainBatcher, build_strategy
from prime_rl.orchestrator.buffer import setup_buffer
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.engine import Group, RolloutEngine
from prime_rl.orchestrator.filters import setup_filters
from prime_rl.orchestrator.inference_admin import InferenceAdmin
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.scheduler import Scheduler
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import setup_training_batch_sender
from prime_rl.utils.config import cli
from prime_rl.utils.heartbeat import Heartbeat
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import get_broadcast_dir
from prime_rl.utils.utils import get_env_ids_to_install, install_env


async def run(cfg: OrchestratorConfig) -> None:
    assert cfg.max_inflight_rollouts is not None
    assert cfg.advantage is not None, "advantage config required"
    assert len(cfg.client.base_url) == 1, "single inference endpoint only"

    logger = get_logger()
    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info(f"Model: {cfg.model.name}")

    # Trainer reads orchestrator config from output_dir/control/orch.toml
    # (see prime_rl.trainer.runs.RunManager.get_orchestrator_config).
    control_dir = cfg.output_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(cfg.model_dump(exclude_none=True, mode="json"), f)
    logger.info(f"Wrote orch config to {control_dir / 'orch.toml'}")
    logger.info(
        f"Batching: {cfg.batch_size.type} | Rollouts/example: {cfg.rollouts_per_example} | "
        f"Max in-flight: {cfg.max_inflight_rollouts} | Max off-policy: {cfg.max_off_policy_steps}"
    )

    env_ids_to_install = set(get_env_ids_to_install(cfg.train.env))
    if cfg.eval is not None:
        env_ids_to_install.update(get_env_ids_to_install(cfg.eval.env))
    for env_id in env_ids_to_install:
        install_env(env_id, prerelease=cfg.env_install_prerelease)

    tokenizer = setup_tokenizer(cfg.tokenizer)
    logger.info(f"Tokenizer ready: {cfg.tokenizer.name}")

    setup_monitor(
        wandb_config=cfg.wandb,
        output_dir=cfg.output_dir,
        tokenizer=tokenizer,
        run_config=cfg,
        prime_config=cfg.prime_monitor,
    )
    if cfg.wandb is not None:
        logger.info(f"Wandb monitor ready (project={cfg.wandb.project}, name={cfg.wandb.name})")

    # Resolve the resume step early so we can suppress eval_at_zero on resume
    # (the original run already evaluated the base model).
    ckpt_manager = setup_ckpt_manager(cfg.output_dir, cfg.ckpt)
    resume_step: int | None = None
    if cfg.ckpt and cfg.ckpt.resume_step is not None and ckpt_manager is not None:
        if cfg.ckpt.resume_step == -1:
            resume_step = ckpt_manager.latest_step()
            if resume_step is None:
                logger.warning("ckpt.resume_step=-1 set but no orch checkpoints found; starting fresh")
        else:
            resume_step = cfg.ckpt.resume_step

    train_env_names = [e.resolved_name for e in cfg.train.env]
    buffer = setup_buffer(cfg.buffer, train_env_names, seed=cfg.seed)
    if buffer is not None:
        thresholds = []
        if cfg.buffer.easy_threshold is not None:
            thresholds.append(f"easy>={cfg.buffer.easy_threshold}")
        if cfg.buffer.hard_threshold is not None:
            thresholds.append(f"hard<={cfg.buffer.hard_threshold}")
        logger.info(f"Difficulty buffer enabled ({', '.join(thresholds)})")

    scheduler = Scheduler(
        train_envs=cfg.train.env,
        train_rollouts_per_example=cfg.rollouts_per_example,
        eval_envs=cfg.eval.env if cfg.eval else None,
        eval_interval=cfg.eval.interval if cfg.eval else None,
        eval_at_zero=(cfg.eval.eval_base_model if cfg.eval else False) and resume_step is None,
        seed=cfg.seed,
        buffer=buffer,
    )
    for task in scheduler.tasks:
        logger.info(f"Train task '{task.id}' ready (rollouts/group={task.rollouts_per_group})")
    for task in scheduler.eval_tasks:
        logger.info(f"Eval task '{task.id}' ready (rollouts/group={task.rollouts_per_group})")
    if cfg.eval is not None:
        logger.info(
            f"Eval interval: {cfg.eval.interval} | eval_base_model: {cfg.eval.eval_base_model} | "
            f"eval envs: {len(scheduler.eval_tasks)}"
        )

    # Engine-wide cap: total concurrent groups across all tasks. Each group
    # fans out to rollouts_per_example rollouts, so divide to match the old
    # orch's semantics where max_inflight_rollouts counts individual rollouts.
    concurrency = max(1, cfg.max_inflight_rollouts // cfg.rollouts_per_example)
    logger.info(f"Engine concurrency: {concurrency} groups across {len(scheduler.tasks)} task(s)")

    # Bounded so the batcher's async-level barrier cascades backpressure into
    # the engine instead of letting it accumulate unbounded in-flight rollouts.
    # Sized from concurrency (upper bound on in-flight groups) rather than
    # batch size, since token/step modes don't have a fixed batch size.
    groups_q: asyncio.Queue[Group] = asyncio.Queue(maxsize=concurrency * (cfg.max_async_level + 1))

    client = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=cfg.client.base_url[0],
        api_key_var=cfg.client.api_key_var,
        timeout=cfg.client.timeout,
        connect_timeout=cfg.client.connect_timeout,
    )

    engine = RolloutEngine(
        scheduler=scheduler,
        out_q=groups_q,
        client=client,
        model=cfg.model.name,
        max_off_policy=cfg.max_off_policy_steps,
        concurrency=concurrency,
        tasks_per_minute=cfg.tasks_per_minute,
        max_rollout_time_seconds=(cfg.max_rollout_time_minutes * 60.0) if cfg.max_rollout_time_minutes else None,
    )
    if cfg.tasks_per_minute is not None:
        logger.info(f"Rate limit: {cfg.tasks_per_minute} tasks/min")
    if cfg.max_rollout_time_minutes is not None:
        logger.info(f"Rollout time cap: {cfg.max_rollout_time_minutes} min/group")
    training_sender = setup_training_batch_sender(cfg.output_dir, cfg.rollout_transport)
    rollout_filters = setup_filters(cfg.filters, vocab_size=tokenizer.vocab_size)
    strategy = build_strategy(cfg.batch_size)
    heartbeat = Heartbeat(cfg.heartbeat.url) if cfg.heartbeat else None
    if heartbeat is not None:
        logger.info(f"Heartbeat enabled ({cfg.heartbeat.url})")

    broadcast_dir = get_broadcast_dir(cfg.output_dir)
    admin = InferenceAdmin(
        cfg.client.base_url[0],
        os.getenv(cfg.client.api_key_var, "EMPTY"),
        broadcast_dir,
        mode=cfg.weight_broadcast.type,
    )
    logger.info(f"Admin client ready ({admin.base_url})")
    logger.info(f"Weight broadcast mode: {cfg.weight_broadcast.type}")

    inference_metrics = None
    if cfg.collect_inference_metrics:
        inference_metrics = InferenceMetricsCollector(admin.client)
        logger.info("Inference metrics collection enabled")

    batcher = TrainBatcher(
        groups_q,
        tokenizer,
        training_sender,
        engine,
        strategy,
        cfg.advantage,
        filters=rollout_filters,
        max_steps=cfg.max_steps,
        max_training_batches_ahead=cfg.max_async_level,
        strict_async_level=cfg.strict_async_level,
        eval_counter=scheduler,
        ckpt_manager=ckpt_manager,
        ckpt_interval=cfg.ckpt.interval if cfg.ckpt else None,
        buffer=buffer,
        heartbeat=heartbeat,
        inference_metrics=inference_metrics,
    )
    logger.info(f"Training batch sender ready ({cfg.rollout_transport.type})")

    logger.info("Waiting for inference server to be healthy...")
    t0 = time.perf_counter()
    await admin.wait_healthy()
    logger.success(f"Inference server ready ({time.perf_counter() - t0:.1f}s)")

    await admin.check_model(cfg.model.name)
    logger.success(f"Model '{cfg.model.name}' loaded on inference server")

    if cfg.weight_broadcast.type == "nccl":
        await admin.init_nccl_broadcaster(
            host=cfg.weight_broadcast.host,
            port=cfg.weight_broadcast.port,
            timeout=cfg.weight_broadcast.timeout,
            inference_world_size=cfg.weight_broadcast.inference_world_size,
            quantize_in_weight_transfer=cfg.weight_broadcast.quantize_in_weight_transfer,
        )
        logger.success(
            f"NCCL broadcast initialized (host={cfg.weight_broadcast.host}, port={cfg.weight_broadcast.port}, "
            f"inference_world_size={cfg.weight_broadcast.inference_world_size})"
        )

    watcher = WeightWatcher(broadcast_dir, observers=[admin, engine, scheduler])

    if resume_step is not None and ckpt_manager is not None:
        state = ckpt_manager.load(resume_step)
        batcher.step = state.step
        scheduler.last_eval_step = state.last_eval_step
        if buffer is not None and state.buffer_state and not (cfg.ckpt and cfg.ckpt.skip_buffer):
            buffer.load_state_dict(state.buffer_state)
        if cfg.eval and cfg.eval.skip_eval_on_resume:
            # bump last_eval_step past current so the next interval boundary
            # is the first eval the resumed run sees
            scheduler.last_eval_step = state.step
            logger.info(f"Skipping next eval on resume (last_eval_step={state.step})")
        await admin.on_new_version(state.step)
        await engine.on_new_version(state.step)
        watcher.current_step = state.step
        logger.success(f"Resumed orch from step {state.step} (eval cursor at {scheduler.last_eval_step})")

    logger.success("Orchestrator starting — producing rollouts")
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(engine.run())
            tg.create_task(batcher.run())
            tg.create_task(watcher.run())
            tg.create_task(admin.watch_health())
    except* Done:
        logger.success(f"Orchestrator finished: reached max_steps={cfg.max_steps}")


def main() -> None:
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")
    cfg = cli(OrchestratorConfig)
    setup_logger(
        log_level=cfg.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=cfg.log.json_logging,
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
