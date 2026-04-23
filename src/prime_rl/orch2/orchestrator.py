import asyncio
import os
import time

import tomli_w
import verifiers as vf

from prime_rl.configs.orchestrator import DefaultAdvantageConfig, OrchestratorConfig
from prime_rl.orch2.batcher import Done, TrainBatcher, build_strategy
from prime_rl.orch2.engine import Group, RolloutEngine
from prime_rl.orch2.inference_admin import InferenceAdmin
from prime_rl.orch2.scheduler import Scheduler
from prime_rl.orch2.watcher import WeightWatcher
from prime_rl.orchestrator.filters import setup_filters
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import setup_training_batch_sender
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import get_broadcast_dir
from prime_rl.utils.utils import get_env_ids_to_install, install_env


async def run(cfg: OrchestratorConfig) -> None:
    assert cfg.max_inflight_rollouts is not None
    assert isinstance(cfg.advantage, DefaultAdvantageConfig), "orch2 minimal only supports default advantage"
    assert len(cfg.client.base_url) == 1, "orch2 assumes a single inference endpoint"

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

    scheduler = Scheduler(cfg.train.env, kind="train")
    for task in scheduler.tasks:
        logger.info(f"Task '{task.id}' ready (kind={task.kind})")

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
        rollouts_per_group=cfg.rollouts_per_example,
        max_off_policy=cfg.max_off_policy_steps,
        concurrency=concurrency,
        tasks_per_minute=cfg.tasks_per_minute,
    )
    if cfg.tasks_per_minute is not None:
        logger.info(f"Rate limit: {cfg.tasks_per_minute} tasks/min")
    training_sender = setup_training_batch_sender(cfg.output_dir, cfg.rollout_transport)
    rollout_filters = setup_filters(cfg.filters, vocab_size=tokenizer.vocab_size)
    strategy = build_strategy(cfg.batch_size)
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
    )
    logger.info(f"Training batch sender ready ({cfg.rollout_transport.type})")

    broadcast_dir = get_broadcast_dir(cfg.output_dir)
    admin = InferenceAdmin(cfg.client.base_url[0], os.getenv(cfg.client.api_key_var, "EMPTY"), broadcast_dir)
    logger.info(f"Admin client ready ({admin.base_url})")

    logger.info(f"Weight broadcast mode: {cfg.weight_broadcast.type}")
    if cfg.weight_broadcast.type == "nccl":
        logger.warning("NCCL weight broadcast is not wired in orch2 yet — falling back to filesystem polling")

    logger.info("Waiting for inference server to be healthy...")
    t0 = time.perf_counter()
    await admin.wait_healthy()
    logger.success(f"Inference server ready ({time.perf_counter() - t0:.1f}s)")

    await admin.check_model(cfg.model.name)
    logger.success(f"Model '{cfg.model.name}' loaded on inference server")

    watcher = WeightWatcher(broadcast_dir, observers=[admin, engine, scheduler])

    logger.success("Orch2 starting — producing rollouts")
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(engine.run())
            tg.create_task(batcher.run())
            tg.create_task(watcher.run())
    except* Done:
        logger.success(f"Orch2 finished: reached max_steps={cfg.max_steps}")


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
