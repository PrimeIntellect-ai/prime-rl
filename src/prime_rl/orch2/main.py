import asyncio
import os
import time
from pathlib import Path

import tomli_w

from prime_rl.configs.orchestrator import DefaultAdvantageConfig, OrchestratorConfig
from prime_rl.orch2.batcher import Done, TrainBatcher
from prime_rl.orch2.engine import Group, RolloutEngine
from prime_rl.orch2.inference_admin import InferenceAdmin
from prime_rl.orch2.pool import build_pool
from prime_rl.orch2.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import setup_training_batch_sender
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.pathing import get_broadcast_dir


async def run(cfg: OrchestratorConfig) -> None:
    assert cfg.max_inflight_rollouts is not None
    assert cfg.batch_size is not None
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
        f"Batch size: {cfg.batch_size} | Rollouts/example: {cfg.rollouts_per_example} | "
        f"Max in-flight: {cfg.max_inflight_rollouts} | Max off-policy: {cfg.max_off_policy_steps}"
    )

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

    num_envs = len(cfg.train.env)
    # pool.concurrency bounds concurrent *groups*; each group fans out to
    # rollouts_per_example rollouts. Divide by that to match the old orch's
    # semantics where max_inflight_rollouts counts individual rollouts.
    per_env_concurrency = max(1, cfg.max_inflight_rollouts // (num_envs * cfg.rollouts_per_example))
    pools = [build_pool(env_cfg, cfg, per_env_concurrency) for env_cfg in cfg.train.env]
    for pool in pools:
        logger.info(
            f"Pool '{pool.id}' ready | concurrency={pool.concurrency} | "
            f"rollouts_per_group={pool.rollouts_per_group} | max_off_policy={pool.max_off_policy}"
        )

    # Bounded so the batcher's async-level barrier cascades backpressure into
    # the engine instead of letting it accumulate unbounded in-flight rollouts.
    groups_per_batch = max(1, cfg.batch_size // cfg.rollouts_per_example)
    groups_q: asyncio.Queue[Group] = asyncio.Queue(maxsize=groups_per_batch * (cfg.max_async_level + 1))

    engine = RolloutEngine(pools, groups_q)
    training_sender = setup_training_batch_sender(cfg.output_dir, cfg.rollout_transport)
    batcher = TrainBatcher(
        groups_q,
        tokenizer,
        training_sender,
        engine,
        cfg.batch_size,
        cfg.advantage,
        cfg.max_steps,
        cfg.max_async_level,
    )
    logger.info(f"Training batch sender ready ({cfg.rollout_transport.type})")

    admin = InferenceAdmin(cfg.client.base_url[0], os.getenv(cfg.client.api_key_var, "EMPTY"))
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

    async def on_new_weights(step: int, weight_dir: Path) -> None:
        t0 = time.perf_counter()
        await admin.update_weights(weight_dir)
        engine.on_new_version(step)
        logger.success(f"Weights updated to step {step} in {time.perf_counter() - t0:.2f}s ({weight_dir})")

    watcher = WeightWatcher(get_broadcast_dir(cfg.output_dir), on_new_weights)

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
