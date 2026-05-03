"""Async-pipelined RL orchestrator.

The interesting story lives in `run()`: five long-lived coroutines
(scheduler, admin, engine, batcher, watcher) cooperating through one
asyncio.Queue and a weight-rotation observer chain. Setup helpers live
in `utils.py`; config invariants live in `OrchestratorConfig` validators.
"""

import asyncio
import os

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.batcher import Done, setup_batcher
from prime_rl.orchestrator.buffer import setup_buffer
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.engine import setup_rollout_engine
from prime_rl.orchestrator.inference_admin import setup_admin
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.scheduler import setup_scheduler
from prime_rl.orchestrator.utils import (
    install_envs,
    make_client,
    make_groups_queue,
    maybe_resume,
    resolve_resume_step,
    write_orch_config,
)
from prime_rl.orchestrator.watcher import setup_watcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor


async def run(cfg: OrchestratorConfig) -> None:
    logger = get_logger()
    logger.info(f"Output dir: {cfg.output_dir} | Model: {cfg.model.name}")

    install_envs(cfg)
    write_orch_config(cfg)
    tokenizer = setup_tokenizer(cfg.tokenizer)
    setup_monitor(
        wandb_config=cfg.wandb,
        output_dir=cfg.output_dir,
        tokenizer=tokenizer,
        run_config=cfg,
        prime_config=cfg.prime_monitor,
        keep_full_history=cfg.bench,
    )

    ckpt_manager = setup_ckpt_manager(cfg.output_dir, cfg.ckpt)
    resume_step = resolve_resume_step(cfg, ckpt_manager)
    buffer = setup_buffer(cfg.buffer, [e.resolved_name for e in cfg.train.env], seed=cfg.seed)
    lora_name = cfg.model.lora.name if cfg.model.lora else None

    # ── The five long-lived coroutines ──
    scheduler = setup_scheduler(cfg, buffer=buffer, resume_step=resume_step)
    admin = setup_admin(cfg, lora_name=lora_name)
    groups_q, concurrency = make_groups_queue(cfg, scheduler)
    engine = setup_rollout_engine(
        cfg,
        scheduler=scheduler,
        out_q=groups_q,
        client=make_client(cfg),
        concurrency=concurrency,
        lora_name=lora_name,
    )
    batcher = setup_batcher(
        cfg,
        in_q=groups_q,
        tokenizer=tokenizer,
        policy=engine,
        eval_counter=scheduler,
        ckpt_manager=ckpt_manager,
        buffer=buffer,
        inference_metrics=InferenceMetricsCollector(admin.client) if cfg.collect_inference_metrics else None,
    )
    watcher = setup_watcher(cfg, observers=[admin, engine, scheduler])

    await admin.start(cfg)
    await maybe_resume(
        cfg,
        resume_step,
        ckpt_manager,
        scheduler=scheduler,
        engine=engine,
        batcher=batcher,
        admin=admin,
        watcher=watcher,
        buffer=buffer,
    )

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
