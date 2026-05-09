"""Async-pipelined RL orchestrator.

`run()` wires three long-lived coroutines: a metronome (`run_groups`) that
calls Groups under a shared concurrency cap, a batcher that ships trainable
cohorts to the trainer, and a watcher that mutates the shared Policy when
fresh weights arrive. Setup helpers live in `utils.py` and `group.py`.
"""

import asyncio
import os

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.batcher import Done, setup_batcher
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.group import (
    Group,
    Policy,
    make_groups_queue,
    run_groups,
    setup_eval_group,
    setup_train_groups,
)
from prime_rl.orchestrator.inference_admin import setup_admin
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.utils import (
    install_envs,
    make_client,
    maybe_resume,
    resolve_resume_step,
    write_orch_config,
)
from prime_rl.orchestrator.watcher import setup_watcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor


async def _start_groups(groups: list[Group]) -> None:
    """Start groups serially. On failure, tear down whatever already started
    so we don't strand subprocess workers."""
    logger = get_logger()
    started: list[Group] = []
    try:
        for g in groups:
            logger.info(f"Starting group {g.name!r}")
            await g.start()
            started.append(g)
    except BaseException:
        for g in reversed(started):
            try:
                await g.stop()
            except Exception as exc:
                logger.warning(f"[{g.name}] stop() during startup-rollback failed: {exc}")
        raise


async def _stop_groups(groups: list[Group]) -> None:
    logger = get_logger()
    for g in reversed(groups):
        try:
            await g.stop()
        except Exception as exc:
            logger.warning(f"[{g.name}] stop() failed: {exc}")


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
    lora_name = cfg.model.lora.name if cfg.model.lora else None

    # Shared mutable state. Watcher writes; Groups read. Initial model_name
    # is the base model since no LoRA adapter is loaded yet.
    policy = Policy(version=0, model_name=cfg.model.name)
    client = make_client(cfg)

    # Build Groups (no env loaded / no worker spawned yet — that happens in start()).
    dispatch_groups, train_groups = setup_train_groups(cfg, client=client, policy=policy)
    eval_group = setup_eval_group(cfg, client=client, policy=policy, resume_step=resume_step)
    groups: list[Group] = list(dispatch_groups) + ([eval_group] if eval_group is not None else [])

    out_q, concurrency = make_groups_queue(cfg, num_groups=len(groups))

    admin = setup_admin(cfg, lora_name=lora_name)
    batcher = setup_batcher(
        cfg,
        in_q=out_q,
        tokenizer=tokenizer,
        policy=policy,
        eval_group=eval_group,
        train_groups=train_groups,
        ckpt_manager=ckpt_manager,
        inference_metrics=InferenceMetricsCollector(admin.client) if cfg.collect_inference_metrics else None,
    )
    watcher = setup_watcher(cfg, observers=[admin], policy=policy, lora_name=lora_name)

    await admin.start(cfg)
    await maybe_resume(
        cfg,
        resume_step,
        ckpt_manager,
        policy=policy,
        train_groups=train_groups,
        eval_group=eval_group,
        batcher=batcher,
        admin=admin,
        watcher=watcher,
    )

    # Spawn env workers + materialize datasets.
    await _start_groups(groups)
    try:
        logger.success("Orchestrator starting — producing rollouts")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(run_groups(groups, out_q, concurrency))
            tg.create_task(batcher.run())
            tg.create_task(watcher.run())
            tg.create_task(admin.watch_health())
    except* Done:
        logger.success(f"Orchestrator finished: reached max_steps={cfg.max_steps}")
    finally:
        await _stop_groups(groups)


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
