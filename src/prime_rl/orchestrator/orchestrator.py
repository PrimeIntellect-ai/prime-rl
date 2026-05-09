"""Async-pipelined RL orchestrator.

`run()` wires three long-lived coroutines: a metronome (`run_samplers`) that
calls Groups under a shared concurrency cap, a batcher that ships trainable
cohorts to the trainer, and a watcher that mutates the shared Policy when
fresh weights arrive. Setup helpers live in `utils.py` and `env_sampler.py`.
"""

import asyncio
import os

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.batcher import Done, setup_batcher
from prime_rl.orchestrator.ckpt import setup_ckpt_manager
from prime_rl.orchestrator.env_sampler import (
    EnvSampler,
    Policy,
    make_samplers_queue,
    run_samplers,
    setup_eval_sampler,
    setup_train_samplers,
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
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor


async def _start_samplers(samplers: list[EnvSampler]) -> None:
    """Start samplers serially. On failure, tear down whatever already started
    so we don't strand subprocess workers."""
    logger = get_logger()
    started: list[EnvSampler] = []
    try:
        for s in samplers:
            logger.info(f"Starting sampler {s.name!r}")
            await s.start()
            started.append(s)
    except BaseException:
        for s in reversed(started):
            try:
                await s.stop()
            except Exception as exc:
                logger.warning(f"[{s.name}] stop() during startup-rollback failed: {exc}")
        raise


async def _stop_samplers(samplers: list[EnvSampler]) -> None:
    logger = get_logger()
    for s in reversed(samplers):
        try:
            await s.stop()
        except Exception as exc:
            logger.warning(f"[{s.name}] stop() failed: {exc}")


async def run(config: OrchestratorConfig) -> None:
    logger = get_logger()
    logger.info(f"Output dir: {config.output_dir} | Model: {config.model.name}")

    install_envs(config)
    write_orch_config(config)
    tokenizer = setup_tokenizer(config.tokenizer)
    setup_monitor(
        wandb_config=config.wandb,
        output_dir=config.output_dir,
        tokenizer=tokenizer,
        run_config=config,
        prime_config=config.prime_monitor,
        keep_full_history=config.bench,
    )

    ckpt_manager = setup_ckpt_manager(config.output_dir, config.ckpt)
    resume_step = resolve_resume_step(config, ckpt_manager)
    lora_name = config.model.lora.name if config.model.lora else None

    # Shared mutable state. Watcher writes; EnvSamplers read. Initial
    # model_name is the base model since no LoRA adapter is loaded yet.
    policy = Policy(version=0, model_name=config.model.name)
    client = make_client(config)

    # Build EnvSamplers (no env loaded / no worker spawned yet — that happens in start()).
    dispatch_samplers, train_samplers = setup_train_samplers(config, client=client, policy=policy)
    eval_sampler = setup_eval_sampler(config, client=client, policy=policy, resume_step=resume_step)
    samplers: list[EnvSampler] = list(dispatch_samplers) + ([eval_sampler] if eval_sampler is not None else [])

    out_q, concurrency = make_samplers_queue(config, num_samplers=len(samplers))

    admin = setup_admin(config, lora_name=lora_name)
    batcher = setup_batcher(
        config,
        in_q=out_q,
        tokenizer=tokenizer,
        policy=policy,
        eval_sampler=eval_sampler,
        train_samplers=train_samplers,
        ckpt_manager=ckpt_manager,
        inference_metrics=InferenceMetricsCollector(admin.client) if config.collect_inference_metrics else None,
    )
    watcher = WeightWatcher(config, observers=[admin], policy=policy, lora_name=lora_name)

    await admin.start(config)
    await maybe_resume(
        config,
        resume_step,
        ckpt_manager,
        policy=policy,
        train_samplers=train_samplers,
        eval_sampler=eval_sampler,
        batcher=batcher,
        admin=admin,
        watcher=watcher,
    )

    # Spawn env workers + materialize datasets.
    await _start_samplers(samplers)
    try:
        logger.success("Orchestrator starting — producing rollouts")
        async with asyncio.TaskGroup() as tg:
            tg.create_task(run_samplers(samplers, out_q, concurrency))
            tg.create_task(batcher.run())
            tg.create_task(watcher.run())
            tg.create_task(admin.watch_health())
    except* Done:
        logger.success(f"Orchestrator finished: reached max_steps={config.max_steps}")
    finally:
        await _stop_samplers(samplers)


def main() -> None:
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")
    config = cli(OrchestratorConfig)
    setup_logger(
        log_level=config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
