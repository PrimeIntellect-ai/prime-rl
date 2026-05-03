"""Async-pipelined RL orchestrator.

The interesting story lives in `run()`: five long-lived coroutines
(scheduler, admin, engine, batcher, watcher) cooperating through one
asyncio.Queue and a weight-rotation observer chain. Setup helpers below
the main flow handle config translation, env install, resume — fluf so
`run()` stays narrative.
"""

import asyncio
import os

import tomli_w
import verifiers as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.batcher import Done, setup_batcher
from prime_rl.orchestrator.buffer import setup_buffer
from prime_rl.orchestrator.ckpt import CkptManager, setup_ckpt_manager
from prime_rl.orchestrator.engine import Group, setup_rollout_engine
from prime_rl.orchestrator.inference_admin import setup_admin
from prime_rl.orchestrator.inference_metrics import InferenceMetricsCollector
from prime_rl.orchestrator.scheduler import Scheduler, setup_scheduler
from prime_rl.orchestrator.watcher import setup_watcher
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.monitor import setup_monitor
from prime_rl.utils.utils import get_env_ids_to_install, install_env


async def run(cfg: OrchestratorConfig) -> None:
    _validate(cfg)
    logger = get_logger()
    logger.info(f"Output dir: {cfg.output_dir} | Model: {cfg.model.name}")

    _install_envs(cfg)
    _write_orch_config(cfg)
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
    resume_step = _resolve_resume_step(cfg, ckpt_manager)
    buffer = setup_buffer(cfg.buffer, [e.resolved_name for e in cfg.train.env], seed=cfg.seed)
    lora_name = cfg.model.lora.name if cfg.model.lora else None

    # ── The five long-lived coroutines ──
    scheduler = setup_scheduler(cfg, buffer=buffer, resume_step=resume_step)
    admin = setup_admin(cfg, lora_name=lora_name)
    groups_q, concurrency = _make_groups_queue(cfg, scheduler)
    engine = setup_rollout_engine(
        cfg,
        scheduler=scheduler,
        out_q=groups_q,
        client=_make_client(cfg),
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
    await _maybe_resume(cfg, resume_step, ckpt_manager, scheduler, engine, batcher, admin, watcher, buffer)

    logger.success("Orchestrator starting — producing rollouts")
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(engine.run())
            tg.create_task(batcher.run())
            tg.create_task(watcher.run())
            tg.create_task(admin.watch_health())
    except* Done:
        logger.success(f"Orchestrator finished: reached max_steps={cfg.max_steps}")


# ───── Setup helpers (config translation, validation, resume) ─────


def _validate(cfg: OrchestratorConfig) -> None:
    """Top-level invariants. Things the new orch doesn't yet support raise
    NotImplementedError up front rather than half-running."""
    assert cfg.max_inflight_rollouts is not None
    assert cfg.advantage is not None, "advantage config required"
    assert len(cfg.client.base_url) == 1, "single inference endpoint only"
    if cfg.use_renderer:
        raise NotImplementedError(
            "orchestrator.use_renderer is not yet supported by the new async orchestrator. "
            "Set use_renderer=false (use_token_client=true for TITO, both false for MITO)."
        )
    if cfg.teacher_rollout_model is not None:
        raise NotImplementedError(
            "orchestrator.teacher_rollout_model is not yet supported by the new async orchestrator. "
            "Remove the teacher_rollout_model field to proceed."
        )
    if cfg.model.lora and cfg.weight_broadcast.type == "nccl":
        raise ValueError("NCCL weight broadcast does not support LoRA — use filesystem broadcast")


def _install_envs(cfg: OrchestratorConfig) -> None:
    env_ids = set(get_env_ids_to_install(cfg.train.env))
    if cfg.eval is not None:
        env_ids.update(get_env_ids_to_install(cfg.eval.env))
    for env_id in env_ids:
        install_env(env_id, prerelease=cfg.env_install_prerelease)


def _write_orch_config(cfg: OrchestratorConfig) -> None:
    """Trainer reads this from output_dir/control/orch.toml at startup
    (see prime_rl.trainer.runs.RunManager.get_orchestrator_config)."""
    control_dir = cfg.output_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(cfg.model_dump(exclude_none=True, mode="json"), f)


def _resolve_resume_step(cfg: OrchestratorConfig, ckpt_manager: CkptManager | None) -> int | None:
    if not (cfg.ckpt and cfg.ckpt.resume_step is not None and ckpt_manager is not None):
        return None
    if cfg.ckpt.resume_step != -1:
        return cfg.ckpt.resume_step
    latest = ckpt_manager.latest_step()
    if latest is None:
        get_logger().warning("ckpt.resume_step=-1 set but no orch checkpoints found; starting fresh")
    return latest


def _make_groups_queue(cfg: OrchestratorConfig, scheduler: Scheduler) -> tuple[asyncio.Queue[Group], int]:
    """Engine concurrency cap + bounded queue between engine and batcher.
    Concurrency is in *groups*; max_inflight_rollouts is the legacy per-rollout
    figure, so divide. Queue is sized so the batcher's async-level barrier
    cascades backpressure into the engine's semaphore (rather than letting
    in-flight rollouts pile up unbounded)."""
    concurrency = max(1, cfg.max_inflight_rollouts // cfg.rollouts_per_example)
    get_logger().info(f"Engine concurrency: {concurrency} groups across {len(scheduler.tasks)} task(s)")
    return asyncio.Queue(maxsize=concurrency * (cfg.max_async_level + 1)), concurrency


def _make_client(cfg: OrchestratorConfig) -> vf.ClientConfig:
    """Verifiers ClientConfig for the rollout engine. TITO (token-in-token-out)
    bypasses server-side chat templating — only safe for linear-history envs."""
    client_type = "openai_chat_completions_token" if cfg.use_token_client else "openai_chat_completions"
    if cfg.use_token_client:
        get_logger().warning(
            "Token-in-token-out (TITO) client is enabled. Only use this if your environment has a "
            "linear history and the chat template has the extension property."
        )
    return vf.ClientConfig(
        client_type=client_type,
        api_base_url=cfg.client.base_url[0],
        api_key_var=cfg.client.api_key_var,
        timeout=cfg.client.timeout,
        connect_timeout=cfg.client.connect_timeout,
    )


async def _maybe_resume(
    cfg: OrchestratorConfig,
    resume_step: int | None,
    ckpt_manager: CkptManager | None,
    scheduler,
    engine,
    batcher,
    admin,
    watcher,
    buffer,
) -> None:
    """Restore state from the last orch checkpoint and prime each component
    with the resumed step so rollouts produced after this point are tagged
    correctly."""
    if resume_step is None or ckpt_manager is None:
        return
    state = ckpt_manager.load(resume_step)
    batcher.step = state.step
    scheduler.last_eval_step = state.last_eval_step
    if buffer is not None and state.buffer_state and not (cfg.ckpt and cfg.ckpt.skip_buffer):
        buffer.load_state_dict(state.buffer_state)
    if cfg.eval and cfg.eval.skip_eval_on_resume:
        # bump last_eval_step past current so the next interval boundary is
        # the first eval the resumed run sees
        scheduler.last_eval_step = state.step
        get_logger().info(f"Skipping next eval on resume (last_eval_step={state.step})")
    await admin.on_new_version(state.step)
    await engine.on_new_version(state.step)
    watcher.current_step = state.step
    get_logger().success(f"Resumed orch from step {state.step} (eval cursor at {scheduler.last_eval_step})")


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
