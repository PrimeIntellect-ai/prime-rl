"""Setup helpers for the orchestrator entrypoint.

These are the cross-cutting concerns kept out of `orchestrator.run()`:
config dump, env install, resume restoration, queue sizing, client mode
selection. Pure translation — no behavior beyond what the comments in
each helper describe."""

import asyncio

import tomli_w
import verifiers as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.ckpt import CkptManager
from prime_rl.orchestrator.engine import Group, RolloutEngine
from prime_rl.orchestrator.inference_admin import InferenceAdmin
from prime_rl.orchestrator.scheduler import Scheduler
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_env_ids_to_install, install_env


def install_envs(cfg: OrchestratorConfig) -> None:
    """Install all train + eval verifiers envs referenced by the config."""
    env_ids = set(get_env_ids_to_install(cfg.train.env))
    if cfg.eval is not None:
        env_ids.update(get_env_ids_to_install(cfg.eval.env))
    for env_id in env_ids:
        install_env(env_id, prerelease=cfg.env_install_prerelease)


def write_orch_config(cfg: OrchestratorConfig) -> None:
    """Trainer reads this from `output_dir/control/orch.toml` at startup
    (see prime_rl.trainer.runs.RunManager.get_orchestrator_config)."""
    control_dir = cfg.output_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(cfg.model_dump(exclude_none=True, mode="json"), f)


def resolve_resume_step(cfg: OrchestratorConfig, ckpt_manager: CkptManager | None) -> int | None:
    """Resolve `cfg.ckpt.resume_step` against on-disk checkpoints.
    `-1` → latest available; missing checkpoints → log a warning + None
    (start fresh)."""
    if not (cfg.ckpt and cfg.ckpt.resume_step is not None and ckpt_manager is not None):
        return None
    if cfg.ckpt.resume_step != -1:
        return cfg.ckpt.resume_step
    latest = ckpt_manager.latest_step()
    if latest is None:
        get_logger().warning("ckpt.resume_step=-1 set but no orch checkpoints found; starting fresh")
    return latest


def make_groups_queue(cfg: OrchestratorConfig, scheduler: Scheduler) -> tuple[asyncio.Queue[Group], int]:
    """Engine concurrency cap + bounded queue between engine and batcher.
    Concurrency is in *groups*; max_inflight_rollouts is the legacy per-rollout
    figure, so divide. Queue is sized so the batcher's async-level barrier
    cascades backpressure into the engine's semaphore (rather than letting
    in-flight rollouts pile up unbounded)."""
    assert cfg.max_inflight_rollouts is not None
    concurrency = max(1, cfg.max_inflight_rollouts // cfg.rollouts_per_example)
    get_logger().info(f"Engine concurrency: {concurrency} groups across {len(scheduler.tasks)} task(s)")
    return asyncio.Queue(maxsize=concurrency * (cfg.max_async_level + 1)), concurrency


def make_client(cfg: OrchestratorConfig) -> vf.ClientConfig:
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


async def maybe_resume(
    cfg: OrchestratorConfig,
    resume_step: int | None,
    ckpt_manager: CkptManager | None,
    *,
    scheduler: Scheduler,
    engine: RolloutEngine,
    batcher,
    admin: InferenceAdmin,
    watcher: WeightWatcher,
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
