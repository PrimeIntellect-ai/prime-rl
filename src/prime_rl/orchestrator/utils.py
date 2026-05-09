"""Setup helpers for the orchestrator entrypoint.

These are the cross-cutting concerns kept out of `orchestrator.run()`:
config dump, env install, resume restoration, client mode selection. Pure
translation — no behavior beyond what the comments in each helper describe."""

import tomli_w
import verifiers as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.ckpt import CkptManager
from prime_rl.orchestrator.env_sampler import EvalEnvSampler, GRPOEnvSampler, Policy
from prime_rl.orchestrator.inference_admin import InferenceAdmin
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_env_ids_to_install, install_env


def install_envs(config: OrchestratorConfig) -> None:
    """Install all train + eval verifiers envs referenced by the config."""
    env_ids = set(get_env_ids_to_install(config.train.env))
    if config.eval is not None:
        env_ids.update(get_env_ids_to_install(config.eval.env))
    for env_id in env_ids:
        install_env(env_id, prerelease=config.env_install_prerelease)


def write_orch_config(config: OrchestratorConfig) -> None:
    """Trainer reads this from `output_dir/control/orch.toml` at startup
    (see prime_rl.trainer.runs.RunManager.get_orchestrator_config)."""
    control_dir = config.output_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(config.model_dump(exclude_none=True, mode="json"), f)


def resolve_resume_step(config: OrchestratorConfig, ckpt_manager: CkptManager | None) -> int | None:
    """Resolve `config.ckpt.resume_step` against on-disk checkpoints."""
    if not (config.ckpt and config.ckpt.resume_step is not None and ckpt_manager is not None):
        return None
    if config.ckpt.resume_step != -1:
        return config.ckpt.resume_step
    latest = ckpt_manager.latest_step()
    if latest is None:
        get_logger().warning("ckpt.resume_step=-1 set but no orch checkpoints found; starting fresh")
    return latest


def make_client(config: OrchestratorConfig) -> vf.ClientConfig:
    """Verifiers ClientConfig for rollouts. TITO bypasses server-side chat
    templating — only safe for linear-history envs."""
    client_type = "openai_chat_completions_token" if config.use_token_client else "openai_chat_completions"
    if config.use_token_client:
        get_logger().warning(
            "Token-in-token-out (TITO) client is enabled. Only use this if your environment has a "
            "linear history and the chat template has the extension property."
        )
    return vf.ClientConfig(
        client_type=client_type,
        api_base_url=config.client.base_url[0],
        api_key_var=config.client.api_key_var,
        timeout=config.client.timeout,
        connect_timeout=config.client.connect_timeout,
    )


async def maybe_resume(
    config: OrchestratorConfig,
    resume_step: int | None,
    ckpt_manager: CkptManager | None,
    *,
    policy: Policy,
    train_samplers: list[GRPOEnvSampler],
    eval_sampler: EvalEnvSampler | None,
    batcher,
    admin: InferenceAdmin,
    watcher: WeightWatcher,
) -> None:
    """Restore state from the last orch checkpoint and prime each component
    with the resumed step."""
    if resume_step is None or ckpt_manager is None:
        return
    state = ckpt_manager.load(resume_step)
    batcher.step = state.step
    if eval_sampler is not None:
        eval_sampler.last_eval_step = state.last_eval_step
    if not (config.ckpt and config.ckpt.skip_buffer):
        for g in train_samplers:
            g.load_state_dict(state.sampler_states.get(g.name, {}))
    if config.eval and config.eval.skip_eval_on_resume and eval_sampler is not None:
        eval_sampler.last_eval_step = state.step
        get_logger().info(f"Skipping next eval on resume (last_eval_step={state.step})")
    await admin.on_new_version(state.step)
    policy.version = state.step
    if watcher.lora_name and policy.model_name != watcher.lora_name:
        policy.model_name = watcher.lora_name
    watcher.current_step = state.step
    last_eval = eval_sampler.last_eval_step if eval_sampler is not None else 0
    get_logger().success(f"Resumed orch from step {state.step} (eval cursor at {last_eval})")
