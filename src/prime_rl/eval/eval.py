import asyncio
import time
from pathlib import Path

import numpy as np
import verifiers as vf
from loguru import logger

from prime_rl.eval.config import EvalEnvConfig, OfflineEvalConfig
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import get_monitor, setup_monitor
from prime_rl.utils.pydantic_config import parse_argv
from prime_rl.utils.utils import clean_exit, install_env


def get_sampling_args(config: OfflineEvalConfig) -> dict:
    """Build sampling args dict from config."""
    return config.sampling.to_dict()


async def run_eval(
    env: vf.Environment,
    env_config: EvalEnvConfig,
    config: OfflineEvalConfig,
    ckpt_step: int | None = None,
) -> dict:
    """Run evaluation on a single environment and return metrics."""
    env_name = env_config.name or env_config.env_id
    num_examples = env_config.num_examples or config.num_examples
    rollouts_per_example = env_config.rollouts_per_example or config.rollouts_per_example

    logger.info(f"Evaluating {env_name} ({num_examples=}, {rollouts_per_example=})")

    # Build client config for verifiers
    base_urls = config.client.base_url
    vf_client = vf.ClientConfig(
        api_base_url=base_urls[0],  # verifiers only supports single URL
        api_key_var=config.client.api_key_var,
        timeout=float(config.client.timeout),
        extra_headers=config.client.headers,
    )

    sampling_args = get_sampling_args(config)
    eval_start = time.perf_counter()

    outputs = await env.evaluate(
        client=vf_client,
        model=config.model.name,
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=config.max_concurrent,
        use_tqdm=True,
    )

    eval_time = time.perf_counter() - eval_start

    # Extract metrics
    rewards = [o["reward"] for o in outputs["outputs"]]
    completion_lens = []
    truncated_count = 0

    for o in outputs["outputs"]:
        if o.get("completion"):
            completion_lens.append(len(o["completion"]))
        if o.get("is_truncated"):
            truncated_count += 1

    avg_reward = np.mean(rewards) if rewards else 0.0
    std_reward = np.std(rewards) if rewards else 0.0
    avg_completion_len = np.mean(completion_lens) if completion_lens else 0.0
    truncated_pct = (truncated_count / len(outputs["outputs"]) * 100) if outputs["outputs"] else 0.0

    logger.success(
        f"Evaluated {env_name} in {eval_time:.2f}s "
        f"(Avg: {avg_reward:.4f}, Std: {std_reward:.4f}, "
        f"Completion Len: {avg_completion_len:.1f}, Truncated: {truncated_pct:.1f}%)"
    )

    # Build metrics dict
    metrics = {
        f"eval/{env_name}/avg": avg_reward,
        f"eval/{env_name}/std": std_reward,
        f"eval/{env_name}/completion_len": avg_completion_len,
        f"eval/{env_name}/truncated_pct": truncated_pct,
        f"eval/{env_name}/time": eval_time,
    }

    if ckpt_step is not None:
        metrics["progress/ckpt_step"] = ckpt_step

    return metrics


async def run_evals(
    envs: list[vf.Environment],
    env_configs: list[EvalEnvConfig],
    config: OfflineEvalConfig,
    ckpt_step: int | None = None,
) -> None:
    """Run evaluations on all environments and log to monitor."""
    all_metrics = {}

    for env, env_config in zip(envs, env_configs):
        metrics = await run_eval(env, env_config, config, ckpt_step)
        all_metrics.update(metrics)

    # Log all metrics
    monitor = get_monitor()
    monitor.log(all_metrics, step=None)


def get_stable_checkpoints(weights_dir: Path) -> list[int]:
    """Get sorted list of checkpoint steps that have STABLE marker."""
    ckpt_steps = []
    for step_path in weights_dir.glob("step_*"):
        if (step_path / "STABLE").exists():
            try:
                step = int(step_path.name.split("_")[-1])
                ckpt_steps.append(step)
            except ValueError:
                continue
    return sorted(ckpt_steps)


@clean_exit
async def eval(config: OfflineEvalConfig):
    """Main evaluation function."""
    # Setup logger
    setup_logger(
        config.log.level,
        log_file=config.output_dir / "logs" / "eval.log" if config.log.file else None,
        json_logging=config.log.json_logging,
    )
    logger.info("Starting offline evaluation")

    # Setup monitor (wandb)
    setup_monitor(
        wandb_config=config.wandb,
        output_dir=config.output_dir,
        run_config=config,
    )

    # Install environments
    for env_config in config.env:
        install_env(env_config.env_id)

    # Load environments
    logger.info(f"Loading {len(config.env)} environment(s)")
    envs: list[vf.Environment] = []
    for env_config in config.env:
        env = vf.load_environment(env_config.env_id, **env_config.env_args)
        await env.start_server()
        envs.append(env)

    # Setup inference pool if watcher mode (for weight reloading)
    inference_pool = None
    if config.watcher.enabled:
        if config.watcher.weights_dir is None:
            raise ValueError("weights_dir is required when watcher is enabled")
        inference_pool = await setup_inference_pool(config.client, model_name=config.model.name)
        await inference_pool.wait_for_ready(config.model.name)

    try:
        if not config.watcher.enabled:
            # Single evaluation run
            await run_evals(envs, config.env, config)
        else:
            # Watcher mode
            weights_dir = config.watcher.weights_dir
            assert weights_dir is not None

            logger.info(f"Watching {weights_dir} for new checkpoints")
            max_evaluated_step = -1

            while True:
                ckpt_steps = get_stable_checkpoints(weights_dir)
                new_steps = [s for s in ckpt_steps if s > max_evaluated_step]

                if new_steps:
                    logger.info(f"New checkpoints to evaluate: {new_steps}")
                    for ckpt_step in new_steps:
                        # Reload weights
                        weights_path = weights_dir / f"step_{ckpt_step}"
                        logger.info(f"Reloading weights from {weights_path}")

                        assert inference_pool is not None
                        await inference_pool.update_weights(weights_path, step=ckpt_step)

                        # Run evals
                        await run_evals(envs, config.env, config, ckpt_step)
                        max_evaluated_step = ckpt_step
                else:
                    logger.info(f"No new checkpoints, waiting {config.watcher.poll_interval}s")
                    await asyncio.sleep(config.watcher.poll_interval)

    finally:
        # Cleanup
        for env in envs:
            await env.stop_server()


def main():
    config = parse_argv(OfflineEvalConfig)
    asyncio.run(eval(config))


if __name__ == "__main__":
    main()
