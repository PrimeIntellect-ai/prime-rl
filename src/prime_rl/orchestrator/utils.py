import asyncio
import ctypes
import gc
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import orjson

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.logger import InterceptHandler, get_logger, setup_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)


async def setup_policy_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Build the live policy inference pool + matching renderer. Returns
    ``(renderer, inference_pool)``.

    Training is renderer-only: the renderer object is the canonical
    messages → token ids path (sft backfill, opsd scoring prefixes, echo role
    attribution) and is always built. The renderer-client sampling path is
    wired onto the pool; when no train env samples from the live policy the
    renderer is still kept for client-side tokenization and the pool's evals
    use plain chat-completions."""
    from renderers.base import create_renderer

    client_config = config.model.client
    model_name = config.model.name
    renderer = create_renderer(tokenizer, config.renderer)
    get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
    if config.any_policy_sourced:
        get_logger().info("Using direct renderer rollout client")
    else:
        get_logger().info("No policy-sourced train env — renderer kept for client-side tokenization only")
    inference_pool = await setup_inference_pool(
        client_config,
        model_name=model_name,
        train_client_type="renderer",
        eval_client_type="openai_chat_completions",
        renderer_config=config.renderer,
        pool_size=config.pool_size,
    )
    return renderer, inference_pool


def save_rollouts(rollouts: list[dict], path: Path) -> None:
    """Append rollouts (Trace record dicts, already JSON-serializable) to a JSONL file.
    The trace streams are append-only: ``all`` grows one rollout at a time as they
    complete, ``effective`` one batch at a time on finalize."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    with open(path, "ab") as f:
        for rollout in rollouts:
            f.write(orjson.dumps(rollout, default=str, option=opts))


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False


def setup_env_server_logging(log_level: str, json_logging: bool = False) -> None:
    """Configure logging for an env-server process: prime-rl's logger + routing v1's stdlib
    logs through it. Passed to verifiers' ``serve_env`` so it runs in the broker and in every
    spawned worker — fresh ``spawn`` processes that otherwise have no handlers and would drop
    their per-rollout logs."""
    setup_logger(log_level, json_logging=json_logging)
    intercept_vf_logging(logger="verifiers.v1", level=log_level)


def set_default_executor(max_workers: int = 64) -> None:
    """Scale the default asyncio thread pool so asyncio.to_thread has enough capacity."""
    get_logger().info(f"Setting default executor to ThreadPoolExecutor(max_workers={max_workers})")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=max_workers))


def trim_process_memory() -> None:
    """Return freed heap pages to the OS on glibc systems."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as exc:
        get_logger().debug(f"malloc_trim(0) failed: {exc!r}")


def get_weight_dir(output_dir: Path, step: int, check_exists: bool = True, wait_timeout: int | None = None) -> Path:
    """Get the weight directory for a given checkpoint step.

    Args:
        output_dir: The output directory for the run.
        step: The checkpoint step.
        check_exists: If True, raises FileNotFoundError if no weight directory exists.
            If False, returns the broadcast directory path without checking existence
            (useful for NCCL mode where weights are broadcasted, not stored on disk).
        wait_timeout: Maximum time in seconds to wait for a stable directory to appear.
            If None, no waiting is performed.
    """
    ckpt_weight_dir = get_step_path(get_ckpt_dir(output_dir), step) / "weight"
    broadcast_weight_dir = get_step_path(get_broadcast_dir(output_dir), step)

    def find_stable_dir() -> Path | None:
        # For checkpoint weights, check STABLE file in parent directory (checkpoints/step_{step}/STABLE)
        ckpt_step_dir = get_step_path(get_ckpt_dir(output_dir), step)
        if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
            return ckpt_weight_dir

        # For broadcast weights, check STABLE file in the broadcast directory itself
        if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
            return broadcast_weight_dir

        return None

    # Check immediately, then wait if needed
    result = find_stable_dir()
    if result is None and wait_timeout:
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            time.sleep(1)
            result = find_stable_dir()
            if result:
                break

    if result:
        return result
    if not check_exists:
        return broadcast_weight_dir

    raise FileNotFoundError(f"No weight directory found for checkpoint step {step}")


def compute_pass_metrics(rewards: list[float]) -> dict[str, float]:
    """Unbiased pass@k and pass^k for one example's binary (0/1) rewards.

    pass@k = 1 - C(n-c, k) / C(n, k)  (at least one of k samples correct)
    pass^k = C(c, k) / C(n, k)        (all k samples correct)

    ``n`` = number of rewards, ``c`` = number correct, ``k`` = powers of 2 in [1, n].
    ``math.comb`` returns 0 when ``k`` exceeds its first argument, so the edge cases
    (``n - c < k`` → pass@k = 1; ``c < k`` → pass^k = 0) fall out without branching.
    """
    n = len(rewards)
    c = sum(1 for r in rewards if r == 1.0)
    out: dict[str, float] = {}
    k = 1
    while k <= n:
        n_choose_k = math.comb(n, k)
        out[f"pass@{k}"] = 1.0 - math.comb(n - c, k) / n_choose_k
        out[f"pass^{k}"] = math.comb(c, k) / n_choose_k
        k *= 2
    return out
