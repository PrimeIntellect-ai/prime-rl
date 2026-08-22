import asyncio
import ctypes
import gc
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import verifiers.v1 as vf

from prime_rl.utils.logger import InterceptHandler, get_logger, setup_logger


def episode_env_name(episode: vf.Episode[Any, Any, Any]) -> str:
    name = episode.env.name
    if name is None:
        raise ValueError("Orchestrated episode is missing its environment name")
    return name


def episode_group_id(episode: vf.Episode[Any, Any, Any]) -> str:
    group = episode.group
    if group is None:
        raise ValueError("Orchestrated episode is missing its rollout group")
    return group.id


def train_work(episode: vf.Episode[Any, Any, Any]) -> vf.TrainWorkInfo:
    run = episode.run
    if not isinstance(run, vf.TrainRunInfo) or not isinstance(run.work, vf.TrainWorkInfo):
        raise ValueError("Train episode is missing training-work provenance")
    return run.work


def eval_work(episode: vf.Episode[Any, Any, Any]) -> vf.EvalWorkInfo:
    run = episode.run
    if not isinstance(run, vf.TrainRunInfo) or not isinstance(run.work, vf.EvalWorkInfo):
        raise ValueError("Eval episode is missing evaluation-work provenance")
    return run.work


def min_fresh_version(step: int, max_off_policy_steps: int) -> int:
    """Oldest dispatch version whose episodes may still train in batch
    ``step`` — anything older would ship past ``max_off_policy_steps``."""
    return (step - 1) - max_off_policy_steps


def episode_staleness(episode: vf.Episode[Any, Any, Any], training_step: int) -> tuple[int, int, int]:
    """``(total, in_flight, in_queue)`` staleness of one train episode when
    consumed by batch ``training_step``: the version the batch trains on
    (v{step-1}) minus the version that generated the episode. ``in_flight``
    is the span's share (weight updates during generation); ``in_queue`` is
    time spent buffered between completion and ship. Frozen-sourced episodes
    (no policy span) are never stale."""
    policy = train_work(episode).policy
    if policy is None:
        return 0, 0, 0
    total = max(0, (training_step - 1) - policy.start)
    in_flight = min(total, max(0, policy.end - policy.start))
    in_queue = total - in_flight
    return total, in_flight, in_queue


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
