from __future__ import annotations

import asyncio
import ctypes
import logging
import math
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, NamedTuple

import verifiers.v1 as vf

from prime_rl.transports.batch.types import UNLABELED
from prime_rl.utils.logger import InterceptHandler, get_logger, setup_logger

if TYPE_CHECKING:
    from prime_rl.env.heads import EpisodeHead


def episode_env_name(episode: EpisodeHead) -> str:
    name = episode.env.name
    if name is None:
        raise ValueError("Orchestrated episode is missing its environment name")
    return name


def episode_group_id(episode: EpisodeHead) -> str:
    group = episode.group
    if group is None:
        raise ValueError("Orchestrated episode is missing its rollout group")
    return group.id


def train_work(episode: EpisodeHead) -> vf.TrainWorkInfo:
    run = episode.run
    if not isinstance(run, vf.TrainRunInfo) or not isinstance(run.work, vf.TrainWorkInfo):
        raise ValueError("Train episode is missing training-work provenance")
    return run.work


def eval_work(episode: EpisodeHead) -> vf.EvalWorkInfo:
    run = episode.run
    if not isinstance(run, vf.TrainRunInfo) or not isinstance(run.work, vf.EvalWorkInfo):
        raise ValueError("Eval episode is missing evaluation-work provenance")
    return run.work


def min_fresh_version(step: int, max_staleness: int) -> int:
    """Oldest policy version whose tokens may still train in batch ``step`` —
    anything older would ship past ``max_staleness``."""
    return (step - 1) - max_staleness


class Staleness(NamedTuple):
    """Staleness of one train rollout in batch ``training_step``.

    A token's staleness is the version the batch trains on (v{step-1}) minus
    the policy version that sampled it (``TraceHead.versions``). A
    rollout's staleness is the max staleness of its tokens. ``in_flight`` is
    its share from weight updates between the oldest token and completion;
    ``in_queue`` is its share from weight updates while it waited to ship.
    ``token_sum`` and ``token_count`` aggregate token staleness over the
    rollout's sampled tokens."""

    rollout: int
    in_flight: int
    in_queue: int
    token_sum: int
    token_count: int


def rollout_staleness(episode: EpisodeHead, training_step: int) -> Staleness | None:
    """Staleness of a train episode in batch ``training_step``, or None for
    frozen-sourced episodes (no policy span), which never go stale. Tokens
    without a policy label take the episode's dispatch version, which no
    token of the episode predates."""
    policy = train_work(episode).policy
    if policy is None:
        return None
    trained_version = training_step - 1
    rollout = token_sum = token_count = 0
    for trace in episode.traces:
        for version, count in trace.versions.items():
            staleness = max(0, trained_version - (policy.start if version == UNLABELED else version))
            rollout = max(rollout, staleness)
            token_sum += staleness * count
            token_count += count
    in_queue = min(rollout, max(0, trained_version - policy.end))
    return Staleness(rollout, rollout - in_queue, in_queue, token_sum, token_count)


def max_rollout_staleness(episodes: Iterable[EpisodeHead], training_step: int) -> int:
    staleness = (rollout_staleness(episode, training_step) for episode in episodes)
    return max((s.rollout for s in staleness if s is not None), default=0)


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False


def setup_env_worker_pool_logging(log_level: str, json_logging: bool = False) -> None:
    """Configure logging for an env-worker-pool process: prime-rl's logger + routing v1's stdlib
    logs through it. Passed to ``run_env_worker_pool`` so it runs in the broker and in every spawned
    worker — fresh ``spawn`` processes that otherwise have no handlers and would drop their
    per-rollout logs. The launcher redirects an env worker pool's stdout to its log file, so
    the console sink colorizes only on a terminal."""
    setup_logger(log_level, json_logging=json_logging, colorize=sys.stdout.isatty())
    intercept_vf_logging(logger="verifiers.v1", level=log_level)


def set_default_executor(max_workers: int = 64) -> None:
    """Scale the default asyncio thread pool so asyncio.to_thread has enough capacity."""
    get_logger().debug(f"Setting default executor to ThreadPoolExecutor(max_workers={max_workers})")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=max_workers))


def release_and_trim(*containers: list) -> None:
    """Release shipped lists incrementally, then trim process memory."""
    for items in containers:
        while items:
            items.pop()
    trim_process_memory()


def trim_process_memory() -> None:
    """Return freed heap pages to the OS on glibc systems."""
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
