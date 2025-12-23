import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import verifiers as vf
from openai import AsyncOpenAI

from prime_rl.utils.vf import from_serializable_state, to_serializable_state

# Main process globals
_process_pool: ProcessPoolExecutor | None = None


def get_process_pool(max_workers: int = 8) -> ProcessPoolExecutor:
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=max_workers)
    return _process_pool


def shutdown_process_pool():
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=True)
        _process_pool = None


# Worker process globals (each worker has its own copy)
_worker_envs: dict[str, vf.Environment] = {}


def _get_or_create_env(
    env_id: str,
    env_args: dict[str, Any],
    max_seq_len: int | None,
    interleaved_rollouts: bool,
) -> vf.Environment:
    """Get cached env or create new one. Cache is per-worker process."""
    # Create a hashable cache key from env config
    args_key = tuple(sorted((k, str(v)) for k, v in env_args.items()))
    cache_key = f"{env_id}:{args_key}:{max_seq_len}:{interleaved_rollouts}"

    if cache_key not in _worker_envs:
        env = vf.load_environment(env_id, **env_args)
        if max_seq_len:
            env.set_max_seq_len(max_seq_len)
        if interleaved_rollouts:
            env.set_interleaved_rollouts(True)
        _worker_envs[cache_key] = env

    return _worker_envs[cache_key]


@dataclass
class RolloutRequest:
    """Picklable request for subprocess."""

    env_id: str
    env_args: dict[str, Any]
    model_name: str
    example: dict
    rollouts_per_example: int
    sampling_args: dict
    base_url: str
    api_key: str
    max_seq_len: int | None = None
    interleaved_rollouts: bool = False


def _run_group_in_subprocess(request: RolloutRequest) -> list[dict]:
    """Runs in separate process with its own event loop."""

    async def _generate():
        client = AsyncOpenAI(base_url=request.base_url, api_key=request.api_key)
        env = _get_or_create_env(
            request.env_id,
            request.env_args,
            request.max_seq_len,
            request.interleaved_rollouts,
        )

        group_inputs = [vf.RolloutInput(**request.example) for _ in range(request.rollouts_per_example)]

        states = await env.run_group(
            group_inputs=group_inputs,
            client=client,
            model=request.model_name,
            gen_sampling_args=request.sampling_args,
        )

        return [to_serializable_state(s) for s in states]

    return asyncio.run(_generate())


async def run_group_in_process(
    request: RolloutRequest,
    pool: ProcessPoolExecutor | None = None,
) -> list[vf.State]:
    """Submit rollout to process pool, await result."""
    loop = asyncio.get_event_loop()
    pool = pool or get_process_pool()

    serialized = await loop.run_in_executor(pool, _run_group_in_subprocess, request)

    return [from_serializable_state(s) for s in serialized]
