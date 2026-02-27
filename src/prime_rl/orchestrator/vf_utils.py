import asyncio
import logging
import multiprocessing as mp
from collections.abc import Awaitable, Callable
from itertools import cycle
from typing import Any

import verifiers as vf
from verifiers.envs.environment import EnvClient
from verifiers.utils.worker_utils import get_free_port_pair
from verifiers.workers import ZMQEnvClient, ZMQEnvServer

from prime_rl.utils.logger import InterceptHandler, ProgressTracker

DEFAULT_RETRIES = 0
REQUIRED_STATE_COLUMNS = ["trajectory", "sampling_args"]
DEFAULT_STATE_COLUMNS = []


def spawn_env_server(
    env_id: str,
    env_args: dict[str, Any],
    extra_env_kwargs: dict[str, Any],
    address: str | None = None,
    # logging configs
    log_level: str | None = None,
    log_file: str | None = None,
    log_file_level: str | None = None,
    json_logging: bool = False,
) -> tuple[str, mp.Process]:
    """
    Starts a ZMQEnvServer process in a subprocess.

    Mirrors vf.Environment.start_server().
    """
    address = address or f"tcp://127.0.0.1:{get_free_port_pair()}"
    # Use spawn to avoid inheriting file descriptors (e.g. sockets) from
    # the parent process, which has caused hangs when multiple env server
    # subprocesses share the same fds.
    process = mp.get_context("spawn").Process(
        target=ZMQEnvServer.run_server,
        args=(
            env_id,
            env_args,
            extra_env_kwargs,
            log_level,
            log_file,
            log_file_level,
        ),
        kwargs=dict(address=address, json_logging=json_logging),
        daemon=False,  # cannot run daemon because env server uses subprocesses
    )
    process.start()

    return address, process


def setup_env_client(
    address: str,
    name: str | None = None,
    # health check configs
    health_check_interval: float = 5.0,  # 5s (we detect an env server as unhealth after 3 * 5s = 15s of unsuccessful health checks)
    startup_timeout: float = 600.0,  # 10m
    recovery_timeout: float = 600.0,  # 10m
) -> EnvClient:
    """Sets up a ZMQEnvClient for a given address."""
    return ZMQEnvClient(
        address=address,
        name=name,
        health_check_interval=health_check_interval,
        startup_timeout=startup_timeout,
        recovery_timeout=recovery_timeout,
    )


async def wait_for_env_servers(env_clients: list[EnvClient]) -> None:
    await asyncio.gather(*[env_client.wait_for_server_startup() for env_client in env_clients])


async def run_group(
    env: vf.Environment,
    client: vf.ClientConfig,
    model_name: str,
    example: dict,
    rollouts_per_example: int,
    sampling_args: dict,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
) -> list[vf.RolloutOutput]:
    """
    Wrapper for vf.Environment.run_group().

    Asynchronously generates and scores a group.
    """
    state_columns = state_columns + REQUIRED_STATE_COLUMNS
    outputs_per_input = getattr(env, "outputs_per_input", 1)
    inputs_needed = max(1, rollouts_per_example // outputs_per_input)
    group_inputs = [vf.RolloutInput(**example) for _ in range(inputs_needed)]
    return await env.run_group(
        group_inputs,
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        max_retries=max_retries,
        state_columns=state_columns,
    )


# TODO: migrate this to vf.Environment.generate() once it supports multiple clients
async def generate(
    env: vf.Environment,
    model_name: str,
    examples: list,
    rollouts_per_example: int,
    sampling_args: dict,
    clients: list[vf.ClientConfig] | None = None,
    get_client: Callable[[], Awaitable[vf.ClientConfig]] | None = None,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
    pbar_description: str = "Generating rollouts",
) -> list[vf.RolloutOutput]:
    """
    Wrapper for vf.Environment.generate().

    NOTE: Currently we cannot use vf.Environment.generate() directly because it does not support multiple clients.

    Asynchronously generates and scores a list of groups.
    """

    if not clients and get_client is None:
        raise ValueError("generate requires at least one client or a get_client callback")

    if get_client is None:
        client_cycle = cycle(clients)

        async def get_client() -> vf.ClientConfig:
            return next(client_cycle)

    total_rollouts = len(examples) * rollouts_per_example
    pbar = ProgressTracker(total=total_rollouts, desc=pbar_description)

    async def run_group_with_progress(example):
        client = await get_client()
        result = await run_group(
            env=env,
            client=client,
            model_name=model_name,
            example=example,
            rollouts_per_example=rollouts_per_example,
            max_retries=max_retries,
            state_columns=state_columns,
            sampling_args=sampling_args,
        )
        pbar.update(rollouts_per_example)
        return result

    try:
        group_outputs_list: list[list[vf.RolloutOutput]] = await asyncio.gather(
            *[run_group_with_progress(example) for example in examples]
        )
    finally:
        pbar.close()

    return [output for group_outputs in group_outputs_list for output in group_outputs]


async def evaluate(
    env: vf.Environment,
    model_name: str,
    sampling_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    clients: list[vf.ClientConfig] | None = None,
    get_client: Callable[[], Awaitable[vf.ClientConfig]] | None = None,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
) -> list[vf.RolloutOutput]:
    """
    Wrapper for vf.Environment.evaluate().

    NOTE: Currently we cannot use vf.Environment.evaluate() directly because it does not support multiple clients.
          Instead, we use our generate() wrapper which round-robins clients.

    """
    inputs = env._get_eval_inputs(num_examples, rollouts_per_example)
    outputs = await generate(
        env=env,
        clients=clients,
        get_client=get_client,
        model_name=model_name,
        examples=inputs,
        # _get_eval_inputs() already repeats the examples, this currently means
        # we do not support eval envs with group scoring well -- this should be
        # resolved once we can use vf.Environment.generate() and
        # vf.Environment.evaluate() directly though
        rollouts_per_example=1,
        sampling_args=sampling_args,
        max_retries=max_retries,
        state_columns=state_columns,
    )
    return outputs


# TODO: remove once usage is tracked by verifiers
def get_prompt_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of prompt tokens from vf.RolloutOutput. Defined as the
    number of prompt ids from the first trajectory step. If raw tokens are not
    available, falls back to checking the usage of the first response.
    """
    if not output["trajectory"]:
        return 0
    first_step = output["trajectory"][0]
    if first_step["tokens"] is not None:
        return len(first_step["tokens"]["prompt_ids"])
    first_step_response = first_step["response"]
    return (first_step_response.get("usage") or {}).get("prompt_tokens", 0)


# TODO: remove once usage is tracked by verifiers
def get_seq_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of tokens from vf.RolloutOutput. Defined as the sum of prompt
    and completion tokens from the last trajectory step. If raw tokens are not
    available, falls back to checking the usage of the last response.
    """
    if not output["trajectory"]:
        return 0
    last_step = output["trajectory"][-1]
    if last_step["tokens"] is not None:
        return len(last_step["tokens"]["prompt_ids"]) + len(last_step["tokens"]["completion_ids"])
    last_step_response = last_step["response"]
    return (last_step_response.get("usage") or {}).get("total_tokens", 0)


# TODO: remove once usage is tracked by verifiers
def get_completion_len(output: vf.RolloutOutput) -> int:
    """
    Computes the number of completion tokens from vf.RolloutOutput. Defined as
    the difference between the total number of tokens and the number of prompt
    tokens.
    """
    return get_seq_len(output) - get_prompt_len(output)


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False
