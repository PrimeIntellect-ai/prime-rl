import asyncio
import logging
import multiprocessing as mp
import os
from collections.abc import Awaitable, Callable
from itertools import count, cycle
from typing import Any

import httpx
import verifiers as vf
from verifiers.envs.environment import EnvClient
from verifiers.utils.worker_utils import get_free_port
from verifiers.workers import ZMQEnvClient, ZMQEnvServer

from prime_rl.utils.logger import InterceptHandler, ProgressTracker, get_logger

DEFAULT_RETRIES = 0
REQUIRED_STATE_COLUMNS = ["trajectory", "sampling_args"]
DEFAULT_STATE_COLUMNS = []


def has_program_id(sampling_args: dict[str, Any]) -> bool:
    """Check if sampling args already define extra_body.program_id."""
    extra_body = sampling_args.get("extra_body")
    if not isinstance(extra_body, dict):
        return False
    return "program_id" in extra_body


def with_program_id(sampling_args: dict[str, Any], program_id: str | None) -> dict[str, Any]:
    """Return sampling args with extra_body.program_id set if provided."""
    if program_id is None:
        return sampling_args

    base_extra_body = sampling_args.get("extra_body")
    extra_body = base_extra_body.copy() if isinstance(base_extra_body, dict) else {}
    extra_body["program_id"] = program_id
    return {**sampling_args, "extra_body": extra_body}


def get_program_release_url(api_base_url: str) -> str:
    """Build ThunderAgent release URL from an OpenAI API base URL."""
    return f"{api_base_url.rstrip('/').removesuffix('/v1')}/programs/release"


async def release_program(client: vf.ClientConfig, program_id: str) -> None:
    """Best-effort release of ThunderAgent program state.

    If the endpoint does not exist (e.g., direct vLLM), this is a no-op.
    """
    headers = client.extra_headers.copy() if client.extra_headers else {}
    api_key = os.getenv(client.api_key_var, "EMPTY")
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    timeout_s = max(1, min(int(client.timeout), 5))
    release_url = get_program_release_url(client.api_base_url)

    try:
        async with httpx.AsyncClient(timeout=timeout_s, headers=headers) as http_client:
            response = await http_client.post(release_url, json={"program_id": program_id})
            if response.status_code in (200, 404):
                return
            get_logger().debug(
                f"ThunderAgent program release returned status {response.status_code} for {program_id} at {release_url}"
            )
    except Exception as e:
        get_logger().debug(f"ThunderAgent program release failed for {program_id} at {release_url}: {e}")


def spawn_env_server(
    env_id: str,
    env_args: dict[str, Any],
    extra_env_kwargs: dict[str, Any],
    address: str | None = None,
    log_level: str | None = None,
    log_file: str | None = None,
    log_file_level: str | None = None,
    daemon: bool = True,
) -> str:
    """
    Starts a ZMQEnvServer process in a subprocess.

    Mirrors vf.Environment.start_server().
    """
    address = address or f"tcp://127.0.0.1:{get_free_port()}"
    # Use spawn to avoid inheriting file descriptors (e.g. sockets) from
    # the parent process, which has caused hangs when multiple env server
    # subprocesses share the same fds.
    mp.get_context("spawn").Process(
        target=ZMQEnvServer.run_server,
        args=(
            env_id,
            env_args,
            extra_env_kwargs,
            log_level,
            log_file,
            log_file_level,
        ),
        kwargs=dict(address=address),
        daemon=daemon,
    ).start()

    return address


def setup_env_client(address: str) -> EnvClient:
    """Sets up a ZMQEnvClient for a given address."""
    return ZMQEnvClient(address=address)


async def wait_for_env_servers(
    env_clients: list[EnvClient], interval: int = 1, log_interval: int = 10, timeout: int = 1800
) -> None:
    logger = get_logger()

    async def wait_for_env_server(env_client: EnvClient) -> None:
        wait_time = 0
        logger.debug(f"Starting pinging environment server at {env_client.address}")
        while wait_time < timeout:
            try:
                await env_client.health(timeout=1)  # quick timeout
                logger.debug(f"Environment server at {env_client.address} is ready after {wait_time} seconds")
                return
            except Exception as e:
                if wait_time % log_interval == 0 and wait_time > 0:
                    logger.warning(
                        f"Environment server at {env_client.address} was not reached after {wait_time} seconds (Error: {e})"
                    )
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Environment server at {env_client.address} is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await asyncio.gather(*[wait_for_env_server(env_client) for env_client in env_clients])


async def run_group(
    env: vf.Environment,
    client: vf.ClientConfig,
    model_name: str,
    example: dict,
    rollouts_per_example: int,
    sampling_args: dict,
    program_id: str | None = None,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
) -> list[vf.RolloutOutput]:
    """
    Wrapper for vf.Environment.run_group().

    Asynchronously generates and scores a group.
    """
    state_columns = state_columns + REQUIRED_STATE_COLUMNS
    group_inputs = [vf.RolloutInput(**example) for _ in range(rollouts_per_example)]
    request_sampling_args = with_program_id(sampling_args, program_id=program_id)
    try:
        return await env.run_group(
            group_inputs,
            client=client,
            model=model_name,
            sampling_args=request_sampling_args,
            max_retries=max_retries,
            state_columns=state_columns,
        )
    finally:
        if program_id is not None:
            await asyncio.shield(release_program(client, program_id))


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
    auto_program_id: bool = False,
    program_id_prefix: str = "prime-rl",
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
    program_counter = count()
    inject_program_id = auto_program_id and not has_program_id(sampling_args)

    async def run_group_with_progress(example):
        client = await get_client()
        program_id = None
        if inject_program_id:
            rollout_idx = next(program_counter)
            example_id = example.get("example_id") if isinstance(example, dict) else None
            program_suffix = example_id if example_id is not None else rollout_idx
            program_id = f"{program_id_prefix}-{program_suffix}-{rollout_idx}"
        result = await run_group(
            env=env,
            client=client,
            model_name=model_name,
            example=example,
            rollouts_per_example=rollouts_per_example,
            max_retries=max_retries,
            state_columns=state_columns,
            sampling_args=sampling_args,
            program_id=program_id,
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
    auto_program_id: bool = False,
    program_id_prefix: str = "eval",
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
        auto_program_id=auto_program_id,
        program_id_prefix=program_id_prefix,
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


def intercept_vf_logging(level: str = "DEBUG", prefix: str = "verifiers"):
    """Intercepts verifiers logging and routes through prime-rl logger with [verifiers] prefix."""
    vf_logger = logging.getLogger("verifiers")
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False
