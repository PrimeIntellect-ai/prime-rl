import asyncio
import logging
from itertools import cycle
from multiprocessing import Process
from typing import Any

import verifiers as vf
from verifiers.envs.environment import EnvClient
from verifiers.utils.worker_utils import get_free_port
from verifiers.workers import ZMQEnvClient, ZMQEnvServer

from prime_rl.utils.logger import InterceptHandler, ProgressTracker, get_logger

DEFAULT_RETRIES = 3
REQUIRED_STATE_COLUMNS = ["trajectory", "sampling_args"]
DEFAULT_STATE_COLUMNS = []


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
    Process(
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


async def run_rollout(
    env: vf.Environment,
    client: vf.ClientConfig,
    model_name: str,
    example: dict,
    sampling_args: dict,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
) -> vf.RolloutOutput:
    """
    Wrapper for vf.Environment.run_rollout().

    Asynchronously generates and scores a rollout.
    """
    state_columns = state_columns + REQUIRED_STATE_COLUMNS
    rollout_input = vf.RolloutInput(**example)
    return await env.run_rollout(
        rollout_input,
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        max_retries=max_retries,
        state_columns=state_columns,
    )


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
    group_inputs = [vf.RolloutInput(**example) for _ in range(rollouts_per_example)]
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
    clients: list[vf.ClientConfig],
    model_name: str,
    examples: list,
    rollouts_per_example: int,
    sampling_args: dict,
    max_retries: int = DEFAULT_RETRIES,
    state_columns: list[str] = DEFAULT_STATE_COLUMNS,
    pbar_description: str = "Generating rollouts",
) -> list[vf.RolloutOutput]:
    """
    Wrapper for vf.Environment.generate().

    NOTE: Currently we cannot use vf.Environment.generate() directly because it does not support multiple clients.

    Asynchronously generates and scores a list of groups.
    """

    total_rollouts = len(examples) * rollouts_per_example
    pbar = ProgressTracker(total=total_rollouts, desc=pbar_description)

    async def run_group_with_progress(client, example):
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
            *[run_group_with_progress(client, example) for client, example in zip(cycle(clients), examples)]
        )
    finally:
        pbar.close()

    return [output for group_outputs in group_outputs_list for output in group_outputs]


async def evaluate(
    env: vf.Environment,
    clients: list[vf.ClientConfig],
    model_name: str,
    sampling_args: dict,
    num_examples: int,
    rollouts_per_example: int,
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


def intercept_vf_logging(level: str = "DEBUG", prefix: str = "verifiers"):
    """Intercepts verifiers logging and routes through prime-rl logger with [verifiers] prefix."""
    vf_logger = logging.getLogger("verifiers")
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False
