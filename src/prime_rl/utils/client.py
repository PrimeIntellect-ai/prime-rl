import asyncio
import os
from collections import defaultdict
from itertools import cycle
from pathlib import Path

import httpx
from datasets import Dataset
from httpx import AsyncClient
from openai import AsyncOpenAI, NotFoundError
from verifiers import Environment
from verifiers.types import GenerateOutputs

from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import get_logger


def setup_clients(client_config: ClientConfig) -> list[AsyncOpenAI]:
    def _setup_client(base_url: str) -> AsyncOpenAI:
        # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
        timeout = httpx.Timeout(timeout=client_config.timeout, connect=5.0)
        # We use as many concurrent connections as possible, but lower than available ports
        limits = httpx.Limits(
            max_connections=28000,  # OAI default: 1000
            max_keepalive_connections=28000,  # OAI default: 100
        )
        http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return AsyncOpenAI(
            base_url=base_url,
            api_key=os.getenv(client_config.api_key_var, "EMPTY"),
            max_retries=10,  # OAI default: 2 (does exponential backoff and reasonable timeout in between retries)
            http_client=http_client,
        )

    return [_setup_client(base_url) for base_url in client_config.base_urls]


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    """Create a dedicated admin client for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    """

    def _setup_admin_client(base_url: str) -> httpx.AsyncClient:
        headers = {}
        api_key = os.getenv(client_config.api_key_var, "EMPTY")
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        # Strip /v1 suffix since admin endpoints are at root level
        base_url = base_url.rstrip("/").removesuffix("/v1")

        return httpx.AsyncClient(
            base_url=base_url,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=None),
        )

    return [_setup_admin_client(base_url) for base_url in client_config.base_urls]


async def check_has_model(clients: list[AsyncOpenAI], model_name: str) -> None:
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(*[client.models.list() for client in clients])
    for client, result in zip(clients, results):
        models = result.data
        if not any(model.id == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


def merge_outputs(generate_outputs_list: list[GenerateOutputs]) -> GenerateOutputs:
    """Merge multiple GenerateOutputs into a single GenerateOutputs."""
    prompt, completion, answer, state, reward, info, task, metrics = [], [], [], [], [], [], [], defaultdict(list)
    for generate_output in generate_outputs_list:
        prompt.extend(generate_output.prompt)
        completion.extend(generate_output.completion)
        answer.extend(generate_output.answer)
        state.extend(generate_output.state)
        reward.extend(generate_output.reward)
        info.extend(generate_output.info)
        task.extend(generate_output.task)
        for key, value in generate_output.metrics.items():
            metrics[key].extend(value)
    return GenerateOutputs(
        prompt=prompt,
        completion=completion,
        answer=answer,
        state=state,
        reward=reward,
        info=info,
        task=task,
        metrics=metrics,
    )


async def generate_group(
    client: AsyncOpenAI,
    env: Environment,
    model_name: str,
    problem: dict,
    rollouts_per_example: int,
    sampling_args: dict,
    max_concurrent: int,
) -> GenerateOutputs:
    """Asynchronously generate and score rollouts for one problem."""
    return await env.a_generate(
        inputs=Dataset.from_list([problem] * rollouts_per_example),
        client=client,
        model=model_name,
        sampling_args=sampling_args,
        max_concurrent=max_concurrent,
    )


async def generate_batch(
    clients: list[AsyncOpenAI],
    env: Environment,
    model_name: str,
    problems: list[dict],
    rollouts_per_example: int,
    sampling_args: dict,
    max_concurrent: int = -1,
) -> GenerateOutputs:
    """Asynchronously generate and score rollouts for a list of problems."""
    generate_outputs_list: list[GenerateOutputs] = await asyncio.gather(
        *[
            generate_group(client, env, model_name, problem, rollouts_per_example, sampling_args, max_concurrent)
            for client, problem in zip(cycle(clients), problems)
        ]
    )
    return merge_outputs(generate_outputs_list)


async def check_health(
    admin_clients: list[AsyncClient], interval: int = 1, log_interval: int = 10, timeout: int = 1800
) -> None:
    logger = get_logger()

    async def _check_health(admin_client: AsyncClient) -> None:
        wait_time = 0
        logger.debug("Starting pinging /health to check health")
        while wait_time < timeout:
            try:
                await admin_client.get("/health")
                logger.debug(f"Inference pool is ready after {wait_time} seconds")
                return
            except NotFoundError:
                logger.warning("The route /health does not exist. Skipping health check.")
                return
            except Exception as e:
                if wait_time % log_interval == 0 and wait_time > 0:
                    logger.warning(f"Inference server was not reached after {wait_time} seconds (Error: {e})")
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await asyncio.gather(*[_check_health(admin_client) for admin_client in admin_clients])


async def update_weights(admin_clients: list[AsyncClient], weight_dir: Path) -> None:
    """Make a HTTP post request to the vLLM server to update the weights."""
    logger = get_logger()

    async def _update_weights(admin_client: AsyncClient, weight_dir: Path) -> None:
        try:
            response = await admin_client.post("/update_weights", json={"weight_dir": weight_dir.as_posix()})
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /update_weights does not exist. Skipping weight update.")
                return
            raise

    await asyncio.gather(*[_update_weights(admin_client, weight_dir) for admin_client in admin_clients])


async def reload_weights(admin_clients: list[AsyncClient]) -> None:
    """Make a HTTP post request to the vLLM server to reload weights (reset to base model)."""
    logger = get_logger()

    async def _reload_weights(admin_client: AsyncClient) -> None:
        logger.debug("Sending request to reload weights (reset to base model)")
        try:
            response = await admin_client.post("/reload_weights", json={})
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /reload_weights does not exist. Skipping weight reload.")
                return
            raise

    await asyncio.gather(*[_reload_weights(admin_client) for admin_client in admin_clients])
