import asyncio
import os
from pathlib import Path

import httpx
from httpx import Response
from openai import AsyncOpenAI, NotFoundError

from prime_rl.orchestrator.config import ClientConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_weight_ckpt_model_path


def setup_client(client_config: ClientConfig) -> AsyncOpenAI:
    # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
    timeout = httpx.Timeout(timeout=client_config.timeout, connect=5.0)
    # We use as many concurrent connections as possible, but lower than available ports
    limits = httpx.Limits(
        max_connections=28000,  # OAI default: 1000
        max_keepalive_connections=28000,  # OAI default: 100
    )
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    return AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=os.getenv(client_config.api_key_var, "EMPTY"),
        max_retries=10,  # OAI default: 2 (does exponential backoff and reasonable timeout in between retries)
        http_client=http_client,
    )


async def check_health(client: AsyncOpenAI, interval: int = 1, log_interval: int = 10, timeout: int = 1800) -> None:
    logger = get_logger()
    wait_time = 0
    url = str(client.base_url).strip()[:-4] + "/health"
    logger.debug(f"Starting pinging {url} to check health")
    while wait_time < timeout:
        try:
            await client.get(url, cast_to=Response, options={"max_retries": 0})
            logger.debug(f"Inference pool is ready after {wait_time} seconds")
            return
        except NotFoundError:
            logger.warning(f"The route {url} does not exist. Skipping health check.")
            return
        except Exception as e:
            if wait_time % log_interval == 0 and wait_time > 0:
                logger.warning(f"Inference server was not reached after {wait_time} seconds (Error: {e})")
            await asyncio.sleep(interval)
            wait_time += interval
    msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
    logger.error(msg)
    raise TimeoutError(msg)


async def check_has_model(client: AsyncOpenAI, model_name: str) -> None:
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    models = (await client.models.list()).data
    if not any(model.id == model_name for model in models):
        raise ValueError(f"Model {model_name} was not found in the inference pool")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def update_weights(client: AsyncOpenAI, path: Path, step: int) -> None:
    """Make a HTTP post request to the vLLM server to update the weights."""
    logger = get_logger()
    url = str(client.base_url).strip()[:-4] + "/update_weights"
    try:
        model_path = get_weight_ckpt_model_path(path, step).absolute()
        logger.debug(f"Sending request to {url} to update weights from {model_path}")
        await client.post(url, cast_to=Response, body={"model_path": model_path.as_posix()})
    except NotFoundError:
        logger.warning(f"The route {url} does not exist. Skipping weight update.")
        return


async def reload_weights(client: AsyncOpenAI) -> None:
    """Make a HTTP post request to the vLLM server to reload weights (reset to base model)."""
    logger = get_logger()
    url = str(client.base_url).strip()[:-4] + "/reload_weights"
    try:
        logger.debug(f"Sending request to {url} to reload weights (reset to base model)")
        await client.post(url, cast_to=Response, body={})
    except NotFoundError:
        logger.warning(f"The route {url} does not exist. Skipping weight reload.")
        return
    await client.post(url, cast_to=Response, body={})
