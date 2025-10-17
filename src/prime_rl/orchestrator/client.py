import asyncio
import os
from pathlib import Path

import httpx
from httpx import AsyncClient, Response
from openai import AsyncOpenAI, NotFoundError

from prime_rl.orchestrator.config import ClientConfig
from prime_rl.utils.logger import get_logger


def setup_admin_client(client_config: ClientConfig) -> httpx.AsyncClient:
    """Create a dedicated admin client for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    """
    headers = {}
    api_key = os.getenv(client_config.api_key_var, "EMPTY")
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    # Strip /v1 suffix since admin endpoints are at root level
    base_url = client_config.base_url.rstrip("/").removesuffix("/v1")

    return httpx.AsyncClient(
        base_url=base_url,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        headers=headers,
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=None),
    )


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


async def update_weights(admin_client: AsyncClient, weight_dir: Path) -> None:
    """Make a HTTP post request to the vLLM server to update the weights."""
    logger = get_logger()
    try:
        response = await admin_client.post("/update_weights", json={"weight_dir": weight_dir.as_posix()})
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning("The route /update_weights does not exist. Skipping weight update.")
            return
        raise


async def reload_weights(admin_client: AsyncClient) -> None:
    """Make a HTTP post request to the vLLM server to reload weights (reset to base model)."""
    logger = get_logger()
    logger.debug("Sending request to reload weights (reset to base model)")
    try:
        response = await admin_client.post("/reload_weights", json={})
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning("The route /reload_weights does not exist. Skipping weight reload.")
            return
        raise
