import asyncio
import os
from pathlib import Path

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI, NotFoundError

from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import get_logger


def setup_clients(client_config: ClientConfig) -> list[AsyncOpenAI]:
    def _setup_client(base_url: str) -> AsyncOpenAI:
        # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
        timeout = httpx.Timeout(timeout=client_config.timeout, connect=5.0)
        # We use as many concurrent connections as possible, but lower than available ports
        limits = httpx.Limits(
            max_connections=1024,  # OAI default: 1000
            max_keepalive_connections=1024,  # OAI default: 100
        )
        http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        return AsyncOpenAI(
            base_url=base_url,
            api_key=os.getenv(client_config.api_key_var, "EMPTY"),
            max_retries=10,  # OAI default: 2 (does exponential backoff and reasonable timeout in between retries)
            http_client=http_client,
        )

    return [_setup_client(base_url) for base_url in client_config.base_url]


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

        return AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            timeout=httpx.Timeout(client_config.timeout, connect=5.0, pool=None),
        )

    return [_setup_admin_client(base_url) for base_url in client_config.base_url]


async def check_has_model(clients: list[AsyncOpenAI], model_name: str) -> None:
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(*[client.models.list() for client in clients])
    for client, result in zip(clients, results):
        models = result.data
        if not any(model.id == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


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


async def update_weights(admin_clients: list[AsyncClient], weight_dir: Path | None) -> None:
    """Make a HTTP post request to the vLLM server to update the weights."""
    logger = get_logger()

    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

    async def _update_weights(admin_client: AsyncClient, weight_dir: str | None) -> None:
        try:
            response = await admin_client.post("/update_weights", json={"weight_dir": weight_dir})
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /update_weights does not exist. Skipping weight update.")
                return
            raise

    await asyncio.gather(*[_update_weights(admin_client, weight_dir_posix) for admin_client in admin_clients])


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


async def init_nccl_broadcast(admin_clients: list[AsyncClient], host: str, port: int) -> None:
    """Make a HTTP post request to the vLLM server to initialize the NCCL broadcast."""
    logger = get_logger()

    async def _init_nccl_broadcast(admin_client: AsyncClient, host: str, port: int, client_num: int) -> None:
        try:
            response = await admin_client.post(
                "/init_broadcaster",
                json={
                    "host": host,
                    "port": port,
                    "server_rank": client_num,
                    "num_inference_server": len(admin_clients),
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization.")
                return

    await asyncio.gather(
        *[
            _init_nccl_broadcast(admin_client, host, port, client_num)
            for client_num, admin_client in enumerate(admin_clients)
        ]
    )
