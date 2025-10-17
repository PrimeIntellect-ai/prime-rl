import asyncio
import itertools
import os
from pathlib import Path

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI, NotFoundError
from openai.resources import AsyncChat, AsyncCompletions

from prime_rl.orchestrator.config import ClientConfig
from prime_rl.utils.logger import get_logger


class RoundRobinAsyncOpenAI(AsyncOpenAI):
    """A AsyncOpenAI client which round-robins (chat) completions requests across multiple clients."""

    def __init__(self, clients: list[AsyncOpenAI]):
        self.clients = clients
        self.cycle = itertools.cycle(clients)

    def _next_client(self) -> AsyncOpenAI:
        return next(self.cycle)

    @property
    def chat(self) -> AsyncChat:
        """Round-robin chat requests"""
        return AsyncChat(self._next_client())

    @property
    def completions(self) -> AsyncCompletions:
        """Round-robin completion requests"""
        return AsyncCompletions(self._next_client())


def setup_client(client_config: ClientConfig) -> RoundRobinAsyncOpenAI:
    def setup_single_client(base_url: str) -> AsyncOpenAI:
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

    clients = [setup_single_client(base_url) for base_url in client_config.base_url]
    return RoundRobinAsyncOpenAI(clients=clients)


async def check_has_model(client: RoundRobinAsyncOpenAI, model_name: str) -> None:
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(*[c.models.list() for c in client.clients])
    for c, result in zip(client.clients, results):
        models = result.data
        if not any(model.id == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {c.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    """Create a dedicated admin client for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    """

    def setup_single_admin_client(base_url: str, api_key_var: str) -> httpx.AsyncClient:
        headers = {}
        api_key = os.getenv(api_key_var, "EMPTY")
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

    return [
        setup_single_admin_client(base_url, api_key_var)
        for base_url, api_key_var in zip(client_config.base_url, client_config.api_key_var)
    ]


async def check_health(
    admin_clients: list[AsyncClient], interval: int = 1, log_interval: int = 10, timeout: int = 1800
) -> None:
    logger = get_logger()

    async def check_health_single_client(admin_client: AsyncClient) -> None:
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

    await asyncio.gather(*[check_health_single_client(admin_client) for admin_client in admin_clients])


async def update_weights(admin_clients: list[AsyncClient], weight_dir: Path) -> None:
    """Make a HTTP post request to the vLLM server to update the weights."""
    logger = get_logger()

    async def update_weights_single_client(admin_client: AsyncClient, weight_dir: Path) -> None:
        try:
            response = await admin_client.post("/update_weights", json={"weight_dir": weight_dir.as_posix()})
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /update_weights does not exist. Skipping weight update.")
                return
            raise

    await asyncio.gather(*[update_weights_single_client(admin_client, weight_dir) for admin_client in admin_clients])


async def reload_weights(admin_clients: list[AsyncClient]) -> None:
    """Make a HTTP post request to the vLLM server to reload weights (reset to base model)."""
    logger = get_logger()

    async def reload_weights_single_client(admin_client: AsyncClient) -> None:
        logger.debug("Sending request to reload weights (reset to base model)")
        try:
            response = await admin_client.post("/reload_weights", json={})
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /reload_weights does not exist. Skipping weight reload.")
                return
            raise

    await asyncio.gather(*[reload_weights_single_client(admin_client) for admin_client in admin_clients])
