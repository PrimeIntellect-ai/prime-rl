import asyncio
import os
from pathlib import Path

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI, NotFoundError

from prime_rl.orchestrator.config import ClientConfig
from prime_rl.utils.logger import get_logger


class OAIClient:
    """Client used to make OAI chat completions requests against a vLLM server."""

    def __init__(self, client_config: ClientConfig):
        self.config = client_config


class AdminClient:
    """Client used to make admin requests against a vLLM server, such as weight reloads/ updates."""

    DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=None)
    DEFAULT_LIMITS = httpx.Limits(max_connections=1, max_keepalive_connections=0)

    def __init__(self, client_config: ClientConfig):
        self.logger = get_logger()
        self.config = client_config

        # Setup authentication
        headers = {}
        api_key = os.getenv(self.config.api_key_var, "EMPTY")
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        self.base_url = self.config.base_url.rstrip("/").removesuffix("/v1")

        # Strip /v1 suffix since admin endpoints are at root level
        self.client = AsyncClient(
            base_url=self.base_url,
            headers=headers,
            limits=AdminClient.DEFAULT_LIMITS,
            timeout=AdminClient.DEFAULT_TIMEOUT,
        )

    async def check_health(self, interval: int = 1, log_interval: int = 10, timeout: int = 1800) -> None:
        wait_time = 0
        self.logger.debug("Starting pinging /health to check health")
        while wait_time < timeout:
            try:
                await self.client.get("/health")
                self.logger.debug(f"Inference pool is ready after {wait_time} seconds")
                return
            except NotFoundError:
                self.logger.warning("The route /health does not exist. Skipping health check.")
                return
            except Exception as e:
                if wait_time % log_interval == 0 and wait_time > 0:
                    self.logger.warning(f"Inference server was not reached after {wait_time} seconds (Error: {e})")
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        self.logger.error(msg)
        raise TimeoutError(msg)

    async def update_weights(self, weight_dir: Path):
        """Update the weights on the inference server to the local weight checkpoint directory."""
        try:
            response = await self.client.post("/update_weights", json={"weight_dir": weight_dir.as_posix()})
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self.logger.warning("The route /update_weights does not exist. Skipping weight update.")
                return
            raise

    async def reload_weights(self):
        """Reload the weights on the inference server to the base model."""
        self.logger.debug("Sending request to /reload_weights (reset to base model)")
        try:
            response = await self.client.post("/reload_weights", json={})
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self.logger.warning("The route /reload_weights does not exist. Skipping weight reload.")
                return
            raise


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


async def check_has_model(client: AsyncOpenAI, model_name: str) -> None:
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    models = (await client.models.list()).data
    if not any(model.id == model_name for model in models):
        raise ValueError(f"Model {model_name} was not found in the inference pool")
    logger.debug(f"Model {model_name} was found in the inference pool")
