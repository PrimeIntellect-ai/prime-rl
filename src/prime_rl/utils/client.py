from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from httpx import AsyncClient
from openai import AsyncOpenAI, NotFoundError
from prime_evals import AsyncEvalsClient
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.utils.elastic import ElasticInferencePool


class InferencePool:
    """Unified interface for static and elastic inference pools.

    Encapsulates the difference between static client list and elastic DNS-based discovery.
    """

    def __init__(
        self,
        clients: list[AsyncOpenAI],
        admin_clients: list[AsyncClient],
        elastic_pool: ElasticInferencePool | None = None,
    ):
        self._clients = clients
        self._admin_clients = admin_clients
        self._elastic_pool = elastic_pool

    @property
    def clients(self) -> list[AsyncOpenAI]:
        """Get inference clients (refreshed from elastic pool if applicable)."""
        if self._elastic_pool is not None:
            return self._elastic_pool.get_inference_clients()
        return self._clients

    @property
    def admin_clients(self) -> list[AsyncClient]:
        """Get admin clients (refreshed from elastic pool if applicable)."""
        if self._elastic_pool is not None:
            return self._elastic_pool.admin_clients
        return self._admin_clients

    @property
    def elastic_pool(self) -> ElasticInferencePool | None:
        """Get underlying elastic pool if in elastic mode."""
        return self._elastic_pool

    @property
    def is_elastic(self) -> bool:
        """Check if running in elastic mode."""
        return self._elastic_pool is not None

    async def wait_for_ready(self, model_name: str, timeout: int = 1800) -> None:
        """Wait for inference pool to be ready."""
        if self._elastic_pool is not None:
            await self._elastic_pool.wait_for_ready(min_servers=1, timeout=timeout)
        else:
            await check_health(self._admin_clients, timeout=timeout)
            await check_has_model(self._clients, model_name)

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        """Update weights on all inference servers."""
        await update_weights(
            self._admin_clients, weight_dir, lora_name=lora_name, elastic_pool=self._elastic_pool, step=step
        )

    def get_metrics(self) -> dict[str, float]:
        """Get pool metrics (only meaningful for elastic mode)."""
        if self._elastic_pool is not None:
            return self._elastic_pool.get_metrics()
        return {}

    async def stop(self) -> None:
        """Stop the inference pool (only needed for elastic mode)."""
        if self._elastic_pool is not None:
            await self._elastic_pool.stop()


async def setup_inference_pool(client_config: ClientConfig, base_model: str | None = None) -> InferencePool:
    """Create an inference pool from config (handles both static and elastic modes)."""
    logger = get_logger()

    if client_config.is_elastic:
        from prime_rl.utils.elastic import ElasticInferencePool

        elastic_pool = await ElasticInferencePool.from_config(client_config, base_model=base_model)
        return InferencePool(
            clients=elastic_pool.get_inference_clients(),
            admin_clients=elastic_pool.admin_clients,
            elastic_pool=elastic_pool,
        )
    else:
        logger.info(
            f"Initializing OpenAI client (base_url={', '.join(client_config.base_url)}, "
            f"api_key_var={client_config.api_key_var}, headers={client_config.headers})"
        )
        return InferencePool(
            clients=setup_clients(client_config),
            admin_clients=setup_admin_clients(client_config),
        )


def setup_clients(client_config: ClientConfig) -> list[AsyncOpenAI]:
    def _setup_client(base_url: str) -> AsyncOpenAI:
        # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
        timeout = httpx.Timeout(client_config.timeout)
        # We use as many concurrent connections as possible, but lower than available ports
        limits = httpx.Limits(
            max_connections=8192,  # OAI default: 1000
            max_keepalive_connections=8192,  # OAI default: 100
        )
        http_client = httpx.AsyncClient(limits=limits, timeout=timeout, headers=client_config.headers)
        return AsyncOpenAI(
            base_url=base_url,
            api_key=os.getenv(client_config.api_key_var, "EMPTY"),
            max_retries=10,  # OAI default: 2 (does exponential backoff and reasonable timeout in between retries)
            http_client=http_client,
        )

    return [_setup_client(base_url) for base_url in client_config.base_url]


def setup_evals_client() -> AsyncEvalsClient:
    return AsyncEvalsClient()


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    """Create a dedicated admin client for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    """

    def _setup_admin_client(base_url: str) -> httpx.AsyncClient:
        headers = client_config.headers.copy()  # avoid mutating config
        api_key = os.getenv(client_config.api_key_var, "EMPTY")
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        # Strip /v1 suffix since admin endpoints are at root level
        base_url = base_url.rstrip("/").removesuffix("/v1")

        return AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
            timeout=httpx.Timeout(client_config.timeout),
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
                    logger.warning(
                        f"Inference server was not reached after {wait_time} seconds (Error: {e}) on {admin_client.base_url}"
                    )
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await asyncio.gather(*[_check_health(admin_client) for admin_client in admin_clients])


NCCL_READY_MARKER = "NCCL_READY"


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | None,
    lora_name: str | None = None,
    *,
    elastic_pool: ElasticInferencePool | None = None,
    step: int = 0,
) -> None:
    """Update weights on inference servers.

    If elastic_pool is provided, uses per-server verification and readiness tracking.
    Otherwise uses fire-and-forget to admin_clients.

    Creates a NCCL_READY marker file before calling the update endpoint to signal
    to the trainer that inference workers are about to enter the receive path.
    This marker is only used in NCCL broadcast mode but is harmless in filesystem mode.
    """
    if elastic_pool is not None:
        await elastic_pool.sync_weights(weight_dir, lora_name, step)
        return

    logger = get_logger()

    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

    if lora_name is not None and weight_dir is not None:
        await load_lora_adapter(admin_clients, lora_name, weight_dir)
    else:

        async def _update_weights(admin_client: AsyncClient, weight_dir: str | None) -> None:
            try:
                response = await admin_client.post("/update_weights", json={"weight_dir": weight_dir})
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning("The route /update_weights does not exist. Skipping weight update.")
                    return
                raise

        # Create ready marker before servers enter receive path (used by NCCL broadcast)
        if weight_dir is not None:
            nccl_ready_file = weight_dir / NCCL_READY_MARKER
            nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
            nccl_ready_file.touch()
            logger.debug(f"Created NCCL_READY marker at {nccl_ready_file}")

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


def _is_retryable_lora_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for LoRA loading."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 404 (adapter not found) or 500 (server error during loading)
        return exception.response.status_code in (404, 500)
    return False


async def load_lora_adapter(admin_clients: list[AsyncClient], lora_name: str, lora_path: Path) -> None:
    """Make a HTTP post request to the vLLM server to load a LoRA adapter.

    Retries with exponential backoff if the adapter files are not found,
    which can happen due to NFS propagation delays.
    """
    logger = get_logger()
    lora_path_posix = lora_path.as_posix()

    @retry(
        retry=retry_if_exception(_is_retryable_lora_error),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _load_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to load LoRA adapter {lora_name} from {lora_path}")
        response = await admin_client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path_posix},
        )
        response.raise_for_status()

    await asyncio.gather(*[_load_lora_adapter(admin_client) for admin_client in admin_clients])


async def unload_lora_adapter(admin_clients: list[AsyncClient], lora_name: str) -> None:
    """Make a HTTP post request to the vLLM server to unload a LoRA adapter."""
    logger = get_logger()

    async def _unload_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to unload LoRA adapter {lora_name}")
        await admin_client.post("/v1/unload_lora_adapter", json={"lora_name": lora_name})
        # TODO: The first one can fail, but subsequent ones should succeed.
        # response.raise_for_status()

    await asyncio.gather(*[_unload_lora_adapter(admin_client) for admin_client in admin_clients])


async def init_nccl_broadcast(admin_clients: list[AsyncClient], host: str, port: int, timeout: int) -> None:
    """Make a HTTP post request to the vLLM server to initialize the NCCL broadcast."""
    logger = get_logger()

    async def _init_nccl_broadcast(
        admin_client: AsyncClient, host: str, port: int, client_num: int, timeout: int
    ) -> None:
        try:
            response = await admin_client.post(
                "/init_broadcaster",
                json={
                    "host": host,
                    "port": port,
                    "server_rank": client_num,
                    "num_inference_server": len(admin_clients),
                    "timeout": timeout,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization.")
                return

    await asyncio.gather(
        *[
            _init_nccl_broadcast(admin_client, host, port, client_num, timeout)
            for client_num, admin_client in enumerate(admin_clients)
        ]
    )
