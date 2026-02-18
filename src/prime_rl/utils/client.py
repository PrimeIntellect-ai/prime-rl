from __future__ import annotations

import asyncio
import os
from itertools import cycle
from pathlib import Path
from typing import Awaitable, Callable, Protocol, runtime_checkable

import httpx
import verifiers as vf
from httpx import AsyncClient
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from prime_rl.utils.config import ClientConfig
from prime_rl.utils.logger import get_logger


@runtime_checkable
class InferencePool(Protocol):
    """Protocol for inference pools (static or elastic)."""

    @property
    def clients(self) -> list[vf.ClientConfig]:
        """Get inference clients."""
        ...

    @property
    def admin_clients(self) -> list[AsyncClient]:
        """Get admin clients."""
        ...

    def update_model_name(self, model_name: str) -> None:
        """Update the model name."""
        ...

    async def get_next_client(self) -> vf.ClientConfig:
        """Get next client in round-robin fashion."""
        ...

    async def wait_for_ready(self, model_name: str, timeout: int = 1800) -> None:
        """Wait for inference pool to be ready."""
        ...

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        """Update weights on all inference servers."""
        ...

    def get_metrics(self) -> dict[str, float]:
        """Get pool metrics."""
        ...

    async def stop(self) -> None:
        """Stop the inference pool."""
        ...


class StaticInferencePool:
    """Static inference pool with fixed client list."""

    def __init__(
        self, clients: list[vf.ClientConfig], admin_clients: list[AsyncClient], skip_model_check: bool = False
    ):
        self._clients = clients
        self._admin_clients = admin_clients
        self._skip_model_check = skip_model_check
        self._idx_to_client = {client.client_idx: client for client in clients}
        self._client_cycle = cycle(clients)
        self.model_name = None  # unused

    @property
    def clients(self) -> list[vf.ClientConfig]:
        return self._clients

    @property
    def admin_clients(self) -> list[AsyncClient]:
        return self._admin_clients

    def update_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    async def get_next_client(self) -> vf.ClientConfig:
        return next(self._client_cycle)

    async def wait_for_ready(self, model_name: str, timeout: int = 1800) -> None:
        await check_health(self._admin_clients, timeout=timeout)
        await maybe_check_has_model(
            self._admin_clients,
            model_name,
            skip_model_check=self._skip_model_check,
            timeout=timeout,
        )

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        await update_weights(self._admin_clients, weight_dir, lora_name=lora_name, step=step)

    def get_metrics(self) -> dict[str, float]:
        return {}

    async def stop(self) -> None:
        pass


async def setup_inference_pool(client_config: ClientConfig, model_name: str) -> InferencePool:
    """Create an inference pool from config (static or elastic)."""
    logger = get_logger()

    if client_config.is_elastic:
        from prime_rl.utils.elastic import ElasticInferencePool

        return await ElasticInferencePool.from_config(client_config, model_name=model_name)

    logger.info(
        f"Initializing static inference pool (base_url={', '.join(client_config.base_url)}, "
        f"api_key_var={client_config.api_key_var}, headers={client_config.headers})"
    )
    return StaticInferencePool(
        clients=setup_clients(client_config),
        admin_clients=setup_admin_clients(client_config),
        skip_model_check=client_config.skip_model_check,
    )


def setup_clients(client_config: ClientConfig) -> list[vf.ClientConfig]:
    def setup_client(client_idx: int, base_url: str) -> vf.ClientConfig:
        return vf.ClientConfig(
            client_idx=client_idx,
            client_type="openai_chat_completions_token",
            api_base_url=base_url,
            api_key_var=client_config.api_key_var,
            timeout=client_config.timeout,
            max_connections=8192,
            max_keepalive_connections=8192,
            max_retries=10,
            extra_headers=client_config.headers,
        )

    return [setup_client(client_idx, base_url) for client_idx, base_url in enumerate(client_config.base_url)]


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


class ModelNotServedError(ValueError):
    """Raised when the requested model is not exposed by an otherwise-ready server."""


async def _poll_until_ready(
    *,
    base_url: str,
    probe_name: str,
    probe: Callable[[], Awaitable[None]],
    interval: float,
    log_interval: int,
    timeout: int,
    non_retryable_exceptions: tuple[type[Exception], ...] = (),
) -> None:
    logger = get_logger()
    wait_time = 0.0
    next_log_at = float(log_interval)

    while wait_time < timeout:
        try:
            await probe()
            return
        except non_retryable_exceptions:
            raise
        except (httpx.HTTPError, RuntimeError, ValueError, KeyError, TypeError) as e:
            if wait_time >= next_log_at:
                logger.warning(f"{probe_name} not ready after {wait_time:.1f}s on {base_url} (Error: {e})")
                next_log_at += float(log_interval)
            await asyncio.sleep(interval)
            wait_time += interval

    msg = f"{probe_name} timed out after {timeout}s on {base_url}."
    logger.error(msg)
    raise TimeoutError(msg)


async def maybe_check_has_model(
    admin_clients: list[AsyncClient],
    model_name: str,
    skip_model_check: bool = False,
    interval: float = 1.0,
    log_interval: int = 10,
    timeout: int = 1800,
) -> None:
    if skip_model_check:
        return
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")

    async def _check_client_model(admin_client: AsyncClient) -> None:
        async def _probe() -> None:
            response = await admin_client.get("/v1/models")
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data")
            if not isinstance(models, list):
                raise RuntimeError(
                    f"/v1/models returned unexpected payload shape on {admin_client.base_url}: {payload}"
                )

            if any(isinstance(model, dict) and model.get("id") == model_name for model in models):
                return

            available_models = [
                model.get("id") for model in models if isinstance(model, dict) and isinstance(model.get("id"), str)
            ]
            raise ModelNotServedError(
                f"Model {model_name} was not found in the inference pool on {admin_client.base_url}. "
                f"Available models: {available_models}"
            )

        await _poll_until_ready(
            base_url=str(admin_client.base_url),
            probe_name="/v1/models check",
            probe=_probe,
            interval=interval,
            log_interval=log_interval,
            timeout=timeout,
            non_retryable_exceptions=(ModelNotServedError,),
        )

    await asyncio.gather(*[_check_client_model(admin_client) for admin_client in admin_clients])
    logger.debug(f"Model {model_name} was found in the inference pool")


async def check_health(
    admin_clients: list[AsyncClient], interval: float = 1.0, log_interval: int = 10, timeout: int = 1800
) -> None:
    logger = get_logger()

    async def _check_health(admin_client: AsyncClient) -> None:
        logger.debug("Starting pinging /health to check health")

        async def _probe() -> None:
            response = await admin_client.get("/health")
            if response.status_code == 404:
                logger.warning("The route /health does not exist. Skipping health check.")
                return
            if response.status_code >= 400:
                raise RuntimeError(f"/health returned {response.status_code}")

        await _poll_until_ready(
            base_url=str(admin_client.base_url),
            probe_name="/health check",
            probe=_probe,
            interval=interval,
            log_interval=log_interval,
            timeout=timeout,
        )
        logger.debug("Inference pool health check succeeded")

    await asyncio.gather(*[_check_health(admin_client) for admin_client in admin_clients])


NCCL_READY_MARKER = "NCCL_READY"


def _is_retryable_admin_error(exception: BaseException) -> bool:
    """Retry transient upstream/proxy failures for admin endpoints."""
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code in (502, 503, 504)
    return isinstance(exception, httpx.TransportError)


def _admin_retry() -> Callable:
    return retry(
        retry=retry_if_exception(_is_retryable_admin_error),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2),
        reraise=True,
    )


async def _post_admin_with_retry(
    admin_client: AsyncClient,
    path: str,
    payload: dict[str, object],
    *,
    missing_route_warning: str | None = None,
) -> None:
    @_admin_retry()
    async def _request() -> None:
        response = await admin_client.post(path, json=payload)
        response.raise_for_status()

    try:
        await _request()
    except httpx.HTTPStatusError as e:
        if missing_route_warning is not None and e.response.status_code == 404:
            get_logger().warning(missing_route_warning)
            return
        raise


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | None,
    lora_name: str | None = None,
    step: int = 0,
) -> None:
    """Update weights on static inference servers.

    Creates a NCCL_READY marker file before calling the update endpoint to signal
    to the trainer that inference workers are about to enter the receive path.
    This marker is only used in NCCL broadcast mode but is harmless in filesystem mode.

    Note: The server-side /update_weights endpoint automatically resets the prefix cache
    to invalidate any cached KV states computed with the old weights.
    """
    logger = get_logger()

    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

    if lora_name is not None and weight_dir is not None:
        await load_lora_adapter(admin_clients, lora_name, weight_dir)
    else:
        # Create ready marker before servers enter receive path (used by NCCL broadcast)
        if weight_dir is not None:
            nccl_ready_file = weight_dir / NCCL_READY_MARKER
            nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
            nccl_ready_file.touch()
            logger.debug(f"Created NCCL_READY marker at {nccl_ready_file}")

        await asyncio.gather(
            *[
                _post_admin_with_retry(
                    admin_client,
                    "/update_weights",
                    {"weight_dir": weight_dir_posix},
                    missing_route_warning="The route /update_weights does not exist. Skipping weight update.",
                )
                for admin_client in admin_clients
            ]
        )


async def reload_weights(admin_clients: list[AsyncClient]) -> None:
    """Make a HTTP post request to the vLLM server to reload weights (reset to base model)."""
    logger = get_logger()
    logger.debug("Sending request to reload weights (reset to base model)")
    await asyncio.gather(
        *[
            _post_admin_with_retry(
                admin_client,
                "/reload_weights",
                {},
                missing_route_warning="The route /reload_weights does not exist. Skipping weight reload.",
            )
            for admin_client in admin_clients
        ]
    )


def _is_retryable_lora_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for LoRA loading."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 404 (adapter not found) or 500 (server error during loading)
        return exception.response.status_code in (404, 500)
    return False


async def load_lora_adapter(admin_clients: list[AsyncClient], lora_name: str, lora_path: Path) -> None:
    """Make a HTTP post request to the vLLM server to load a LoRA adapter.

    Uses our wrapper endpoint that also resets the prefix cache to invalidate
    KV states computed with old weights.

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
            "/load_lora_adapter",
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
