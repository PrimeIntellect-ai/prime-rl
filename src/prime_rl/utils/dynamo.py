from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
from httpx import AsyncClient
from pydantic import BaseModel, ConfigDict, Field
from tenacity import AsyncRetrying, retry_if_exception, stop_after_delay, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import StaticInferencePool, setup_admin_clients

DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION = 1
DYNAMO_READINESS_REQUEST_TIMEOUT_S = 30.0


class DiscoveredDynamoWorker(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    component: str = Field(min_length=1)
    instance_id: int = Field(ge=0, strict=True)
    model: str
    admin_base_url: str = Field(min_length=1)
    world_size: int = Field(gt=0, strict=True)


class DynamoDiscoverySnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    protocol_version: int = Field(
        strict=True,
        ge=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
        le=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
    )
    workers: list[dict[str, Any]]


class DynamoDiscoveryPending(ValueError):
    """A well-formed discovery snapshot that is not ready yet."""


def _is_retryable_dynamo_error(exception: BaseException) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code == 429 or exception.response.status_code >= 500
    return isinstance(exception, (DynamoDiscoveryPending, httpx.TransportError))


def _parse_dynamo_workers(payload: object, model_name: str) -> tuple[DiscoveredDynamoWorker, ...]:
    snapshot = DynamoDiscoverySnapshot.model_validate(payload)
    workers = []
    for raw_worker in snapshot.workers:
        if raw_worker.get("model") not in (None, model_name):
            continue
        if error := raw_worker.get("error"):
            raise DynamoDiscoveryPending(f"Dynamo RL worker probe is not ready: {error}")
        workers.append(DiscoveredDynamoWorker.model_validate(raw_worker))
    if not workers:
        raise DynamoDiscoveryPending("Dynamo RL discovery returned no workers yet")

    identities = [(worker.component, worker.instance_id) for worker in workers]
    admin_urls = [worker.admin_base_url for worker in workers]
    if len(set(identities)) != len(identities):
        raise ValueError("Dynamo RL discovery returned duplicate worker identities")
    if len(set(admin_urls)) != len(admin_urls):
        raise ValueError("Dynamo RL discovery returned duplicate admin endpoints")
    return tuple(sorted(workers, key=lambda worker: (worker.component, worker.instance_id)))


def _setup_control_clients(urls: list[str]) -> list[AsyncClient]:
    return [
        AsyncClient(
            base_url=url.rstrip("/"),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(None),
        )
        for url in urls
    ]


async def _wait_for_model(clients: list[AsyncClient], model_name: str, timeout: float) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    async with asyncio.timeout(timeout):
        async for attempt in AsyncRetrying(
            stop=stop_after_delay(timeout),
            wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
            retry=retry_if_exception(_is_retryable_dynamo_error),
            reraise=True,
        ):
            with attempt:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                request_timeout = httpx.Timeout(min(DYNAMO_READINESS_REQUEST_TIMEOUT_S, remaining))
                responses = await asyncio.gather(
                    *(client.get("/v1/models", timeout=request_timeout) for client in clients)
                )
                for response in responses:
                    response.raise_for_status()
                    models = response.json().get("data", [])
                    if not any(model.get("id") == model_name for model in models):
                        raise DynamoDiscoveryPending(f"Dynamo frontend has not published model {model_name!r}")


class DynamoInferencePool(StaticInferencePool):
    """Static request pool whose direct admin clients come from Dynamo discovery."""

    def __init__(self, client_config: ClientConfig, workers: tuple[DiscoveredDynamoWorker, ...], **kwargs):
        admin_clients = _setup_control_clients([worker.admin_base_url for worker in workers])
        super().__init__(client_config, admin_clients=admin_clients, **kwargs)
        self._admin_world_sizes = [worker.world_size for worker in workers]
        self._frontend_model_clients = setup_admin_clients(client_config)
        self._readiness_deadline: float | None = None

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        effective_timeout = self._wait_for_ready_timeout if timeout is None else timeout
        loop = asyncio.get_running_loop()
        deadline = (
            self._readiness_deadline
            if timeout is None and self._readiness_deadline is not None
            else loop.time() + effective_timeout
        )
        remaining = max(0.0, deadline - loop.time())
        try:
            async with asyncio.timeout(remaining):
                await super().wait_for_ready(model_name, timeout=remaining)
                if not self._skip_model_check:
                    await _wait_for_model(
                        self._frontend_model_clients,
                        model_name,
                        timeout=max(0.0, deadline - loop.time()),
                    )
        finally:
            self._readiness_deadline = None

    async def stop(self) -> None:
        await super().stop()
        await asyncio.gather(*(client.aclose() for client in [*self._admin_clients, *self._frontend_model_clients]))

    @classmethod
    async def from_config(
        cls,
        client_config: ClientConfig,
        model_name: str,
        expected_inference_world_size: int,
        **kwargs,
    ) -> DynamoInferencePool:
        discovery_url = cast(str, client_config.dynamo_discovery_url).rstrip("/").removesuffix("/v1")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + client_config.wait_for_ready_timeout
        async with asyncio.timeout(client_config.wait_for_ready_timeout):
            async with AsyncClient(timeout=httpx.Timeout(None)) as client:
                workers: tuple[DiscoveredDynamoWorker, ...] = ()
                async for attempt in AsyncRetrying(
                    stop=stop_after_delay(client_config.wait_for_ready_timeout),
                    wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
                    retry=retry_if_exception(_is_retryable_dynamo_error),
                    reraise=True,
                ):
                    with attempt:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            raise TimeoutError
                        response = await client.get(
                            f"{discovery_url}/v1/rl/workers",
                            timeout=httpx.Timeout(min(DYNAMO_READINESS_REQUEST_TIMEOUT_S, remaining)),
                        )
                        response.raise_for_status()
                        workers = _parse_dynamo_workers(response.json(), model_name)
                        discovered_world_size = sum(worker.world_size for worker in workers)
                        if discovered_world_size != expected_inference_world_size:
                            raise DynamoDiscoveryPending(
                                "Dynamo RL discovery returned "
                                f"inference_world_size={discovered_world_size}; "
                                f"waiting for expected inference_world_size={expected_inference_world_size}"
                            )
        pool = cls(client_config, workers, model_name=model_name, **kwargs)
        pool._readiness_deadline = deadline
        return pool
