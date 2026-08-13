from __future__ import annotations

import asyncio
from typing import Any, cast

import httpx
from httpx import AsyncClient
from pydantic import BaseModel, ConfigDict, Field
from tenacity import AsyncRetrying, retry_if_exception, stop_after_delay, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import setup_admin_clients

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


def setup_dynamo_admin_clients(
    client_config: ClientConfig,
    workers: tuple[DiscoveredDynamoWorker, ...],
) -> list[AsyncClient]:
    return setup_admin_clients(client_config, [worker.admin_base_url for worker in workers])


async def discover_dynamo_workers(
    client_config: ClientConfig,
    model_name: str,
    expected_inference_world_size: int,
) -> tuple[DiscoveredDynamoWorker, ...]:
    discovery_url = cast(str, client_config.dynamo_discovery_url).rstrip("/").removesuffix("/v1")
    timeout = client_config.wait_for_ready_timeout
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    async with asyncio.timeout(timeout):
        async with AsyncClient(timeout=httpx.Timeout(None)) as client:
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
                    response = await client.get(
                        f"{discovery_url}/v1/rl/workers",
                        timeout=httpx.Timeout(min(DYNAMO_READINESS_REQUEST_TIMEOUT_S, remaining)),
                    )
                    response.raise_for_status()
                    workers = _parse_dynamo_workers(response.json(), model_name)
                    discovered_world_size = sum(worker.world_size for worker in workers)
                    if discovered_world_size != expected_inference_world_size:
                        raise DynamoDiscoveryPending(
                            f"Dynamo discovery returned inference_world_size={discovered_world_size}; "
                            f"waiting for expected inference_world_size={expected_inference_world_size}"
                        )
                    return workers
    raise TimeoutError("Dynamo worker discovery timed out")
