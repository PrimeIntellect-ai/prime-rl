from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import (
    UPDATE_WEIGHTS_TIMEOUT_S,
    AdminPlane,
    _pause_engines,
    _resume_engines,
    check_health,
    maybe_check_has_model,
    setup_admin_clients,
)
from prime_rl.utils.logger import get_logger

DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION = 1


def _validate_http_url(value: str, *, field_name: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an http(s) URL with a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain a query or fragment")
    return value.rstrip("/")


class DynamoWorker(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: str = Field(min_length=1)
    component: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    instance_id: int = Field(ge=0, strict=True)
    transport: str | dict[str, str]
    request_plane_url: str = Field(min_length=1)
    system_url: str = Field(min_length=1)
    admin_base_url: str = Field(min_length=1)
    world_size: int = Field(gt=0, strict=True)
    model: str = Field(min_length=1)
    routes: tuple[str, ...]
    error: str | None = None

    @field_validator("system_url", "admin_base_url")
    @classmethod
    def validate_control_url(cls, value: str, info) -> str:
        return _validate_http_url(value, field_name=info.field_name)

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, value: str | dict[str, str]) -> str | dict[str, str]:
        if isinstance(value, str):
            if value:
                return value
            raise ValueError("transport must not be empty")
        if len(value) != 1 or any(not key or not address for key, address in value.items()):
            raise ValueError("transport must contain one named, non-empty endpoint")
        return value


class DynamoSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: int = Field(
        strict=True,
        ge=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
        le=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
    )
    namespace: str = Field(min_length=1)
    workers: tuple[dict[str, Any], ...]


class DynamoDiscoveryPending(RuntimeError):
    """The discovery endpoint is healthy but has not published a complete worker set."""


def parse_dynamo_workers(payload: object, model_name: str) -> tuple[DynamoWorker, ...]:
    snapshot = DynamoSnapshot.model_validate(payload)
    matching_workers: list[DynamoWorker] = []
    for raw_worker in snapshot.workers:
        if raw_worker.get("model") != model_name:
            continue
        if error := raw_worker.get("error"):
            raise DynamoDiscoveryPending(f"Dynamo worker is not ready: {error}")
        missing_metadata = [name for name in ("admin_base_url", "world_size") if raw_worker.get(name) is None]
        if missing_metadata:
            raise DynamoDiscoveryPending(
                f"Dynamo worker is missing required RL metadata: {', '.join(missing_metadata)}"
            )
        matching_workers.append(DynamoWorker.model_validate(raw_worker))

    if not matching_workers:
        raise DynamoDiscoveryPending(f"Dynamo returned no bound workers for model {model_name!r}")

    identities = [
        (worker.namespace, worker.component, worker.endpoint, worker.instance_id) for worker in matching_workers
    ]
    admin_urls = [worker.admin_base_url for worker in matching_workers]
    if len(set(identities)) != len(identities):
        raise ValueError("Dynamo returned duplicate worker identities")
    if len(set(admin_urls)) != len(admin_urls):
        raise ValueError("Dynamo returned duplicate worker admin endpoints")

    return tuple(
        sorted(
            matching_workers,
            key=lambda worker: (worker.namespace, worker.component, worker.endpoint, worker.instance_id),
        )
    )


def topology_fingerprint(workers: tuple[DynamoWorker, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            worker.namespace,
            worker.component,
            worker.endpoint,
            worker.instance_id,
            json.dumps(worker.transport, sort_keys=True),
            worker.request_plane_url,
            worker.system_url,
            worker.admin_base_url,
            worker.world_size,
            worker.model,
            worker.routes,
        )
        for worker in workers
    )


def _discovery_headers(client_config: ClientConfig) -> dict[str, str]:
    env_headers = {
        name: value
        for name, env_name in client_config.headers_from_env.items()
        if (value := os.getenv(env_name)) is not None
    }
    headers = {**client_config.headers, **env_headers}
    api_key = os.getenv(client_config.api_key_var)
    if api_key:
        headers = {**headers, "Authorization": f"Bearer {api_key}"}
    return headers


async def discover_dynamo_workers(
    discovery_url: str,
    model_name: str,
    *,
    headers: dict[str, str],
    timeout: float,
) -> tuple[DynamoWorker, ...]:
    base_url = _validate_http_url(discovery_url, field_name="dynamo.discovery_url")
    url = base_url.removesuffix("/v1") + "/v1/rl/workers"
    async with httpx.AsyncClient(headers=headers, timeout=timeout, trust_env=False) as client:
        response = await client.get(url)
        response.raise_for_status()
    return parse_dynamo_workers(response.json(), model_name)


class DynamoAdminPlane(AdminPlane):
    """Admin plane pinned to two identical snapshots from a trusted Dynamo discovery endpoint."""

    def __init__(
        self,
        client_config: ClientConfig,
        model_name: str,
        *,
        poll_interval: float = 1.0,
    ) -> None:
        if client_config.dynamo is None:
            raise ValueError("Dynamo discovery configuration is required")
        self._client_config = client_config
        self._model_name = model_name
        self._poll_interval = poll_interval
        self._timeout = client_config.wait_for_ready_timeout
        self._headers = _discovery_headers(client_config)
        self._frontend_clients = setup_admin_clients(client_config.model_copy(update={"admin_base_url": None}))
        self.clients: list[httpx.AsyncClient] = []
        self.workers: tuple[DynamoWorker, ...] = ()
        self._fingerprint: tuple[tuple[object, ...], ...] | None = None

    async def _discover(self) -> tuple[DynamoWorker, ...]:
        dynamo = self._client_config.dynamo
        assert dynamo is not None
        return await discover_dynamo_workers(
            dynamo.discovery_url,
            self._model_name,
            headers=self._headers,
            timeout=min(30.0, max(1.0, float(self._timeout))),
        )

    async def wait_for_ready(self, model_name: str) -> None:
        if model_name != self._model_name:
            raise ValueError(f"Dynamo admin plane was configured for {self._model_name!r}, not {model_name!r}")
        await check_health(self._frontend_clients, timeout=self._timeout)
        await maybe_check_has_model(
            self._frontend_clients,
            model_name,
            skip_model_check=self._client_config.skip_model_check,
        )

        deadline = time.monotonic() + self._timeout
        previous_fingerprint: tuple[tuple[object, ...], ...] | None = None
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                workers = await self._discover()
                fingerprint = topology_fingerprint(workers)
                if fingerprint == previous_fingerprint:
                    self._bind(workers, fingerprint)
                    await check_health(self.clients, timeout=self._timeout, quiet=True)
                    return
                previous_fingerprint = fingerprint
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    raise
                previous_fingerprint = None
                last_error = error
            except (DynamoDiscoveryPending, httpx.TransportError, ValueError) as error:
                previous_fingerprint = None
                last_error = error
            await asyncio.sleep(self._poll_interval)
        raise TimeoutError("Dynamo workers did not become ready before the discovery timeout") from last_error

    def _bind(
        self,
        workers: tuple[DynamoWorker, ...],
        fingerprint: tuple[tuple[object, ...], ...],
    ) -> None:
        self.workers = workers
        self._fingerprint = fingerprint
        self.clients = [
            httpx.AsyncClient(
                base_url=worker.admin_base_url,
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
                timeout=httpx.Timeout(None),
                trust_env=False,
            )
            for worker in workers
        ]

    async def ensure_topology_current(self) -> None:
        if self._fingerprint is None:
            raise RuntimeError("Dynamo topology has not been pinned")
        previous_changed_fingerprint: tuple[tuple[object, ...], ...] | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                workers = await self._discover()
                fingerprint = topology_fingerprint(workers)
                if fingerprint == self._fingerprint:
                    return
                if fingerprint == previous_changed_fingerprint:
                    raise RuntimeError("Dynamo worker topology changed after initialization")
                previous_changed_fingerprint = fingerprint
                last_error = None
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    raise
                last_error = error
            except (DynamoDiscoveryPending, httpx.TransportError, ValueError) as error:
                last_error = error
            if attempt < 2:
                await asyncio.sleep(self._poll_interval)
        if last_error is not None:
            raise RuntimeError("Could not verify the pinned Dynamo topology") from last_error
        raise RuntimeError("Dynamo topology changed but could not be confirmed")

    def _rank_offsets(self, inference_world_size: int) -> tuple[int, ...]:
        worker_world_sizes = tuple(worker.world_size for worker in self.workers)
        discovered_world_size = sum(worker_world_sizes)
        if discovered_world_size != inference_world_size:
            raise ValueError(
                f"Discovered worker world sizes ({discovered_world_size}) do not match "
                f"inference_world_size ({inference_world_size})"
            )

        offsets: list[int] = []
        next_offset = 0
        for worker_world_size in worker_world_sizes:
            offsets.append(next_offset)
            next_offset += worker_world_size
        return tuple(offsets)

    async def _collective_rpc(
        self,
        client: httpx.AsyncClient,
        *,
        method: Literal["init_broadcaster", "update_weights_from_path"],
        timeout: int | float,
        args: list[object],
    ) -> None:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(
                lambda error: isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
            ),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            reraise=True,
        ):
            with attempt:
                response = await asyncio.wait_for(
                    client.post(
                        "/collective_rpc",
                        json={
                            "method": method,
                            "timeout": timeout,
                            "args": args,
                            "kwargs": {},
                        },
                    ),
                    timeout=max(1.0, float(timeout) + 10.0),
                )
                response.raise_for_status()

    async def initialize_nccl(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        quantize_in_weight_transfer: bool = False,
    ) -> None:
        await self.ensure_topology_current()
        rank_offsets = self._rank_offsets(inference_world_size)
        get_logger().info(
            f"Initializing Dynamo NCCL broadcast: {len(self.clients)} servers, "
            f"inference_world_size={inference_world_size}, "
            f"worker_world_sizes={tuple(worker.world_size for worker in self.workers)}"
        )
        await asyncio.gather(
            *(
                self._collective_rpc(
                    client,
                    method="init_broadcaster",
                    timeout=timeout,
                    args=[
                        host,
                        port,
                        rank_offset,
                        inference_world_size,
                        timeout,
                        quantize_in_weight_transfer,
                        "default",
                    ],
                )
                for client, rank_offset in zip(self.clients, rank_offsets)
            )
        )

    async def update_weights(
        self,
        weight_dir: Path | None,
        *,
        transport: Literal["filesystem", "nccl", "nixl"],
        step: int = 0,
        on_paused: Callable[[], None] | None = None,
    ) -> None:
        if transport != "nccl":
            await super().update_weights(
                weight_dir,
                transport=transport,
                step=step,
                on_paused=on_paused,
            )
            return
        if weight_dir is None:
            raise ValueError("NCCL weight updates require a broadcast directory")

        await self.ensure_topology_current()
        await _pause_engines(self.clients, step=step)
        if on_paused is not None:
            on_paused()
        await asyncio.gather(
            *(
                self._collective_rpc(
                    client,
                    method="update_weights_from_path",
                    timeout=UPDATE_WEIGHTS_TIMEOUT_S,
                    args=[weight_dir.as_posix()],
                )
                for client in self.clients
            )
        )
        await _resume_engines(self.clients)

    async def aclose(self) -> None:
        unique_clients = {id(client): client for client in [*self._frontend_clients, *self.clients]}
        await asyncio.gather(*(client.aclose() for client in unique_clients.values()))
