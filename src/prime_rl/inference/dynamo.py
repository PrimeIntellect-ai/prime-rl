from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION = 1
REQUIRED_ROUTES = frozenset(
    {
        "control/pause_generation",
        "control/resume_generation",
        "control/is_paused",
        "control/get_weight_version",
        "update/update_weight_version",
    }
)
NATIVE_NCCL_ROUTES = frozenset(
    {
        "update/init_weight_transfer_engine",
        "update/start_weight_update",
        "update/update_weights",
        "update/finish_weight_update",
    }
)


class DynamoWorker(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    namespace: str = Field(min_length=1)
    component: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    instance_id: int = Field(ge=0, strict=True)
    model: str = Field(min_length=1)
    request_plane_url: str = Field(min_length=1)
    system_url: str = Field(min_length=1)
    admin_base_url: str | None = Field(default=None, min_length=1)
    world_size: int | None = Field(default=None, gt=0, strict=True)
    routes: tuple[str, ...]
    error: str | None = None


class DynamoSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    protocol_version: int = Field(
        strict=True,
        ge=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
        le=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
    )
    workers: list[dict[str, Any]]


class DynamoDiscoveryPending(RuntimeError):
    pass


def client_headers(
    headers: dict[str, str],
    headers_from_env: dict[str, str],
    api_key_var: str,
) -> dict[str, str]:
    resolved = {name: value for name, env in headers_from_env.items() if (value := os.getenv(env)) is not None}
    resolved = {**headers, **resolved}
    if api_key := os.getenv(api_key_var):
        resolved["Authorization"] = f"Bearer {api_key}"
    return resolved


def parse_dynamo_workers(
    payload: object,
    model_name: str,
    *,
    require_world_size: bool = True,
) -> tuple[DynamoWorker, ...]:
    snapshot = DynamoSnapshot.model_validate(payload)
    workers: list[DynamoWorker] = []
    for raw_worker in snapshot.workers:
        if raw_worker.get("model") != model_name:
            continue
        if error := raw_worker.get("error"):
            raise DynamoDiscoveryPending(f"Dynamo worker is not ready: {error}")
        worker = DynamoWorker.model_validate(raw_worker)
        missing = REQUIRED_ROUTES.difference(worker.routes)
        if missing:
            raise DynamoDiscoveryPending(f"Dynamo worker is missing required routes: {sorted(missing)}")
        if require_world_size and worker.world_size is None:
            raise DynamoDiscoveryPending("Dynamo worker has not published world_size")
        workers.append(worker)
    if not workers:
        raise DynamoDiscoveryPending(f"Dynamo returned no bound workers for model {model_name!r}")

    identities = [(worker.namespace, worker.component, worker.endpoint, worker.instance_id) for worker in workers]
    urls = [worker.system_url.rstrip("/") for worker in workers]
    if len(set(identities)) != len(identities):
        raise ValueError("Dynamo returned duplicate worker identities")
    if len(set(urls)) != len(urls):
        raise ValueError("Dynamo returned duplicate worker control endpoints")
    return tuple(
        sorted(workers, key=lambda worker: (worker.namespace, worker.component, worker.endpoint, worker.instance_id))
    )


def topology_fingerprint(
    workers: tuple[DynamoWorker, ...],
) -> tuple[tuple[str, str, str, int, str, str | None, int | None], ...]:
    return tuple(
        (
            worker.namespace,
            worker.component,
            worker.endpoint,
            worker.instance_id,
            worker.system_url.rstrip("/"),
            worker.admin_base_url.rstrip("/") if worker.admin_base_url else None,
            worker.world_size,
        )
        for worker in workers
    )


def discover_dynamo_workers(
    discovery_url: str,
    model_name: str,
    *,
    headers: dict[str, str],
    timeout: float,
    require_world_size: bool = True,
) -> tuple[DynamoWorker, ...]:
    url = discovery_url.rstrip("/").removesuffix("/v1") + "/v1/rl/workers"
    response = httpx.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return parse_dynamo_workers(response.json(), model_name, require_world_size=require_world_size)


class DynamoAdminClients:
    """Prime's Dynamo-specific admin plane, bound to one stable worker snapshot."""

    def __init__(self, client_config, model_name: str, *, require_world_size: bool) -> None:
        from prime_rl.orchestrator.clients import setup_admin_clients

        self.client_config = client_config
        self.model_name = model_name
        self.require_world_size = require_world_size
        self.timeout = client_config.wait_for_ready_timeout
        self.headers = client_headers(
            client_config.headers,
            client_config.headers_from_env,
            client_config.api_key_var,
        )
        self.frontend_clients = setup_admin_clients(client_config.model_copy(update={"admin_base_url": None}))
        self.clients: list[httpx.AsyncClient] = []
        self.system_clients: list[httpx.AsyncClient] = []
        self.collective_clients: list[httpx.AsyncClient] = []
        self.workers: tuple[DynamoWorker, ...] = ()

    async def wait_for_ready(self, model_name: str) -> None:
        from prime_rl.orchestrator.clients import check_health, maybe_check_has_model

        if self.client_config.dynamo is None:
            raise ValueError("Dynamo discovery configuration is required")
        await check_health(self.frontend_clients, timeout=self.timeout)
        await maybe_check_has_model(
            self.frontend_clients,
            model_name,
            skip_model_check=self.client_config.skip_model_check,
        )

        deadline = time.monotonic() + self.timeout
        previous_fingerprint = None
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                workers = await asyncio.to_thread(
                    discover_dynamo_workers,
                    self.client_config.dynamo.discovery_url,
                    model_name,
                    headers=self.headers,
                    timeout=min(30.0, max(1.0, deadline - time.monotonic())),
                    require_world_size=self.require_world_size,
                )
                fingerprint = topology_fingerprint(workers)
                if fingerprint == previous_fingerprint:
                    self._bind(workers)
                    return
                previous_fingerprint = fingerprint
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    raise
                previous_fingerprint = None
                last_error = error
            except (DynamoDiscoveryPending, httpx.TransportError) as error:
                previous_fingerprint = None
                last_error = error
            await asyncio.sleep(1)
        raise TimeoutError("Dynamo workers did not become ready before the discovery timeout") from last_error

    def _bind(self, workers: tuple[DynamoWorker, ...]) -> None:
        self.workers = workers
        self.system_clients = [self._client(worker.system_url) for worker in workers]
        admin_urls = [worker.admin_base_url for worker in workers]
        if all(admin_urls):
            self.collective_clients = [self._client(url) for url in admin_urls if url is not None]
            self.clients = self.collective_clients
        else:
            self.clients = self.frontend_clients

    def _client(self, base_url: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=self.headers,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(None),
        )

    async def _fanout(
        self,
        clients: list[httpx.AsyncClient],
        path: str,
        bodies: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        async def request(client: httpx.AsyncClient, body: dict[str, Any]) -> dict[str, Any]:
            response = await client.post(path, json=body)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"Dynamo route {path} returned a non-object response")
            if payload.get("status") == "error":
                raise RuntimeError(payload.get("message", f"Dynamo route {path} failed"))
            return payload

        if len(clients) != len(bodies):
            raise ValueError(f"Dynamo route {path} received {len(bodies)} bodies for {len(clients)} clients")
        return await asyncio.wait_for(
            asyncio.gather(*(request(client, body) for client, body in zip(clients, bodies))),
            timeout=self.timeout,
        )

    async def fanout_system(self, path: str, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self._fanout(self.system_clients, path, bodies)

    async def fanout_collective(self, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(self.collective_clients) != len(self.workers):
            raise ValueError("Dynamo operation requires every worker to publish admin_base_url")
        return await self._fanout(self.collective_clients, "/collective_rpc", bodies)

    async def pause(self) -> None:
        await self.fanout_system(
            "/engine/control/pause_generation",
            [{"mode": "keep", "clear_cache": False} for _ in self.system_clients],
        )

    async def is_paused(self) -> bool:
        results = await self.fanout_system("/engine/control/is_paused", [{} for _ in self.system_clients])
        return all(result.get("is_paused") is True for result in results)

    async def resume(self) -> None:
        await self.fanout_system("/engine/control/resume_generation", [{} for _ in self.system_clients])

    async def update_weight_version(self, version: str) -> None:
        await self.fanout_system(
            "/engine/update/update_weight_version",
            [{"new_version": version} for _ in self.system_clients],
        )

    async def weight_versions(self) -> list[str | None]:
        results = await self.fanout_system("/engine/control/get_weight_version", [{} for _ in self.system_clients])
        return [result.get("weight_version") for result in results]

    async def aclose(self) -> None:
        unique = {id(client): client for client in [*self.frontend_clients, *self.system_clients, *self.collective_clients]}
        await asyncio.gather(*(client.aclose() for client in unique.values()))
