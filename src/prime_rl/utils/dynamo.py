from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import (
    NCCL_READY_MARKER,
    InferencePool,
    check_health,
    maybe_check_has_model,
    setup_admin_clients,
)
from prime_rl.utils.logger import get_logger

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
    instance_id: int = Field(ge=0, strict=True)
    model: str = Field(min_length=1)
    system_url: str = Field(min_length=1)
    admin_base_url: str | None = Field(default=None, min_length=1)
    world_size: int = Field(gt=0, strict=True)
    weight_transfer_backend: str | None = Field(default=None, min_length=1)
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


class DynamoVLLMWeightSyncClient:
    def __init__(self, workers: tuple[DynamoWorker, ...], headers: dict[str, str], timeout: float) -> None:
        self.workers = workers
        self.clients = [
            httpx.Client(base_url=worker.system_url.rstrip("/"), headers=headers, timeout=timeout) for worker in workers
        ]

    @staticmethod
    def _validate(response: httpx.Response, path: str) -> None:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Dynamo route {path} returned a non-object response")
        if payload.get("status") == "error":
            raise RuntimeError(payload.get("message", f"Dynamo route {path} failed"))

    def _fanout(self, path: str, bodies: list[dict[str, Any]]) -> None:
        def request(client: httpx.Client, body: dict[str, Any]) -> None:
            self._validate(client.post(path, json=body), path)

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = [executor.submit(request, client, body) for client, body in zip(self.clients, bodies)]
            for future in futures:
                future.result()

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        rank_offset = 1
        bodies = []
        for worker in self.workers:
            bodies.append({"init_info": {**init_info, "rank_offset": rank_offset}})
            rank_offset += worker.world_size
        self._fanout("/engine/update/init_weight_transfer_engine", bodies)

    def start_weight_update(self) -> None:
        self._fanout("/engine/update/start_weight_update", [{} for _ in self.clients])

    def update_weights(self, update_info: dict[str, Any]) -> None:
        self._fanout("/engine/update/update_weights", [{"update_info": update_info} for _ in self.clients])

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self._fanout(
            "/engine/update/finish_weight_update",
            [{"weight_version": weight_version} for _ in self.clients],
        )

    def update_weight_version(self, weight_version: str) -> None:
        self._fanout(
            "/engine/update/update_weight_version",
            [{"new_version": weight_version} for _ in self.clients],
        )


def client_headers(
    headers: dict[str, str],
    headers_from_env: dict[str, str],
    api_key_var: str,
) -> dict[str, str]:
    resolved = {name: value for name, env in headers_from_env.items() if (value := os.getenv(env)) is not None}
    resolved = {**headers, **resolved}
    api_key = os.getenv(api_key_var)
    if api_key:
        resolved["Authorization"] = f"Bearer {api_key}"
    return resolved


def parse_dynamo_workers(payload: object, model_name: str) -> tuple[DynamoWorker, ...]:
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
            raise DynamoDiscoveryPending(f"Dynamo worker is missing native routes: {sorted(missing)}")
        workers.append(worker)
    if not workers:
        raise DynamoDiscoveryPending(f"Dynamo returned no bound workers for model {model_name!r}")

    identities = [(worker.namespace, worker.component, worker.instance_id) for worker in workers]
    urls = [worker.system_url.rstrip("/") for worker in workers]
    if len(set(identities)) != len(identities):
        raise ValueError("Dynamo returned duplicate worker identities")
    if len(set(urls)) != len(urls):
        raise ValueError("Dynamo returned duplicate worker control endpoints")
    return tuple(sorted(workers, key=lambda worker: (worker.namespace, worker.component, worker.instance_id)))


def topology_fingerprint(
    workers: tuple[DynamoWorker, ...],
) -> tuple[tuple[str, str, int, str, str | None, int, str | None], ...]:
    return tuple(
        (
            worker.namespace,
            worker.component,
            worker.instance_id,
            worker.system_url.rstrip("/"),
            worker.admin_base_url.rstrip("/") if worker.admin_base_url else None,
            worker.world_size,
            worker.weight_transfer_backend,
        )
        for worker in workers
    )


def discover_dynamo_workers(
    discovery_url: str,
    model_name: str,
    *,
    headers: dict[str, str],
    timeout: float,
) -> tuple[DynamoWorker, ...]:
    url = discovery_url.rstrip("/").removesuffix("/v1") + "/v1/rl/workers"
    response = httpx.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return parse_dynamo_workers(response.json(), model_name)


async def _post(client: httpx.AsyncClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(path, json=body)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(payload.get("message", f"Dynamo route {path} failed"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Dynamo route {path} returned a non-object response")
    return payload


class DynamoInferencePool(InferencePool):
    def __init__(
        self,
        client_config: ClientConfig,
        workers: tuple[DynamoWorker, ...],
        model_name: str,
        **kwargs,
    ) -> None:
        self.workers = workers
        self._weight_update_timeout = client_config.wait_for_ready_timeout
        self._weight_update_backend: str | None = None
        self._frontend_clients = setup_admin_clients(client_config.model_copy(update={"admin_base_url": None}))
        admin_urls = [worker.admin_base_url for worker in workers]
        self._collective_rpc_clients = (
            setup_admin_clients(
                client_config.model_copy(update={"admin_base_url": [url for url in admin_urls if url is not None]})
            )
            if all(admin_urls)
            else []
        )
        super().__init__(
            client_config.model_copy(update={"admin_base_url": None}),
            model_name,
            admin_clients=setup_admin_clients(
                client_config.model_copy(update={"admin_base_url": [worker.system_url for worker in workers]})
            ),
            **kwargs,
        )

    @classmethod
    async def from_config(
        cls,
        client_config: ClientConfig,
        model_name: str,
        *,
        inference_world_size: int | None,
        **kwargs,
    ) -> DynamoInferencePool:
        if client_config.dynamo is None:
            raise ValueError("Dynamo configuration is required")
        headers = client_headers(
            client_config.headers,
            client_config.headers_from_env,
            client_config.api_key_var,
        )
        deadline = time.monotonic() + client_config.wait_for_ready_timeout
        last_error: Exception | None = None
        previous_fingerprint = None
        while time.monotonic() < deadline:
            try:
                workers = await asyncio.to_thread(
                    discover_dynamo_workers,
                    client_config.dynamo.discovery_url,
                    model_name,
                    headers=headers,
                    timeout=min(30.0, max(1.0, deadline - time.monotonic())),
                )
                discovered_world_size = sum(worker.world_size for worker in workers)
                if inference_world_size is not None and discovered_world_size != inference_world_size:
                    raise DynamoDiscoveryPending(
                        f"Dynamo world size {discovered_world_size} does not match Prime inference capacity "
                        f"{inference_world_size}"
                    )
                fingerprint = topology_fingerprint(workers)
                if fingerprint == previous_fingerprint:
                    return cls(client_config, workers, model_name, **kwargs)
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

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        await check_health(
            self._frontend_clients,
            timeout=timeout if timeout is not None else self._wait_for_ready_timeout,
        )
        await maybe_check_has_model(self._frontend_clients, model_name, skip_model_check=self._skip_model_check)

    async def init_nccl_broadcast(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int | None,
    ) -> None:
        discovered_world_size = sum(worker.world_size for worker in self.workers)
        if inference_world_size is not None and inference_world_size != discovered_world_size:
            raise ValueError(
                f"Configured inference_world_size={inference_world_size} does not match Dynamo "
                f"world size {discovered_world_size}"
            )
        for worker in self.workers:
            missing = NATIVE_NCCL_ROUTES.difference(worker.routes)
            if worker.weight_transfer_backend != "nccl" or missing:
                raise ValueError(
                    f"Dynamo worker {worker.component}/{worker.instance_id} does not support native NCCL "
                    f"weight transfer (backend={worker.weight_transfer_backend!r}, missing_routes={sorted(missing)})"
                )
        self._weight_update_timeout = timeout
        self._weight_update_backend = "nccl"
        get_logger().info(f"Dynamo trainer will initialize native NCCL transfer for {discovered_world_size} ranks")

    async def init_nixl_broadcast(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        session_id: str,
    ) -> None:
        discovered_world_size = sum(worker.world_size for worker in self.workers)
        if inference_world_size != discovered_world_size:
            raise ValueError(
                f"Configured inference_world_size={inference_world_size} does not match Dynamo "
                f"world size {discovered_world_size}"
            )
        if len(self._collective_rpc_clients) != len(self.workers):
            missing = [
                f"{worker.component}/{worker.instance_id}" for worker in self.workers if worker.admin_base_url is None
            ]
            raise ValueError(
                "Dynamo NIXL requires vLLM HTTP admin endpoints for /collective_rpc; "
                f"workers missing admin_base_url: {missing}"
            )

        rank_offset = 0
        bodies: list[dict[str, Any]] = []
        for worker in self.workers:
            bodies.append(
                {
                    "method": "init_broadcaster",
                    "timeout": timeout,
                    "args": [host, port, rank_offset, inference_world_size, timeout, session_id],
                    "kwargs": {},
                }
            )
            rank_offset += worker.world_size
        self._weight_update_timeout = timeout
        await self._fanout_to(self._collective_rpc_clients, "/collective_rpc", bodies)
        self._weight_update_backend = "nixl"
        get_logger().info(f"Dynamo initialized Prime NIXL transfer for {discovered_world_size} ranks")

    async def _fanout_to(
        self,
        clients: list[httpx.AsyncClient],
        path: str,
        bodies: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self._weight_update_timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Dynamo operation timed out before {path}")
        return await asyncio.wait_for(
            asyncio.gather(*(_post(client, path, body) for client, body in zip(clients, bodies))),
            timeout=remaining,
        )

    async def _fanout(self, path: str, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self._fanout_to(self._admin_clients, path, bodies)

    async def _fanout_same(self, path: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        return await self._fanout(path, [body for _ in self._admin_clients])

    async def update_weights(
        self,
        weight_dir: Path | None,
        lora_name: str | None = None,
        step: int = 0,
        native_nccl: bool = False,
    ) -> None:
        if lora_name is not None:
            raise ValueError("Dynamo does not support LoRA weight updates yet")
        if native_nccl and weight_dir is None:
            raise ValueError("Dynamo native NCCL updates require a weight directory")
        if not native_nccl and weight_dir is None and self._weight_update_backend != "nixl":
            raise ValueError("Dynamo custom weight updates require an initialized NIXL broadcaster")
        if not native_nccl and len(self._collective_rpc_clients) != len(self.workers):
            raise ValueError("Dynamo filesystem and NIXL updates require vLLM HTTP admin endpoints for /collective_rpc")

        await self._fanout_same("/engine/control/pause_generation", {"mode": "keep", "clear_cache": False})
        paused = await self._fanout_same("/engine/control/is_paused", {})
        if not all(result.get("is_paused") is True for result in paused):
            raise RuntimeError("Dynamo did not confirm every pinned worker was paused")

        expected_version = str(step)
        if native_nccl:
            assert weight_dir is not None
            (weight_dir / NCCL_READY_MARKER).touch()
        else:
            if self._weight_update_backend == "nixl":
                collective_body = {
                    "method": "update_weights_from_path",
                    "timeout": self._weight_update_timeout,
                    "args": [None],
                    "kwargs": {},
                }
            else:
                assert weight_dir is not None
                collective_body = {
                    "method": "reload_weights",
                    "timeout": self._weight_update_timeout,
                    "args": [],
                    "kwargs": {"weights_path": weight_dir.as_posix()},
                }
            await self._fanout_to(
                self._collective_rpc_clients,
                "/collective_rpc",
                [collective_body for _ in self._collective_rpc_clients],
            )
            await self._fanout_same("/engine/update/update_weight_version", {"new_version": expected_version})

        deadline = time.monotonic() + self._weight_update_timeout
        while time.monotonic() < deadline:
            versions = await self._fanout_same("/engine/control/get_weight_version", {})
            if all(result.get("weight_version") == expected_version for result in versions):
                await self._fanout_same("/engine/control/resume_generation", {})
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Dynamo workers did not commit weight version {expected_version}; engines remain paused")

    async def stop(self) -> None:
        await super().stop()
        await asyncio.gather(
            *(
                client.aclose()
                for client in [*self._admin_clients, *self._frontend_clients, *self._collective_rpc_clients]
            )
        )
