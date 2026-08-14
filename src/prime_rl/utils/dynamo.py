from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import (
    NCCL_MANIFEST,
    NCCL_READY_MARKER,
    InferencePool,
    check_health,
    get_nccl_chunk_manifest,
    maybe_check_has_model,
    setup_admin_clients,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import wait_for_path

DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION = 1
REQUIRED_ROUTES = frozenset(
    {
        "control/pause_generation",
        "control/resume_generation",
        "control/is_paused",
        "control/get_weight_version",
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
    world_size: int = Field(gt=0, strict=True)
    weight_transfer_backend: str = Field(min_length=1)
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
        if worker.weight_transfer_backend.strip() != "nccl":
            raise ValueError(
                f"Dynamo worker {worker.component}/{worker.instance_id} uses unsupported "
                f"weight-transfer backend {worker.weight_transfer_backend!r}"
            )
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


def topology_fingerprint(workers: tuple[DynamoWorker, ...]) -> tuple[tuple[str, str, int, str, int, str], ...]:
    return tuple(
        (
            worker.namespace,
            worker.component,
            worker.instance_id,
            worker.system_url.rstrip("/"),
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
        self._frontend_clients = setup_admin_clients(client_config.model_copy(update={"admin_base_url": None}))
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
        self._weight_update_timeout = timeout
        rank_offset = 1
        bodies = []
        for worker in self.workers:
            bodies.append(
                {
                    "init_info": {
                        "master_address": host,
                        "master_port": port,
                        "rank_offset": rank_offset,
                        "world_size": discovered_world_size + 1,
                    }
                }
            )
            rank_offset += worker.world_size
        await self._fanout("/engine/update/init_weight_transfer_engine", bodies)
        get_logger().info(f"Initialized native NCCL transfer for {discovered_world_size} Dynamo ranks")

    async def init_nixl_broadcast(self, **kwargs: Any) -> None:
        raise ValueError("Dynamo does not support NIXL weight updates yet.")

    async def _fanout(self, path: str, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self._weight_update_timeout
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Dynamo operation timed out before {path}")
        return await asyncio.wait_for(
            asyncio.gather(*(_post(client, path, body) for client, body in zip(self._admin_clients, bodies))),
            timeout=remaining,
        )

    async def _fanout_same(self, path: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        return await self._fanout(path, [body for _ in self._admin_clients])

    async def update_weights(
        self,
        weight_dir: Path | None,
        lora_name: str | None = None,
        step: int = 0,
        native_nccl: bool = False,
    ) -> None:
        if not native_nccl or lora_name is not None or weight_dir is None:
            raise ValueError("Dynamo currently supports full-model native NCCL updates only")

        deadline = time.monotonic() + self._weight_update_timeout
        await self._fanout_same("/engine/control/pause_generation", {"mode": "keep", "clear_cache": False})
        paused = await self._fanout_same("/engine/control/is_paused", {})
        if not all(result.get("is_paused") is True for result in paused):
            raise RuntimeError("Dynamo did not confirm every pinned worker was paused")

        update_succeeded = False
        try:
            (weight_dir / NCCL_MANIFEST).unlink(missing_ok=True)
            for chunk_manifest in weight_dir.glob("NCCL_CHUNK_*.json"):
                chunk_manifest.unlink()
            await self._fanout_same("/engine/update/start_weight_update", {})
            (weight_dir / NCCL_READY_MARKER).touch()

            manifest_path = weight_dir / NCCL_MANIFEST
            await asyncio.wait_for(
                wait_for_path(manifest_path, interval=0.1, log_interval=10), deadline - time.monotonic()
            )
            num_chunks = int(json.loads(manifest_path.read_text())["num_chunks"])
            for chunk_id in range(num_chunks):
                chunk_path = get_nccl_chunk_manifest(weight_dir, chunk_id)
                await asyncio.wait_for(
                    wait_for_path(chunk_path, interval=0.1, log_interval=10), deadline - time.monotonic()
                )
                update_info = json.loads(chunk_path.read_text())
                await self._fanout_same("/engine/update/update_weights", {"update_info": update_info})

            expected_version = str(step)
            await self._fanout_same("/engine/update/finish_weight_update", {"weight_version": expected_version})
            versions = await self._fanout_same("/engine/control/get_weight_version", {})
            if not all(result.get("weight_version") == expected_version for result in versions):
                raise RuntimeError(f"Dynamo workers did not commit weight version {expected_version}")
            update_succeeded = True
        finally:
            if update_succeeded:
                await self._fanout_same("/engine/control/resume_generation", {})

    async def stop(self) -> None:
        await super().stop()
        await asyncio.gather(*(client.aclose() for client in [*self._admin_clients, *self._frontend_clients]))
