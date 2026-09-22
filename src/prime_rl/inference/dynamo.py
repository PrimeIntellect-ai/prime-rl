from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable
from ipaddress import ip_address
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import (
    ADMIN_TIMEOUT_S,
    UPDATE_WEIGHTS_TIMEOUT_S,
    AdminPlane,
    _admin_post,
    _is_retryable_admin_error,
    check_health,
    maybe_check_has_model,
    setup_admin_clients,
)
from prime_rl.utils.logger import get_logger

_PYTHON_ENGINE_ROUTES = frozenset(
    {
        "init_weights_update_group",
        "pause_generation",
        "resume_generation",
        "update_weights_from_distributed",
    }
)
_PYTHON_LORA_ROUTES = frozenset({"load_lora", "unload_lora"})
_DYNAMO_LORA_RESIDENT_VERSIONS = 2
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_WILDCARD_HOSTS = {"0.0.0.0", "::"}


class DynamoWorker(BaseModel):
    admin_base_url: str
    instance_id: int = Field(ge=0, strict=True)
    world_size: int = Field(gt=0, strict=True)
    admin_contract: Literal["collective_rpc", "engine_routes"] = "collective_rpc"
    routes: tuple[str, ...] = ()


class DynamoSnapshot(BaseModel):
    protocol_version: Literal[1]
    workers: tuple[dict[str, object], ...]


class DynamoDiscoveryPending(RuntimeError):
    """The discovery endpoint is healthy but has not published a complete worker set."""


def parse_dynamo_worker(
    payload: object,
    model_name: str,
    *,
    expected_admin_host: str | None = None,
) -> DynamoWorker:
    snapshot = DynamoSnapshot.model_validate(payload)
    matching_workers: list[DynamoWorker] = []
    for raw_worker in snapshot.workers:
        if raw_worker.get("model") != model_name:
            continue
        if raw_worker.get("error") is not None:
            raise DynamoDiscoveryPending("Dynamo worker is not ready")
        admin_base_url = raw_worker.get("admin_base_url")
        admin_contract: Literal["collective_rpc", "engine_routes"] = "collective_rpc"
        routes: tuple[str, ...] = ()
        if admin_base_url is None:
            admin_base_url = raw_worker.get("system_url")
            if admin_base_url is None:
                raise DynamoDiscoveryPending("Dynamo worker is missing admin_base_url or system_url")
            raw_routes = raw_worker.get("routes")
            if not isinstance(raw_routes, list) or not all(isinstance(route, str) for route in raw_routes):
                raise TypeError("Dynamo worker routes must be a list of strings")
            routes = tuple(raw_routes)
            missing_routes = _PYTHON_ENGINE_ROUTES.difference(routes)
            if missing_routes:
                raise DynamoDiscoveryPending(
                    "Dynamo worker is missing required admin routes: " + ", ".join(sorted(missing_routes))
                )
            admin_contract = "engine_routes"
        worker = DynamoWorker.model_validate(
            {
                **raw_worker,
                "admin_base_url": admin_base_url,
                "admin_contract": admin_contract,
                "routes": routes,
            }
        )
        if worker.admin_contract == "collective_rpc" and worker.world_size != 1:
            raise ValueError("Dynamo sidecar currently supports exactly one inference rank")
        try:
            admin_url = httpx.URL(worker.admin_base_url)
        except httpx.InvalidURL as error:
            raise ValueError("admin_base_url must be a valid URL") from error
        if (
            admin_url.scheme not in {"http", "https"}
            or not admin_url.host
            or (admin_url.port is not None and not 1 <= admin_url.port <= 65535)
            or admin_url.userinfo
            or admin_url.query
            or admin_url.fragment
            or admin_url.path != "/"
        ):
            raise ValueError("admin_base_url must be an http(s) origin")
        if admin_url.host in _WILDCARD_HOSTS:
            raise ValueError("Dynamo worker admin origin must not use a wildcard host")
        if expected_admin_host is not None:
            if worker.admin_contract == "collective_rpc" and admin_url.host != expected_admin_host:
                raise ValueError(
                    f"Dynamo worker admin host {admin_url.host!r} does not match discovery host {expected_admin_host!r}"
                )
            if worker.admin_contract == "engine_routes":
                try:
                    admin_ip = ip_address(admin_url.host)
                except ValueError as error:
                    raise ValueError("Dynamo Python worker system_url must use an IP address") from error
                admin_ip = getattr(admin_ip, "ipv4_mapped", None) or admin_ip
                if admin_ip.is_unspecified:
                    raise ValueError("Dynamo worker admin origin must not use a wildcard host")
                if admin_ip.is_loopback and expected_admin_host not in _LOOPBACK_HOSTS:
                    raise ValueError("Remote Dynamo discovery must not redirect administration to loopback")
                if admin_ip.is_link_local or admin_ip.is_multicast or admin_ip.is_reserved:
                    raise ValueError("Dynamo Python worker system_url uses an unsafe IP address")
        matching_workers.append(worker)

    if not matching_workers:
        raise DynamoDiscoveryPending(f"Dynamo returned no bound workers for model {model_name!r}")
    if len(matching_workers) != 1:
        raise ValueError("Dynamo RL currently supports exactly one inference worker")
    return matching_workers[0]


def topology_fingerprint(worker: DynamoWorker) -> tuple[int, str, int, str, tuple[str, ...]]:
    capabilities = tuple(sorted((_PYTHON_ENGINE_ROUTES | _PYTHON_LORA_ROUTES).intersection(worker.routes)))
    return (
        worker.instance_id,
        str(httpx.URL(worker.admin_base_url)),
        worker.world_size,
        worker.admin_contract,
        capabilities,
    )


def _discovery_headers(client_config: ClientConfig) -> dict[str, str]:
    env_headers = {
        name: value
        for name, env_name in client_config.headers_from_env.items()
        if (value := os.getenv(env_name)) is not None
    }
    headers = {**client_config.headers, **env_headers}
    api_key = os.getenv(client_config.api_key_var)
    if api_key and api_key != "EMPTY":
        headers = {name: value for name, value in headers.items() if name.lower() != "authorization"}
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def resolve_dynamo_discovery_url(client_config: ClientConfig) -> str:
    dynamo = client_config.dynamo
    if dynamo is None or not dynamo.enabled:
        raise ValueError("Dynamo discovery is not enabled")
    if dynamo.discovery_url is not None:
        return dynamo.discovery_url

    try:
        base_url = httpx.URL(client_config.base_url)
    except httpx.InvalidURL as error:
        raise ValueError("Set dynamo.discovery_url when client.base_url is not a valid URL") from error
    if base_url.port is None or base_url.port == 65535:
        raise ValueError("Set dynamo.discovery_url when client.base_url has no incrementable port")
    return str(base_url.copy_with(port=base_url.port + 1, path="", query=None, fragment=None, userinfo=b""))


async def discover_dynamo_worker(
    discovery_url: str,
    model_name: str,
    *,
    headers: dict[str, str],
    timeout: float,
) -> DynamoWorker:
    try:
        base_url = httpx.URL(discovery_url)
    except httpx.InvalidURL as error:
        raise ValueError("dynamo.discovery_url must be a valid URL") from error
    if (
        base_url.scheme not in {"http", "https"}
        or not base_url.host
        or (base_url.port is not None and not 1 <= base_url.port <= 65535)
        or base_url.userinfo
        or base_url.query
        or base_url.fragment
    ):
        raise ValueError("dynamo.discovery_url must be an http(s) URL without credentials, query, or fragment")
    base_path = base_url.path.rstrip("/")
    if base_path.endswith("/v1"):
        base_path = base_path.removesuffix("/v1")
    url = base_url.copy_with(path=f"{base_path}/v1/rl/workers")
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(headers=headers, timeout=timeout, trust_env=False) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
    except TimeoutError as error:
        raise TimeoutError(f"Dynamo discovery request exceeded {timeout} seconds") from error
    return parse_dynamo_worker(payload, model_name, expected_admin_host=base_url.host)


class DynamoAdminPlane(AdminPlane):
    """Admin plane pinned to two identical snapshots from a trusted Dynamo discovery endpoint."""

    def __init__(
        self,
        client_config: ClientConfig,
        model_name: str,
        *,
        poll_interval: float = 1.0,
    ) -> None:
        if client_config.dynamo is None or not client_config.dynamo.enabled:
            raise ValueError("Dynamo discovery configuration is required")
        self._discovery_url = resolve_dynamo_discovery_url(client_config)
        self._client_config = client_config
        self._model_name = model_name
        self._poll_interval = poll_interval
        self._timeout = client_config.wait_for_ready_timeout
        self._headers = _discovery_headers(client_config)
        self._frontend_clients = setup_admin_clients(client_config.model_copy(update={"admin_base_url": None}))
        self.clients: list[httpx.AsyncClient] = []
        self._fingerprint: tuple[int, str, int, str, tuple[str, ...]] | None = None
        self._admin_contract: Literal["collective_rpc", "engine_routes"] = "collective_rpc"
        self._inference_world_size = 1
        self._worker_routes: frozenset[str] = frozenset()
        self._lora_names_by_step: dict[int, str] = {}
        self._loaded_loras: list[str] = []
        self._nccl_initialization_state: Literal["uninitialized", "initializing", "ready", "terminal"] = "uninitialized"
        self._mutation_lock = asyncio.Lock()

    def _require_uninitialized_nccl(self) -> None:
        if self._nccl_initialization_state != "uninitialized":
            raise RuntimeError(
                "Dynamo NCCL can only be initialized once; restart the inference workers before retrying"
            )

    def _require_ready_nccl(self) -> None:
        if self._nccl_initialization_state != "ready":
            raise RuntimeError("Dynamo weight updates require a ready NCCL initialization")

    def _terminalize_nccl(self) -> None:
        self._nccl_initialization_state = "terminal"

    async def _discover(self) -> DynamoWorker:
        return await discover_dynamo_worker(
            self._discovery_url,
            self._model_name,
            headers=self._headers,
            timeout=min(30.0, max(1.0, float(self._timeout))),
        )

    async def wait_for_ready(self, model_name: str) -> None:
        if model_name != self._model_name:
            raise ValueError(f"Dynamo admin plane was configured for {self._model_name!r}, not {model_name!r}")
        deadline = time.monotonic() + self._timeout
        try:
            async with asyncio.timeout(self._remaining(deadline)):
                await check_health(self._frontend_clients, timeout=self._remaining(deadline))
                await maybe_check_has_model(
                    self._frontend_clients,
                    model_name,
                    skip_model_check=self._client_config.skip_model_check,
                )
        except TimeoutError as error:
            raise TimeoutError(f"Dynamo frontend readiness exceeded {self._timeout} seconds") from error

        previous_fingerprint: tuple[int, str, int, str, tuple[str, ...]] | None = None
        last_error: Exception | None = None
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                async with asyncio.timeout(remaining):
                    worker = await self._discover()
                fingerprint = topology_fingerprint(worker)
                if fingerprint == previous_fingerprint:
                    candidate_client = self._make_worker_client(worker)
                    try:
                        remaining = self._remaining(deadline)
                        async with asyncio.timeout(remaining):
                            await check_health([candidate_client], timeout=remaining, quiet=True)
                    except BaseException:
                        await candidate_client.aclose()
                        raise
                    self._bind(worker, fingerprint, candidate_client)
                    return
                previous_fingerprint = fingerprint
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    raise
                previous_fingerprint = None
                last_error = error
            except (DynamoDiscoveryPending, httpx.TransportError, TimeoutError) as error:
                previous_fingerprint = None
                last_error = error
            remaining = deadline - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(min(self._poll_interval, remaining))
        raise TimeoutError("Dynamo workers did not become ready before the discovery timeout") from last_error

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Dynamo readiness deadline expired")
        return remaining

    def _make_worker_client(self, worker: DynamoWorker) -> httpx.AsyncClient:
        worker_timeout = min(30.0, max(1.0, float(self._timeout)))
        return httpx.AsyncClient(
            base_url=worker.admin_base_url,
            headers=self._headers if worker.admin_contract == "collective_rpc" else None,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(worker_timeout),
            trust_env=False,
        )

    def _bind(
        self,
        worker: DynamoWorker,
        fingerprint: tuple[int, str, int, str, tuple[str, ...]],
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._fingerprint = fingerprint
        self._admin_contract = worker.admin_contract
        self._inference_world_size = worker.world_size
        self.clients = [client if client is not None else self._make_worker_client(worker)]
        self._worker_routes = frozenset(worker.routes)

    async def ensure_topology_current(self) -> None:
        if self._nccl_initialization_state == "terminal":
            raise RuntimeError("Dynamo administration is in a terminal state; restart is required")
        if self._fingerprint is None:
            raise RuntimeError("Dynamo topology has not been pinned")
        previous_changed_fingerprint: tuple[int, str, int, str, tuple[str, ...]] | None = None
        last_error: Exception | None = None
        deadline = time.monotonic() + self._timeout
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
                    worker = await self._discover()
                fingerprint = topology_fingerprint(worker)
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
            except (DynamoDiscoveryPending, httpx.TransportError, TimeoutError) as error:
                last_error = error
            if attempt < 2:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(self._poll_interval, remaining))
        if last_error is not None:
            raise RuntimeError("Could not verify the pinned Dynamo topology") from last_error
        raise RuntimeError("Dynamo topology changed but could not be confirmed")

    async def _collective_rpc(
        self,
        client: httpx.AsyncClient,
        *,
        method: Literal["init_broadcaster", "update_weights_from_path"],
        timeout: int | float,
        args: list[object],
    ) -> None:
        operation_timeout = max(1.0, float(timeout))
        async with asyncio.timeout(operation_timeout + 15.0):
            response = await client.post(
                "/collective_rpc",
                timeout=httpx.Timeout(connect=10.0, read=operation_timeout, write=10.0, pool=10.0),
                json={"method": method, "timeout": operation_timeout, "args": args, "kwargs": {}},
            )
            response.raise_for_status()
            payload = response.json()
        if payload != {"results": [None]}:
            raise ValueError("Dynamo worker returned an invalid collective RPC response")

    async def _engine_post(
        self,
        path: str,
        body: dict[str, object],
        timeout: int | float,
    ) -> None:
        operation_timeout = max(1.0, float(timeout))
        async with asyncio.timeout(operation_timeout + 15.0):
            response = await self.clients[0].post(
                path,
                timeout=httpx.Timeout(connect=10.0, read=operation_timeout, write=10.0, pool=10.0),
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise ValueError(f"Dynamo worker route {path} returned an error")

    async def _set_generation_paused(self, paused: bool) -> None:
        if self._admin_contract == "engine_routes":
            operation = "pause_generation" if paused else "resume_generation"
            body = {"mode": "keep", "clear_cache": False} if paused else {}
            async with asyncio.timeout(2 * ADMIN_TIMEOUT_S):
                for attempt in range(10):
                    try:
                        await self._engine_post(
                            f"/engine/{operation}",
                            body,
                            ADMIN_TIMEOUT_S,
                        )
                        return
                    except BaseException as error:
                        if attempt == 9 or not _is_retryable_admin_error(error):
                            raise
                        await asyncio.sleep(min(2**attempt, 10))
        if paused:
            await _admin_post(self.clients[0], "/pause", params={"mode": "keep", "clear_cache": "false"})
        else:
            await _admin_post(self.clients[0], "/resume", timeout_s=ADMIN_TIMEOUT_S)

    async def update_weights(
        self,
        weight_dir: Path | None,
        *,
        transport: Literal["filesystem", "nccl", "nixl"],
        step: int = 0,
        on_paused: Callable[[], None] | None = None,
    ) -> None:
        if self._admin_contract == "engine_routes" and transport != "nccl":
            raise ValueError("A Dynamo Python worker currently only supports NCCL weight updates")
        if transport != "nccl":
            async with self._mutation_lock:
                await self.ensure_topology_current()
                await super().update_weights(
                    weight_dir,
                    transport=transport,
                    step=step,
                    on_paused=on_paused,
                )
            return
        if weight_dir is None:
            raise ValueError(f"{transport.upper()} weight updates require a broadcast directory")
        self._require_ready_nccl()
        async with self._mutation_lock:
            await self.ensure_topology_current()
            self._nccl_initialization_state = "terminal"

            try:
                await self._set_generation_paused(True)
            except BaseException as failure:
                raise RuntimeError("Dynamo pause failed; worker state is unknown and restart is required") from failure

            if on_paused is not None:
                try:
                    on_paused()
                except BaseException as error:
                    raise RuntimeError(
                        "Dynamo pause callback failed; engines remain paused and restart is required"
                    ) from error
            try:
                if self._admin_contract == "engine_routes":
                    await self._engine_post(
                        "/engine/update_weights_from_distributed",
                        {
                            "engine_rpc": "update_weights_from_path",
                            "weight_dir": weight_dir.as_posix(),
                            "weight_version": step,
                        },
                        UPDATE_WEIGHTS_TIMEOUT_S,
                    )
                else:
                    await self._collective_rpc(
                        self.clients[0],
                        method="update_weights_from_path",
                        timeout=UPDATE_WEIGHTS_TIMEOUT_S,
                        args=[weight_dir.as_posix()],
                    )
            except BaseException as failure:
                self._terminalize_nccl()
                raise RuntimeError(
                    f"Dynamo {transport} update failed; engines remain paused and restart is required"
                ) from failure

            try:
                await self._set_generation_paused(False)
            except BaseException as failure:
                self._terminalize_nccl()
                raise RuntimeError("Dynamo resume failed; worker state is unknown and restart is required") from failure
            self._nccl_initialization_state = "ready"
            get_logger().info(f"Applied NCCL weights for policy v{step} to the Dynamo worker")

    def _require_lora_routes(self) -> None:
        missing_routes = _PYTHON_LORA_ROUTES.difference(self._worker_routes)
        if missing_routes:
            raise RuntimeError("Dynamo worker is missing required LoRA routes: " + ", ".join(sorted(missing_routes)))

    def _lora_name(self, step: int) -> str:
        if step not in self._lora_names_by_step:
            self._lora_names_by_step[step] = f"prime-rl-policy-v{step}-{uuid.uuid4().hex}"
        return self._lora_names_by_step[step]

    async def _post_lora(
        self,
        operation: Literal["load_lora", "unload_lora"],
        body: dict[str, object],
    ) -> None:
        expected_name = body["lora_name"]
        response = await self.clients[0].post(
            f"/engine/{operation}",
            json=body,
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
        )
        response.raise_for_status()
        payload = response.json()
        if (
            not isinstance(payload, dict)
            or payload.get("status") != "success"
            or payload.get("lora_name") != expected_name
            or type(payload.get("lora_id")) is not int
            or payload["lora_id"] <= 0
        ):
            raise RuntimeError(f"Dynamo worker rejected LoRA operation: {payload!r}")

    async def _wait_for_lora_model(self, model_name: str) -> None:
        deadline = time.monotonic() + min(float(self._timeout), 120.0)
        last_error: Exception | None = None
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                async with asyncio.timeout(remaining):
                    await maybe_check_has_model(self._frontend_clients, model_name)
                return
            except (RuntimeError, ValueError, httpx.TransportError, TimeoutError) as error:
                last_error = error
            await asyncio.sleep(min(self._poll_interval, remaining))
        raise TimeoutError(f"Dynamo LoRA adapter {model_name!r} did not become routable") from last_error

    async def load_lora_adapter(self, lora_name: str, lora_path: Path, *, step: int) -> str:
        if self._admin_contract != "engine_routes":
            return await super().load_lora_adapter(lora_name, lora_path, step=step)
        if lora_name != self._model_name:
            raise ValueError(f"Dynamo admin plane was configured for {self._model_name!r}, not {lora_name!r}")

        source_uri = lora_path.resolve(strict=True).as_uri()
        active_model = self._lora_name(step)
        async with self._mutation_lock:
            self._require_lora_routes()
            await self.ensure_topology_current()
            if active_model in self._loaded_loras:
                await self._wait_for_lora_model(active_model)
                return active_model
            if len(self._loaded_loras) >= _DYNAMO_LORA_RESIDENT_VERSIONS:
                oldest = self._loaded_loras[0]
                await self._post_lora("unload_lora", {"lora_name": oldest})
                self._loaded_loras.pop(0)
            await self._post_lora(
                "load_lora",
                {"lora_name": active_model, "source": {"uri": source_uri}},
            )
            try:
                await self._wait_for_lora_model(active_model)
            except BaseException:
                try:
                    await self._post_lora("unload_lora", {"lora_name": active_model})
                except Exception as cleanup_error:
                    get_logger().warning(f"Failed to roll back Dynamo LoRA {active_model}: {cleanup_error!r}")
                raise
            self._loaded_loras.append(active_model)

        get_logger().info(f"Activated Dynamo LoRA {active_model} for policy v{step}")
        return active_model

    async def initialize_nccl(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
    ) -> None:
        async with self._mutation_lock:
            self._require_uninitialized_nccl()
            if inference_world_size != self._inference_world_size:
                if self._admin_contract == "collective_rpc":
                    raise ValueError("Dynamo sidecar currently supports exactly one inference rank")
                raise ValueError("Configured inference world size does not match Dynamo discovery")
            self._nccl_initialization_state = "initializing"
            try:
                await self.ensure_topology_current()
                get_logger().info(
                    f"Initializing Dynamo NCCL broadcast for {self._inference_world_size} inference ranks"
                )
                if self._admin_contract == "engine_routes":
                    await self._engine_post(
                        "/engine/init_weights_update_group",
                        {
                            "engine_rpc": "init_broadcaster",
                            "host": host,
                            "port": port,
                            "rank_offset": 0,
                            "inference_world_size": self._inference_world_size,
                            "timeout": timeout,
                            "session_id": "default",
                        },
                        timeout,
                    )
                else:
                    await self._collective_rpc(
                        self.clients[0],
                        method="init_broadcaster",
                        timeout=timeout,
                        args=[
                            host,
                            port,
                            0,
                            self._inference_world_size,
                            timeout,
                            "default",
                        ],
                    )
            except asyncio.CancelledError:
                self._terminalize_nccl()
                raise
            except BaseException as error:
                self._terminalize_nccl()
                raise RuntimeError(
                    "Dynamo NCCL initialization failed; inference workers must restart before retrying"
                ) from error

            self._nccl_initialization_state = "ready"

    async def aclose(self) -> None:
        unique_clients = {id(client): client for client in [*self._frontend_clients, *self.clients]}
        await asyncio.gather(*(client.aclose() for client in unique_clients.values()))
