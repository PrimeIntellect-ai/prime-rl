from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.clients import (
    ADMIN_TIMEOUT_S,
    UPDATE_WEIGHTS_TIMEOUT_S,
    AdminPlane,
    _admin_post,
    check_health,
    maybe_check_has_model,
    setup_admin_clients,
)
from prime_rl.utils.logger import get_logger

DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION = 1
MAX_DISCOVERY_BODY_BYTES = 1024 * 1024
MAX_ADMIN_RESPONSE_BYTES = 1024 * 1024
MAX_DISCOVERY_WORKERS = 128


class DynamoWorker(BaseModel):
    admin_base_url: str
    instance_id: int = Field(ge=0, strict=True)


class DynamoSnapshot(BaseModel):
    protocol_version: int = Field(
        strict=True,
        ge=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
        le=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
    )
    workers: tuple[dict[str, Any], ...] = Field(max_length=MAX_DISCOVERY_WORKERS)


class DynamoDiscoveryPending(RuntimeError):
    """The discovery endpoint is healthy but has not published a complete worker set."""


async def _bounded_json_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    max_response_bytes: int = MAX_ADMIN_RESPONSE_BYTES,
    **kwargs,
) -> object:
    request_headers = kwargs.pop("headers", {})
    request_headers = {name: value for name, value in request_headers.items() if name.lower() != "accept-encoding"}
    request_headers["Accept-Encoding"] = "identity"
    async with client.stream(method, path, headers=request_headers, **kwargs) as response:
        response.raise_for_status()
        content_encoding = response.headers.get("Content-Encoding")
        if content_encoding and content_encoding.strip().lower() != "identity":
            raise ValueError(f"Dynamo endpoint returned unsupported Content-Encoding {content_encoding!r}")
        body = bytearray()
        async for chunk in response.aiter_raw():
            if len(body) + len(chunk) > max_response_bytes:
                raise ValueError(f"Dynamo response body exceeds {max_response_bytes} bytes")
            body.extend(chunk)
    return json.loads(body)


def parse_dynamo_workers(
    payload: object,
    model_name: str,
    *,
    expected_admin_host: str | None = None,
) -> tuple[DynamoWorker, ...]:
    snapshot = DynamoSnapshot.model_validate(payload)
    matching_workers: list[DynamoWorker] = []
    for raw_worker in snapshot.workers:
        if raw_worker.get("model") != model_name:
            continue
        if raw_worker.get("error") is not None:
            raise DynamoDiscoveryPending("Dynamo worker is not ready")
        missing_metadata = [name for name in ("admin_base_url", "world_size") if raw_worker.get(name) is None]
        if missing_metadata:
            raise DynamoDiscoveryPending(
                f"Dynamo worker is missing required RL metadata: {', '.join(missing_metadata)}"
            )
        if type(raw_worker["world_size"]) is not int or raw_worker["world_size"] != 1:
            raise ValueError("Dynamo RL currently supports exactly one inference rank")
        worker = DynamoWorker.model_validate(raw_worker)
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
        if expected_admin_host is not None and admin_url.host != expected_admin_host:
            raise ValueError(
                f"Dynamo worker admin host {admin_url.host!r} does not match discovery host {expected_admin_host!r}"
            )
        matching_workers.append(worker)

    if not matching_workers:
        raise DynamoDiscoveryPending(f"Dynamo returned no bound workers for model {model_name!r}")
    if len(matching_workers) != 1:
        raise ValueError("Dynamo RL currently supports exactly one inference worker")
    return tuple(matching_workers)


def topology_fingerprint(workers: tuple[DynamoWorker, ...]) -> tuple[tuple[int, str], ...]:
    return tuple((worker.instance_id, worker.admin_base_url) for worker in workers)


def _discovery_headers(client_config: ClientConfig) -> dict[str, str]:
    env_headers = {
        name: value
        for name, env_name in client_config.headers_from_env.items()
        if (value := os.getenv(env_name)) is not None
    }
    headers = {
        name: value
        for name, value in {**client_config.headers, **env_headers}.items()
        if name.lower() != "accept-encoding"
    }
    api_key = os.getenv(client_config.api_key_var)
    if api_key:
        headers = {name: value for name, value in headers.items() if name.lower() != "authorization"}
        headers["Authorization"] = f"Bearer {api_key}"
    headers["Accept-Encoding"] = "identity"
    return headers


async def discover_dynamo_workers(
    discovery_url: str,
    model_name: str,
    *,
    headers: dict[str, str],
    timeout: float,
) -> tuple[DynamoWorker, ...]:
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
    request_headers = {name: value for name, value in headers.items() if name.lower() != "accept-encoding"}
    request_headers["Accept-Encoding"] = "identity"
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(headers=request_headers, timeout=timeout, trust_env=False) as client:
                payload = await _bounded_json_request(
                    client,
                    "GET",
                    url,
                    max_response_bytes=MAX_DISCOVERY_BODY_BYTES,
                )
    except TimeoutError as error:
        raise TimeoutError(f"Dynamo discovery request exceeded {timeout} seconds") from error
    return parse_dynamo_workers(payload, model_name, expected_admin_host=base_url.host)


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
        self._fingerprint: tuple[tuple[object, ...], ...] | None = None
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

        previous_fingerprint: tuple[tuple[object, ...], ...] | None = None
        last_error: Exception | None = None
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                async with asyncio.timeout(remaining):
                    workers = await self._discover()
                fingerprint = topology_fingerprint(workers)
                if fingerprint == previous_fingerprint:
                    candidate_clients = self._make_worker_clients(workers)
                    try:
                        remaining = self._remaining(deadline)
                        async with asyncio.timeout(remaining):
                            await check_health(candidate_clients, timeout=remaining, quiet=True)
                    except BaseException:
                        await asyncio.gather(*(client.aclose() for client in candidate_clients))
                        raise
                    self._bind(workers, fingerprint, candidate_clients)
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

    def _make_worker_clients(self, workers: tuple[DynamoWorker, ...]) -> list[httpx.AsyncClient]:
        worker_timeout = min(30.0, max(1.0, float(self._timeout)))
        return [
            httpx.AsyncClient(
                base_url=worker.admin_base_url,
                headers={"Accept-Encoding": "identity"},
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
                timeout=httpx.Timeout(worker_timeout),
                trust_env=False,
            )
            for worker in workers
        ]

    def _bind(
        self,
        workers: tuple[DynamoWorker, ...],
        fingerprint: tuple[tuple[object, ...], ...],
        clients: list[httpx.AsyncClient] | None = None,
    ) -> None:
        self._fingerprint = fingerprint
        self.clients = clients if clients is not None else self._make_worker_clients(workers)

    async def ensure_topology_current(self) -> None:
        if self._nccl_initialization_state == "terminal":
            raise RuntimeError("Dynamo administration is in a terminal state; restart is required")
        if self._fingerprint is None:
            raise RuntimeError("Dynamo topology has not been pinned")
        previous_changed_fingerprint: tuple[tuple[object, ...], ...] | None = None
        last_error: Exception | None = None
        deadline = time.monotonic() + self._timeout
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
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
            payload = await _bounded_json_request(
                client,
                "POST",
                "/collective_rpc",
                timeout=httpx.Timeout(connect=10.0, read=operation_timeout, write=10.0, pool=10.0),
                json={"method": method, "timeout": operation_timeout, "args": args, "kwargs": {}},
            )
        if payload != {"results": [None]}:
            raise ValueError("Dynamo worker returned an invalid collective RPC response")

    async def update_weights(
        self,
        weight_dir: Path | None,
        *,
        transport: Literal["filesystem", "nccl", "nixl"],
        step: int = 0,
        on_paused: Callable[[], None] | None = None,
    ) -> None:
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
                await _admin_post(self.clients[0], "/pause", params={"mode": "keep", "clear_cache": "false"})
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
                await _admin_post(self.clients[0], "/resume", timeout_s=ADMIN_TIMEOUT_S)
            except BaseException as failure:
                self._terminalize_nccl()
                raise RuntimeError("Dynamo resume failed; worker state is unknown and restart is required") from failure
            self._nccl_initialization_state = "ready"
            get_logger().info(f"Applied NCCL weights for policy v{step} to the Dynamo worker")

    async def initialize_nccl(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        quantize_in_weight_transfer: bool = False,
    ) -> None:
        async with self._mutation_lock:
            self._require_uninitialized_nccl()
            if inference_world_size != 1:
                raise ValueError("Dynamo RL currently supports exactly one inference rank")
            self._nccl_initialization_state = "initializing"
            try:
                await self.ensure_topology_current()
                get_logger().info("Initializing Dynamo NCCL broadcast for one inference rank")
                await self._collective_rpc(
                    self.clients[0],
                    method="init_broadcaster",
                    timeout=timeout,
                    args=[
                        host,
                        port,
                        0,
                        1,
                        timeout,
                        quantize_in_weight_transfer,
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
