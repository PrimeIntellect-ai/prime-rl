from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from prime_rl.configs.shared import ClientConfig, PolicyClientConfig
from prime_rl.orchestrator.clients import (
    ADMIN_TIMEOUT_S,
    ENGINE_ADMIN_LIMITS,
    UPDATE_WEIGHTS_TIMEOUT_S,
    AdminPlane,
    _admin_post,
    _pause_engines,
    _resume_engines,
    check_health,
    maybe_check_has_model,
    setup_admin_clients,
)
from prime_rl.utils.logger import get_logger


class DynamoWorker(BaseModel):
    admin_base_url: str
    instance_id: int = Field(ge=0, strict=True)
    world_size: int = Field(ge=1, strict=True)


class DynamoSnapshot(BaseModel):
    protocol_version: Literal[1]
    workers: tuple[dict[str, object], ...]


class DynamoDiscoveryPending(RuntimeError):
    """The discovery endpoint is healthy but has not published a complete worker set."""


def parse_dynamo_workers(
    payload: object,
    model_name: str,
    *,
    expected_admin_host: str | None = None,
) -> list[DynamoWorker]:
    """The workers serving ``model_name``, ordered by instance id."""
    snapshot = DynamoSnapshot.model_validate(payload)
    matching_workers: list[DynamoWorker] = []
    for raw_worker in snapshot.workers:
        if raw_worker.get("model") != model_name:
            continue
        if raw_worker.get("error") is not None:
            raise DynamoDiscoveryPending("Dynamo worker is not ready")
        if raw_worker.get("admin_base_url") is None:
            raise DynamoDiscoveryPending("Dynamo worker is missing admin_base_url")
        if type(raw_worker.get("world_size")) is not int or raw_worker["world_size"] < 1:
            raise ValueError("Dynamo worker must report a positive world_size")
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
    if len({worker.instance_id for worker in matching_workers}) != len(matching_workers):
        raise ValueError("Dynamo returned duplicate worker instance ids")
    return sorted(matching_workers, key=lambda worker: worker.instance_id)


def worker_fingerprint(worker: DynamoWorker) -> tuple[int, str]:
    return worker.instance_id, str(httpx.URL(worker.admin_base_url))


def topology_fingerprint(workers: list[DynamoWorker]) -> tuple[tuple[int, str], ...]:
    return tuple(sorted(map(worker_fingerprint, workers)))


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


def resolve_dynamo_discovery_url(client_config: PolicyClientConfig) -> str:
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


async def discover_dynamo_workers(
    discovery_url: str,
    model_name: str,
    *,
    headers: dict[str, str],
    timeout: float,
) -> list[DynamoWorker]:
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
    return parse_dynamo_workers(payload, model_name, expected_admin_host=base_url.host)


@dataclass(frozen=True)
class _ModelExpressSession:
    host: str
    port: int
    timeout: int
    staging_buffer_bytes: int | None
    staging_buffers_count: int
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def init_broadcaster_args(self, worker: DynamoWorker) -> list[object]:
        # A worker names its ranks under its own instance id, so no inference-wide rank layout is needed.
        return [
            self.host,
            self.port,
            0,
            worker.world_size,
            self.timeout,
            f"{self.session_id}-{worker.instance_id}",
            self.staging_buffer_bytes,
            self.staging_buffers_count,
        ]


class DynamoAdminPlane(AdminPlane):
    """Admin plane over the workers a trusted Dynamo discovery endpoint reports.

    Readiness binds the worker set of two identical discovery snapshots. NCCL and filesystem updates require that set
    to stay unchanged. ModelExpress updates rediscover the workers before every update: workers that joined are
    initialized and receive the update, workers that left are dropped."""

    def __init__(
        self,
        client_config: PolicyClientConfig,
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
        # The bound workers and their admin clients, in the same order.
        self._workers: list[DynamoWorker] = []
        self.clients: list[httpx.AsyncClient] = []
        self._nccl_initialization_state: Literal["uninitialized", "initializing", "ready", "terminal"] = "uninitialized"
        self._modelexpress: _ModelExpressSession | None = None
        self._modelexpress_workers: set[tuple[int, str]] = set()
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

    async def _discover(self) -> list[DynamoWorker]:
        return await discover_dynamo_workers(
            self._discovery_url,
            self._model_name,
            headers=self._headers,
            timeout=min(30.0, max(1.0, float(self._timeout))),
        )

    async def _discover_until(self, deadline: float) -> list[DynamoWorker]:
        """Poll discovery until it reports a complete worker set."""
        last_error: Exception | None = None
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                async with asyncio.timeout(remaining):
                    return await self._discover()
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    raise
                last_error = error
            except (DynamoDiscoveryPending, httpx.TransportError, TimeoutError) as error:
                last_error = error
            remaining = deadline - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(min(self._poll_interval, remaining))
        raise TimeoutError("Dynamo workers did not become ready before the discovery timeout") from last_error

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

        # Bind only a worker set that two consecutive snapshots agree on.
        workers = await self._discover_until(deadline)
        while True:
            await asyncio.sleep(min(self._poll_interval, self._remaining(deadline)))
            confirmed = await self._discover_until(deadline)
            if topology_fingerprint(confirmed) == topology_fingerprint(workers):
                break
            workers = confirmed
        await self._bind(workers, deadline)

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
            headers=self._headers,
            limits=ENGINE_ADMIN_LIMITS,
            timeout=httpx.Timeout(worker_timeout),
            trust_env=False,
        )

    async def _bind(self, workers: list[DynamoWorker], deadline: float) -> None:
        """Point the admin clients at ``workers``, keeping the clients of workers that stay bound."""
        bound = {worker_fingerprint(worker): client for worker, client in zip(self._workers, self.clients, strict=True)}
        joined = {
            worker_fingerprint(worker): self._make_worker_client(worker)
            for worker in workers
            if worker_fingerprint(worker) not in bound
        }
        if joined:
            try:
                remaining = self._remaining(deadline)
                async with asyncio.timeout(remaining):
                    await check_health(list(joined.values()), timeout=remaining, quiet=True)
            except BaseException:
                await asyncio.gather(*(client.aclose() for client in joined.values()))
                raise
        fingerprints = [worker_fingerprint(worker) for worker in workers]
        left = [fingerprint for fingerprint in bound if fingerprint not in fingerprints]
        for fingerprint in left:
            self._modelexpress_workers.discard(fingerprint)
            await bound[fingerprint].aclose()
        if bound and (joined or left):
            get_logger().info(f"Dynamo workers changed: joined {sorted(joined)}, left {left}")
        self._workers = list(workers)
        self.clients[:] = [
            bound[fingerprint] if fingerprint in bound else joined[fingerprint] for fingerprint in fingerprints
        ]

    async def ensure_topology_current(self) -> None:
        if self._nccl_initialization_state == "terminal":
            raise RuntimeError("Dynamo administration is in a terminal state; restart is required")
        if not self._workers:
            raise RuntimeError("Dynamo topology has not been pinned")
        pinned_fingerprint = topology_fingerprint(self._workers)
        previous_changed_fingerprint: tuple[tuple[int, str], ...] | None = None
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
                if fingerprint == pinned_fingerprint:
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
        method: Literal["init_broadcaster", "update_weights_from_path", "update_weights_from_modelexpress"],
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
        # One result per rank the worker reached.
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or not results or any(result is not None for result in results):
            raise ValueError("Dynamo worker returned an invalid collective RPC response")

    async def update_weights(
        self,
        weight_dir: Path | None,
        *,
        transport: Literal["filesystem", "nccl"],
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
    ) -> None:
        async with self._mutation_lock:
            self._require_uninitialized_nccl()
            if inference_world_size != 1 or any(worker.world_size != 1 for worker in self._workers):
                raise ValueError("Dynamo NCCL currently supports exactly one inference rank")
            if len(self._workers) != 1:
                raise ValueError("Dynamo NCCL currently supports exactly one inference worker")
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

    async def initialize_modelexpress(
        self,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        staging_buffer_bytes: int | None,
        staging_buffers_count: int,
    ) -> None:
        # Workers join and leave between updates and each reports its own world size.
        del inference_world_size
        async with self._mutation_lock:
            self._modelexpress = _ModelExpressSession(
                host=host,
                port=port,
                timeout=timeout,
                staging_buffer_bytes=staging_buffer_bytes,
                staging_buffers_count=staging_buffers_count,
            )
            await self._sync_modelexpress_workers()

    async def _sync_modelexpress_workers(self) -> None:
        """Bind the workers Dynamo reports now and initialize ModelExpress on the ones that joined."""
        if self._modelexpress is None:
            raise RuntimeError("Dynamo ModelExpress updates require initialize_modelexpress first")
        session = self._modelexpress
        deadline = time.monotonic() + self._timeout
        await self._bind(await self._discover_until(deadline), deadline)

        async def initialize(worker: DynamoWorker, client: httpx.AsyncClient) -> None:
            await self._collective_rpc(
                client,
                method="init_broadcaster",
                timeout=max(ADMIN_TIMEOUT_S, session.timeout),
                args=session.init_broadcaster_args(worker),
            )
            self._modelexpress_workers.add(worker_fingerprint(worker))

        joined = [
            (worker, client)
            for worker, client in zip(self._workers, self.clients, strict=True)
            if worker_fingerprint(worker) not in self._modelexpress_workers
        ]
        results = await asyncio.gather(*(initialize(*pair) for pair in joined), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def update_modelexpress_weights(
        self,
        *,
        version_uid: str,
        step: int = 0,
        on_paused: Callable[[], None] | None = None,
    ) -> None:
        """Install an exact MX version on every worker Dynamo reports; keep them paused if installation is uncertain.

        A worker that joins after this discovery serves its launch weights until the next update."""
        if not version_uid:
            raise ValueError("modelexpress requires version_uid")
        async with self._mutation_lock:
            await self._sync_modelexpress_workers()
            clients = list(self.clients)
            pause_times = await _pause_engines(clients, step=step)
            if on_paused is not None:
                on_paused()

            async def install(client: httpx.AsyncClient) -> float:
                start = time.perf_counter()
                await self._collective_rpc(
                    client,
                    method="update_weights_from_modelexpress",
                    timeout=UPDATE_WEIGHTS_TIMEOUT_S,
                    args=[version_uid],
                )
                return time.perf_counter() - start

            results = await asyncio.gather(*map(install, clients), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            if self.after_transfer is not None:
                await self.after_transfer()
            resume_times = await _resume_engines(clients)
            self.update_times = {"pause": pause_times, "transfer": results, "resume": resume_times}
        get_logger().info(f"Installed policy v{step} on {len(clients)} Dynamo worker(s) through ModelExpress")

    async def aclose(self) -> None:
        unique_clients = {id(client): client for client in [*self._frontend_clients, *self.clients]}
        await asyncio.gather(*(client.aclose() for client in unique_clients.values()))
