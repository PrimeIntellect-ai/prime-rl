from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from prime_rl.configs.shared import (
    ClientConfig,
    DynamoConfig,
    is_secure_or_loopback_url,
    normalize_admin_host_allowlist_entry,
)
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
MAX_IDENTITY_LENGTH = 256
MAX_MODEL_LENGTH = 512
MAX_URL_LENGTH = 2048
MAX_ROUTES = 128
MAX_ROUTE_LENGTH = 256
MAX_WORKER_ERROR_BYTES = 2048
BIDI_CONTROL_CHARACTERS = frozenset("\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")

IdentityString = Annotated[str, Field(min_length=1, max_length=MAX_IDENTITY_LENGTH)]
UrlString = Annotated[str, Field(min_length=1, max_length=MAX_URL_LENGTH)]
RouteString = Annotated[str, Field(min_length=1, max_length=MAX_ROUTE_LENGTH)]


class DynamoWorker(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: IdentityString
    component: IdentityString
    endpoint: IdentityString
    instance_id: int = Field(ge=0, strict=True)
    transport: str | dict[str, str]
    request_plane_url: UrlString
    system_url: UrlString | None = None
    admin_base_url: UrlString
    world_size: int = Field(gt=0, strict=True)
    model: str = Field(min_length=1, max_length=MAX_MODEL_LENGTH)
    routes: tuple[RouteString, ...] = Field(max_length=MAX_ROUTES)
    error: str | None = Field(None, max_length=MAX_URL_LENGTH)


class DynamoSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_version: int = Field(
        strict=True,
        ge=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
        le=DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION,
    )
    namespace: IdentityString
    workers: tuple[dict[str, Any], ...] = Field(max_length=MAX_DISCOVERY_WORKERS)


class DynamoDiscoveryPending(RuntimeError):
    """The discovery endpoint is healthy but has not published a complete worker set."""


async def _bounded_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    *,
    max_response_bytes: int = MAX_ADMIN_RESPONSE_BYTES,
    **kwargs,
) -> httpx.Response:
    """Send a Dynamo admin request without buffering an unbounded response."""
    request_headers = kwargs.pop("headers", {})
    request_headers = {name: value for name, value in request_headers.items() if name.lower() != "accept-encoding"}
    request_headers["Accept-Encoding"] = "identity"
    async with client.stream(method, path, headers=request_headers, **kwargs) as response:
        content_encoding = response.headers.get("Content-Encoding")
        if content_encoding and content_encoding.strip().lower() != "identity":
            raise ValueError(f"Dynamo admin endpoint returned unsupported Content-Encoding {content_encoding!r}")
        body = bytearray()
        async for chunk in response.aiter_raw():
            if len(body) + len(chunk) > max_response_bytes:
                raise ValueError(f"Dynamo admin response body exceeds {max_response_bytes} bytes")
            body.extend(chunk)
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=bytes(body),
            request=response.request,
        )


def validate_dynamo_config(config: DynamoConfig) -> None:
    has_discovery_credentials = bool(config.api_key_var or config.headers_from_env)
    if has_discovery_credentials and not is_secure_or_loopback_url(config.discovery_url):
        raise ValueError("dynamo.discovery_url must use HTTPS when discovery credentials are configured")
    has_admin_credentials = bool(config.admin_api_key_var or config.admin_headers_from_env)
    if has_admin_credentials and config.admin_origin_allowlist is None:
        raise ValueError("dynamo.admin_origin_allowlist is required when admin credentials are configured")
    if has_admin_credentials and any(
        not is_secure_or_loopback_url(origin) for origin in config.admin_origin_allowlist or ()
    ):
        raise ValueError(
            "dynamo.admin_origin_allowlist must use HTTPS or loopback when admin credentials are configured"
        )


def _admin_host_allowed(hostname: str, allowlist: tuple[str, ...]) -> bool:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return normalize_admin_host_allowlist_entry(hostname) in allowlist

    for entry in allowlist:
        try:
            if address in ipaddress.ip_network(entry):
                return True
        except ValueError:
            continue
    return False


def _worker_sort_key(worker: DynamoWorker) -> tuple[object, ...]:
    return worker.namespace, worker.component, worker.endpoint, worker.instance_id


def _validate_worker_error(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Dynamo worker error must be a string")
    if len(value.encode("utf-8")) > MAX_WORKER_ERROR_BYTES:
        raise ValueError(f"Dynamo worker error must not exceed {MAX_WORKER_ERROR_BYTES} bytes")
    if any(
        unicodedata.category(character) in {"Cc", "Zl", "Zp"} or character in BIDI_CONTROL_CHARACTERS
        for character in value
    ):
        raise ValueError("Dynamo worker error must not contain control or directional characters")
    return value


def parse_dynamo_workers(
    payload: object,
    model_name: str,
    *,
    expected_namespace: str,
    admin_host_allowlist: tuple[str, ...],
    admin_origin_allowlist: tuple[str, ...] | None = None,
) -> tuple[DynamoWorker, ...]:
    snapshot = DynamoSnapshot.model_validate(payload)
    if snapshot.namespace != expected_namespace:
        raise ValueError(
            f"Dynamo snapshot namespace {snapshot.namespace!r} does not match expected namespace {expected_namespace!r}"
        )
    normalized_host_allowlist = tuple(
        dict.fromkeys(normalize_admin_host_allowlist_entry(entry) for entry in admin_host_allowlist)
    )
    allowed_origins: set[tuple[str, str, int]] | None = None
    if admin_origin_allowlist is not None:
        allowed_origins = set()
        for origin in admin_origin_allowlist:
            try:
                origin_url = httpx.URL(origin)
            except httpx.InvalidURL as error:
                raise ValueError("admin_origin_allowlist must contain valid URLs") from error
            if (
                origin_url.scheme not in {"http", "https"}
                or not origin_url.host
                or (origin_url.port is not None and not 1 <= origin_url.port <= 65535)
                or origin_url.userinfo
                or origin_url.query
                or origin_url.fragment
                or origin_url.path != "/"
            ):
                raise ValueError("admin_origin_allowlist must contain http(s) origins")
            allowed_origins.add(
                (
                    origin_url.scheme,
                    normalize_admin_host_allowlist_entry(origin_url.host),
                    origin_url.port or (80 if origin_url.scheme == "http" else 443),
                )
            )
    matching_workers: list[DynamoWorker] = []
    matching_origins: list[tuple[str, str, int]] = []
    for raw_worker in snapshot.workers:
        if raw_worker.get("model") != model_name:
            continue
        raw_error = raw_worker.get("error")
        if raw_error is not None and (error := _validate_worker_error(raw_error)):
            raise DynamoDiscoveryPending(f"Dynamo worker is not ready: {error}")
        missing_metadata = [name for name in ("admin_base_url", "world_size") if raw_worker.get(name) is None]
        if missing_metadata:
            raise DynamoDiscoveryPending(
                f"Dynamo worker is missing required RL metadata: {', '.join(missing_metadata)}"
            )
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
        if worker.namespace != expected_namespace:
            raise ValueError(
                f"Dynamo worker namespace {worker.namespace!r} does not match expected namespace {expected_namespace!r}"
            )
        if not _admin_host_allowed(admin_url.host, normalized_host_allowlist):
            raise ValueError(f"Dynamo worker admin host {admin_url.host!r} is not in the configured allowlist")
        admin_origin = (
            admin_url.scheme,
            normalize_admin_host_allowlist_entry(admin_url.host),
            admin_url.port or (80 if admin_url.scheme == "http" else 443),
        )
        if allowed_origins is not None and admin_origin not in allowed_origins:
            raise ValueError(f"Dynamo worker admin origin {admin_origin!r} is not in the configured origin allowlist")
        matching_workers.append(worker)
        matching_origins.append(admin_origin)

    if not matching_workers:
        raise DynamoDiscoveryPending(f"Dynamo returned no bound workers for model {model_name!r}")

    identities = [
        (worker.namespace, worker.component, worker.endpoint, worker.instance_id) for worker in matching_workers
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Dynamo returned duplicate worker identities")
    if len(set(matching_origins)) != len(matching_origins):
        raise ValueError("Dynamo returned duplicate worker admin endpoints")

    return tuple(
        sorted(
            matching_workers,
            key=_worker_sort_key,
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
    dynamo = client_config.dynamo
    if dynamo is None:
        raise ValueError("Dynamo discovery configuration is required")
    env_headers = {
        name: value for name, env_name in dynamo.headers_from_env.items() if (value := os.getenv(env_name)) is not None
    }
    headers = {name: value for name, value in env_headers.items() if name.lower() != "accept-encoding"}
    api_key = os.getenv(dynamo.api_key_var) if dynamo.api_key_var is not None else None
    if api_key:
        headers = {name: value for name, value in headers.items() if name.lower() != "authorization"}
        headers["Authorization"] = f"Bearer {api_key}"
    headers["Accept-Encoding"] = "identity"
    return headers


def _admin_headers(client_config: ClientConfig) -> dict[str, str]:
    dynamo = client_config.dynamo
    if dynamo is None:
        raise ValueError("Dynamo discovery configuration is required")
    env_headers = {
        name: value
        for name, env_name in dynamo.admin_headers_from_env.items()
        if (value := os.getenv(env_name)) is not None
    }
    headers = env_headers
    api_key = os.getenv(dynamo.admin_api_key_var) if dynamo.admin_api_key_var is not None else None
    if api_key:
        headers = {name: value for name, value in headers.items() if name.lower() != "authorization"}
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def discover_dynamo_workers(
    discovery_url: str,
    model_name: str,
    *,
    headers: dict[str, str],
    timeout: float,
    expected_namespace: str,
    admin_host_allowlist: tuple[str, ...],
    admin_origin_allowlist: tuple[str, ...] | None = None,
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
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_encoding = response.headers.get("Content-Encoding")
                    if content_encoding and content_encoding.strip().lower() != "identity":
                        raise ValueError(f"Dynamo discovery returned unsupported Content-Encoding {content_encoding!r}")
                    body = bytearray()
                    async for chunk in response.aiter_raw():
                        if len(body) + len(chunk) > MAX_DISCOVERY_BODY_BYTES:
                            raise ValueError(f"Dynamo discovery response body exceeds {MAX_DISCOVERY_BODY_BYTES} bytes")
                        body.extend(chunk)
    except TimeoutError as error:
        raise TimeoutError(f"Dynamo discovery request exceeded {timeout} seconds") from error
    return parse_dynamo_workers(
        json.loads(body),
        model_name,
        expected_namespace=expected_namespace,
        admin_host_allowlist=admin_host_allowlist,
        admin_origin_allowlist=admin_origin_allowlist,
    )


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
        validate_dynamo_config(client_config.dynamo)
        self._client_config = client_config
        self._model_name = model_name
        self._poll_interval = poll_interval
        self._timeout = client_config.wait_for_ready_timeout
        self._headers = _discovery_headers(client_config)
        self._admin_headers = _admin_headers(client_config)
        self._frontend_clients = setup_admin_clients(
            client_config.model_copy(update={"admin_base_url": None}),
            timeout=max(1.0, min(self._timeout, 30.0)),
            trust_env=False,
        )
        self.clients: list[httpx.AsyncClient] = []
        self.workers: tuple[DynamoWorker, ...] = ()
        self._fingerprint: tuple[tuple[object, ...], ...] | None = None
        self._control_terminal = False
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
        self._control_terminal = True

    async def _discover(self) -> tuple[DynamoWorker, ...]:
        dynamo = self._client_config.dynamo
        assert dynamo is not None
        return await discover_dynamo_workers(
            dynamo.discovery_url,
            self._model_name,
            headers=self._headers,
            timeout=min(30.0, max(1.0, float(self._timeout))),
            expected_namespace=dynamo.expected_namespace,
            admin_host_allowlist=dynamo.admin_host_allowlist,
            admin_origin_allowlist=dynamo.admin_origin_allowlist,
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
                            await self._check_admin_capabilities(
                                candidate_clients,
                                workers,
                                timeout=self._remaining(deadline),
                            )
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
                headers={**self._admin_headers, "Accept-Encoding": "identity"},
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
                timeout=httpx.Timeout(worker_timeout),
                trust_env=False,
            )
            for worker in workers
        ]

    async def _check_admin_capabilities(
        self,
        clients: list[httpx.AsyncClient],
        workers: tuple[DynamoWorker, ...],
        *,
        timeout: float,
    ) -> None:
        results = await asyncio.gather(
            *(
                self._collective_rpc(
                    client,
                    method="liveness_probe",
                    timeout=timeout,
                    args=[],
                    expected_result_count=worker.world_size,
                )
                for client, worker in zip(clients, workers, strict=True)
            ),
            return_exceptions=True,
        )
        if failure := next((result for result in results if isinstance(result, BaseException)), None):
            if isinstance(failure, httpx.HTTPStatusError) and failure.response.status_code >= 500:
                raise failure
            if isinstance(failure, (httpx.TransportError, TimeoutError)):
                raise failure
            raise RuntimeError(
                "A discovered Dynamo worker does not expose the required vLLM development admin contract"
            ) from failure

    def _bind(
        self,
        workers: tuple[DynamoWorker, ...],
        fingerprint: tuple[tuple[object, ...], ...],
        clients: list[httpx.AsyncClient] | None = None,
    ) -> None:
        self.workers = workers
        self._fingerprint = fingerprint
        self.clients = clients if clients is not None else self._make_worker_clients(workers)

    async def ensure_topology_current(self) -> None:
        if self._control_terminal:
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
        method: Literal["init_broadcaster", "liveness_probe", "update_weights_from_path"],
        timeout: int | float,
        args: list[object],
        expected_result_count: int,
    ) -> None:
        operation_timeout = max(1.0, float(timeout))
        async with asyncio.timeout(operation_timeout + 15.0):
            response = await _bounded_request(
                client,
                "POST",
                "/collective_rpc",
                timeout=httpx.Timeout(connect=10.0, read=operation_timeout, write=10.0, pool=10.0),
                json={"method": method, "timeout": operation_timeout, "args": args, "kwargs": {}},
            )
            response.raise_for_status()
        payload = response.json()
        if (
            not isinstance(payload, dict)
            or set(payload) != {"results"}
            or not isinstance(payload["results"], list)
            or not payload["results"]
            or any(result is not None for result in payload["results"])
            or len(payload["results"]) != expected_result_count
        ):
            raise ValueError("Dynamo worker returned an invalid collective RPC response")

    async def update_weights(
        self,
        weight_dir: Path | None,
        *,
        transport: Literal["filesystem", "nccl", "nixl"],
        step: int = 0,
        on_paused: Callable[[], None] | None = None,
    ) -> None:
        if transport not in ("filesystem", "nccl"):
            raise ValueError("The Dynamo admin plane supports only filesystem and NCCL weight updates")
        if weight_dir is None:
            raise ValueError(f"{transport.upper()} weight updates require a broadcast directory")
        if transport == "nccl":
            self._require_ready_nccl()
        async with self._mutation_lock:
            await self.ensure_topology_current()
            self._control_terminal = True
            if transport == "nccl":
                self._nccl_initialization_state = "terminal"

            pause_results = await asyncio.gather(
                *(
                    _admin_post(client, "/pause", params={"mode": "keep", "clear_cache": "false"})
                    for client in self.clients
                ),
                return_exceptions=True,
            )
            if failure := next((result for result in pause_results if isinstance(result, BaseException)), None):
                raise RuntimeError("Dynamo pause failed; worker state is unknown and restart is required") from failure

            if on_paused is not None:
                try:
                    on_paused()
                except BaseException as error:
                    raise RuntimeError(
                        "Dynamo pause callback failed; engines remain paused and restart is required"
                    ) from error
            update_results = await asyncio.gather(
                *(
                    self._collective_rpc(
                        client,
                        method="update_weights_from_path",
                        timeout=UPDATE_WEIGHTS_TIMEOUT_S,
                        args=[weight_dir.as_posix()],
                        expected_result_count=worker.world_size,
                    )
                    for client, worker in zip(self.clients, self.workers, strict=True)
                ),
                return_exceptions=True,
            )
            if failure := next((result for result in update_results if isinstance(result, BaseException)), None):
                if transport == "nccl":
                    self._terminalize_nccl()
                raise RuntimeError(
                    f"Dynamo {transport} update failed; engines remain paused and restart is required"
                ) from failure

            resume_results = await asyncio.gather(
                *(_admin_post(client, "/resume", timeout_s=ADMIN_TIMEOUT_S) for client in self.clients),
                return_exceptions=True,
            )
            if failure := next((result for result in resume_results if isinstance(result, BaseException)), None):
                if transport == "nccl":
                    self._terminalize_nccl()
                raise RuntimeError("Dynamo resume failed; worker state is unknown and restart is required") from failure
            self._control_terminal = False
            if transport == "nccl":
                self._nccl_initialization_state = "ready"
            if transport == "filesystem":
                get_logger().info(
                    f"Applied filesystem weights for policy v{step} across {len(self.clients)} Dynamo worker endpoints"
                )
            else:
                get_logger().info(
                    f"Applied NCCL weights for policy v{step} across {len(self.clients)} Dynamo worker endpoints"
                )

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
            if self._control_terminal:
                raise RuntimeError("Dynamo administration is in a terminal state; restart is required")
            self._require_uninitialized_nccl()
            self._nccl_initialization_state = "initializing"
            try:
                await self.ensure_topology_current()
                rank_offsets = self._rank_offsets(inference_world_size)
                get_logger().info(
                    f"Initializing Dynamo NCCL broadcast: {len(self.clients)} workers, "
                    f"inference_world_size={inference_world_size}, "
                    f"worker_world_sizes={tuple(worker.world_size for worker in self.workers)}"
                )
                results = await asyncio.gather(
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
                            expected_result_count=worker.world_size,
                        )
                        for client, worker, rank_offset in zip(
                            self.clients,
                            self.workers,
                            rank_offsets,
                            strict=True,
                        )
                    ),
                    return_exceptions=True,
                )
            except asyncio.CancelledError:
                self._terminalize_nccl()
                raise
            except BaseException as error:
                self._terminalize_nccl()
                raise RuntimeError(
                    "Dynamo NCCL initialization failed; inference workers must restart before retrying"
                ) from error

            failures = tuple(index for index, result in enumerate(results) if isinstance(result, BaseException))
            successes = tuple(index for index, result in enumerate(results) if not isinstance(result, BaseException))
            if failures:
                first_failure = results[failures[0]]
                assert isinstance(first_failure, BaseException)
                self._terminalize_nccl()
                raise RuntimeError(
                    "Dynamo NCCL initialization could not be reconciled because no supported teardown RPC exists; "
                    f"successful workers={successes}, failed workers={failures}; inference workers must restart"
                ) from first_failure
            self._nccl_initialization_state = "ready"

    async def initialize_nixl(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        session_id: str,
    ) -> None:
        raise ValueError("The standalone Dynamo admin plane does not support NIXL weight updates")

    async def load_lora_adapter(self, lora_name: str, lora_path: Path) -> None:
        raise ValueError("The standalone Dynamo admin plane does not support LoRA weight updates")

    async def aclose(self) -> None:
        unique_clients = {id(client): client for client in [*self._frontend_clients, *self.clients]}
        await asyncio.gather(*(client.aclose() for client in unique_clients.values()))
