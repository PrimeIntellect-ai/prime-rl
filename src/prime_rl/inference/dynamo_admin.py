"""Dynamo worker discovery and engine administration."""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import urlsplit, urlunsplit

import httpx
from httpx import AsyncClient
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.async_utils import gather_shielded
from prime_rl.utils.logger import get_logger

NCCL_READY_MARKER = "NCCL_READY"
ADMIN_TIMEOUT_S = 300.0
UPDATE_WEIGHTS_TIMEOUT_S = 720.0
DISCOVERY_REQUEST_TIMEOUT_S = 10.0
DISCOVERY_POLL_INTERVAL_S = 1.0
_RETRYABLE_DISCOVERY_HTTP_STATUS_CODES = frozenset({408, 409, 429})
_REQUIRED_ROUTES = frozenset(
    {
        "init_weights_update_group",
        "pause_generation",
        "resume_generation",
        "update_weights_from_disk",
        "update_weights_from_distributed",
    }
)

WorkerRole: TypeAlias = Literal["agg", "prefill", "decode"]


class _DiscoveryConvergenceError(RuntimeError):
    """A structurally valid discovery state that may converge during startup."""


@dataclass(frozen=True, slots=True)
class DynamoWorker:
    """Restart-safe identity and admin capabilities for one Dynamo worker."""

    instance_id: int
    component: str
    role: WorkerRole
    system_url: str
    model: str
    routes: frozenset[str]


@dataclass(frozen=True, slots=True)
class DynamoTopology:
    """The exact worker shape Prime expects to administer."""

    roles: tuple[WorkerRole, ...]
    gpus_per_worker: int

    def __post_init__(self) -> None:
        if not self.roles:
            raise ValueError("Dynamo topology must contain at least one worker")
        invalid_roles = set(self.roles) - {"agg", "prefill", "decode"}
        if invalid_roles:
            raise ValueError(f"Dynamo topology contains invalid roles: {sorted(invalid_roles)}")
        if isinstance(self.gpus_per_worker, bool) or self.gpus_per_worker < 1:
            raise ValueError("Dynamo topology gpus_per_worker must be at least one")

    def validate(self, workers: Sequence[DynamoWorker]) -> None:
        expected = Counter(self.roles)
        observed = Counter(worker.role for worker in workers)
        if observed != expected:
            raise ValueError(
                "Dynamo worker topology does not match the configured roles: "
                f"expected {dict(sorted(expected.items()))}, observed {dict(sorted(observed.items()))}"
            )

    def role_for_component(self, component: str) -> WorkerRole:
        normalized = component.casefold()
        if "prefill" in normalized:
            return "prefill"
        if "decode" in normalized:
            return "decode"
        if normalized in {"backend", "vllmworker", "agg", "aggregate", "aggregated", "vllmaggworker"}:
            expected = set(self.roles)
            if "decode" in expected and "agg" not in expected:
                return "decode"
            if "agg" in expected and "decode" not in expected:
                return "agg"
            raise ValueError(
                f"Dynamo worker component {component!r} is ambiguous for configured roles {sorted(expected)}"
            )
        raise ValueError(f"Dynamo worker component {component!r} has no recognized inference role")


def _root_url(url: str) -> str:
    return url.rstrip("/").removesuffix("/v1")


def discovery_urls(config: ClientConfig) -> list[str]:
    if config.rl_base_url:
        return [_root_url(url) for url in config.rl_base_url]

    configured = os.getenv("DYN_RL_DISCOVERY_URL")
    if configured:
        return [_root_url(url.strip()) for url in configured.split(",") if url.strip()]

    port = int(os.getenv("DYN_RL_PORT", "8001"))
    urls: list[str] = []
    for base_url in config.base_url:
        parsed = urlsplit(_root_url(base_url))
        host = parsed.hostname or "localhost"
        if ":" in host:
            host = f"[{host}]"
        netloc = f"{host}:{port}"
        urls.append(urlunsplit((parsed.scheme or "http", netloc, "", "", "")))
    return urls


def _worker_sort_key(worker: DynamoWorker) -> tuple[str, str, int, str]:
    return (worker.role, worker.component, worker.instance_id, worker.system_url)


def _parse_worker(value: object, model_name: str, topology: DynamoTopology) -> DynamoWorker:
    if not isinstance(value, Mapping):
        raise ValueError("Dynamo worker discovery returned a non-object worker")
    component = value.get("component")
    instance_id = value.get("instance_id")
    system_url = value.get("system_url")
    model = value.get("model")
    routes = value.get("routes")
    if not isinstance(component, str) or not component:
        raise ValueError("Dynamo worker discovery response is missing component")
    if not isinstance(instance_id, int) or isinstance(instance_id, bool) or instance_id < 0:
        raise ValueError(f"Dynamo worker {component!r} has an invalid instance_id")
    error = value.get("error")
    if error is not None:
        if not isinstance(error, str) or not error:
            raise ValueError(f"Dynamo worker {component}[{instance_id}] has an invalid error")
        raise _DiscoveryConvergenceError(f"Dynamo worker {component}[{instance_id}] is unhealthy: {error}")
    if not isinstance(system_url, str) or not system_url:
        raise ValueError(f"Dynamo worker {component}[{instance_id}] is missing system_url")
    if model != model_name:
        raise ValueError(
            f"Dynamo worker {component}[{instance_id}] at {system_url} serves {model!r}, expected {model_name!r}"
        )
    if not isinstance(routes, list) or not all(isinstance(route, str) for route in routes):
        raise ValueError(f"Dynamo worker {component}[{instance_id}] has invalid routes")
    route_set = frozenset(routes)
    missing = _REQUIRED_ROUTES - route_set
    if missing:
        raise ValueError(f"Dynamo worker {system_url} is missing RL routes: {sorted(missing)}")
    return DynamoWorker(
        instance_id=instance_id,
        component=component,
        role=topology.role_for_component(component),
        system_url=_root_url(system_url),
        model=model,
        routes=route_set,
    )


def _parse_snapshot(
    payload: object,
    model_name: str,
    topology: DynamoTopology,
) -> tuple[str, tuple[DynamoWorker, ...]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("workers"), list):
        raise ValueError("Dynamo worker discovery returned an invalid response")
    namespace = payload.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Dynamo worker discovery response is missing namespace")
    workers = tuple(
        sorted(
            (_parse_worker(value, model_name, topology) for value in payload["workers"]),
            key=_worker_sort_key,
        )
    )
    identities = {(worker.component, worker.instance_id) for worker in workers}
    if len(identities) != len(workers):
        raise ValueError("Dynamo worker discovery returned duplicate worker identities")
    if len({worker.system_url for worker in workers}) != len(workers):
        raise ValueError("Dynamo worker discovery returned duplicate system URLs")
    return namespace, workers


def _retryable_discovery_error(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code in _RETRYABLE_DISCOVERY_HTTP_STATUS_CODES or status_code >= 500
    return isinstance(
        error,
        (
            _DiscoveryConvergenceError,
            httpx.TimeoutException,
            httpx.TransportError,
            TimeoutError,
        ),
    )


async def discover_workers(
    discovery_clients: list[AsyncClient],
    timeout: float,
    *,
    model_name: str,
    topology: DynamoTopology,
) -> tuple[DynamoWorker, ...]:
    """Wait for all frontends to report one identical, complete worker set.

    Incomplete membership, inconsistent valid snapshots, and worker probe errors
    are startup convergence states. A model mismatch or missing RL route is an
    incompatible deployment contract and fails immediately, as do malformed
    payloads and permanent HTTP errors.
    """
    if not discovery_clients:
        raise ValueError("Dynamo worker discovery requires at least one frontend")
    if timeout <= 0:
        raise TimeoutError("Dynamo worker discovery deadline has already expired")
    logger = get_logger()
    last_error: Exception | None = None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while (remaining := deadline - loop.time()) > 0:
        try:
            request_timeout = min(remaining, DISCOVERY_REQUEST_TIMEOUT_S)
            async with asyncio.timeout(remaining):
                results = await asyncio.gather(
                    *(
                        client.get("/v1/rl/workers", timeout=httpx.Timeout(request_timeout))
                        for client in discovery_clients
                    )
                )
            snapshots: list[tuple[str, tuple[DynamoWorker, ...]]] = []
            for response in results:
                response.raise_for_status()
                snapshots.append(_parse_snapshot(response.json(), model_name, topology))
            first = snapshots[0]
            if any(snapshot != first for snapshot in snapshots[1:]):
                raise _DiscoveryConvergenceError("Dynamo discovery frontends returned inconsistent worker snapshots")
            workers = first[1]
            try:
                topology.validate(workers)
            except ValueError as exc:
                raise _DiscoveryConvergenceError(str(exc)) from exc
            logger.info(f"Discovered {len(workers)} Dynamo inference worker(s)")
            return workers
        except Exception as exc:
            if not _retryable_discovery_error(exc):
                raise
            last_error = exc
        remaining = deadline - loop.time()
        if remaining > 0:
            await asyncio.sleep(min(DISCOVERY_POLL_INTERVAL_S, remaining))

    raise TimeoutError(f"Dynamo workers were not ready after {timeout} seconds: {last_error!r}")


def validate_worker_membership(
    expected: Sequence[DynamoWorker],
    discovered: Sequence[DynamoWorker],
) -> None:
    expected_workers = tuple(sorted(expected, key=_worker_sort_key))
    discovered_workers = tuple(sorted(discovered, key=_worker_sort_key))
    if discovered_workers != expected_workers:
        raise RuntimeError(
            "Dynamo worker membership changed after initialization: "
            f"expected {expected_workers!r}, discovered {discovered_workers!r}"
        )


class DynamoAdminAPI:
    """Typed adapter for Dynamo's per-worker ``/engine`` endpoints."""

    def __init__(self) -> None:
        self._distributed_updates = False
        self._distributed_initialization_indeterminate = False
        self._weight_update_indeterminate = False

    def _require_unambiguous_admin_state(self) -> None:
        if self._distributed_initialization_indeterminate:
            raise RuntimeError(
                "Dynamo distributed weight-group initialization is indeterminate after a prior failure or "
                "cancellation; refusing further admin mutation"
            )
        if self._weight_update_indeterminate:
            raise RuntimeError(
                "Dynamo worker weight state is indeterminate after a prior update or resume failure; refusing "
                "further admin mutation"
            )

    @staticmethod
    def _retryable(exception: BaseException) -> bool:
        if isinstance(exception, httpx.HTTPStatusError):
            return exception.response.status_code >= 500
        return isinstance(exception, (httpx.TimeoutException, httpx.TransportError))

    async def _post(
        self,
        client: AsyncClient,
        method: str,
        body: dict | None = None,
        *,
        timeout_s: float = ADMIN_TIMEOUT_S,
        retry_transient: bool = False,
    ) -> dict:
        async def post_once() -> dict:
            response = await client.post(
                f"/engine/{method}",
                json=body or {},
                timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=10.0),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"Dynamo /engine/{method} returned a non-object response")
            if payload.get("status") != "ok":
                raise RuntimeError(payload.get("message", f"Dynamo /engine/{method} failed"))
            return payload

        if not retry_transient:
            return await post_once()

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(self._retryable),
            stop=stop_after_delay(2 * timeout_s) | stop_after_attempt(10),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        ):
            with attempt:
                return await post_once()
        raise AssertionError("unreachable")

    @staticmethod
    async def _settle_fanout(awaitables: Iterable[Awaitable[dict]], operation: str) -> None:
        """Await every sibling even when one fails or the caller is cancelled."""
        tasks = [asyncio.create_task(awaitable) for awaitable in awaitables]
        if not tasks:
            return
        results, cancellation = await gather_shielded(*tasks)

        failures = [result for result in results if isinstance(result, BaseException)]
        primary: BaseException | None = cancellation or (failures[0] if failures else None)
        if primary is None:
            return
        siblings = failures if cancellation is not None else failures[1:]
        for sibling in siblings:
            primary.add_note(f"Dynamo {operation} sibling also failed: {sibling!r}")
        raise primary

    async def initialize_nccl(
        self,
        clients: list[AsyncClient],
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int | None,
        gpus_per_worker: int,
        quantize_in_weight_transfer: bool,
    ) -> None:
        self._require_unambiguous_admin_state()
        if not clients:
            raise ValueError("Cannot initialize NCCL without Dynamo workers")
        if isinstance(gpus_per_worker, bool) or gpus_per_worker < 1:
            raise ValueError("gpus_per_worker must be at least one")
        expected_world_size = len(clients) * gpus_per_worker
        world_size = expected_world_size if inference_world_size is None else inference_world_size
        if world_size != expected_world_size:
            raise ValueError(
                f"inference_world_size={world_size} does not match {len(clients)} Dynamo workers "
                f"with {gpus_per_worker} GPUs each ({expected_world_size})"
            )
        try:
            await self._settle_fanout(
                (
                    self._post(
                        client,
                        "init_weights_update_group",
                        {
                            "host": host,
                            "port": port,
                            "rank_offset": index * gpus_per_worker,
                            "inference_world_size": world_size,
                            "timeout": timeout,
                            "quantize_in_weight_transfer": quantize_in_weight_transfer,
                            "engine_rpc": "init_broadcaster",
                        },
                    )
                    for index, client in enumerate(clients)
                ),
                "init_weights_update_group",
            )
        except BaseException:
            # A lost response or one failed sibling cannot prove whether every
            # rank committed the collective group. This object must not choose
            # the filesystem path or retry into that ambiguous state.
            self._distributed_initialization_indeterminate = True
            raise
        self._distributed_updates = True

    async def update_weights(self, clients: list[AsyncClient], weight_dir: Path | None, step: int) -> None:
        self._require_unambiguous_admin_state()
        primary_error: BaseException | None = None
        try:
            await self._settle_fanout(
                (
                    self._post(
                        client,
                        "pause_generation",
                        {"mode": "wait", "clear_cache": False},
                        retry_transient=True,
                    )
                    for client in clients
                ),
                "pause_generation",
            )
            if weight_dir is not None:
                marker = weight_dir / NCCL_READY_MARKER
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()

            if self._distributed_updates:
                body = {
                    "weight_version": str(step),
                    "weight_dir": weight_dir.as_posix() if weight_dir is not None else None,
                    "engine_rpc": "update_weights_from_path",
                }
                method = "update_weights_from_distributed"
            else:
                if weight_dir is None:
                    raise ValueError("Dynamo filesystem weight updates require weight_dir")
                body = {
                    "model_path": str(weight_dir.resolve()),
                    "weight_version": str(step),
                    "engine_rpc": "update_weights_from_path",
                }
                method = "update_weights_from_disk"

            await self._settle_fanout(
                (self._post(client, method, body, timeout_s=UPDATE_WEIGHTS_TIMEOUT_S) for client in clients),
                method,
            )
        except BaseException as exc:
            primary_error = exc
            self._weight_update_indeterminate = True
            raise
        finally:
            try:
                await self._settle_fanout(
                    (self._post(client, "resume_generation", retry_transient=True) for client in clients),
                    "resume_generation",
                )
            except BaseException as exc:
                self._weight_update_indeterminate = True
                if primary_error is None:
                    raise
                primary_error.add_note(f"Dynamo resume_generation cleanup also failed: {exc!r}")
