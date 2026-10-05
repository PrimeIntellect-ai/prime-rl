from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
from httpx import AsyncClient

from prime_rl.utils.logger import get_logger

STAGE_UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024


class RelayPeerClient:
    def __init__(self, client: AsyncClient, index: int) -> None:
        self.client = client
        self.prefix = f"/weight_peer/{index}"
        self.base_url = httpx.URL(f"{str(client.base_url).rstrip('/')}{self.prefix}")

    async def get(self, route: str, **kwargs: Any) -> httpx.Response:
        return await self.client.get(f"{self.prefix}/{route.lstrip('/')}", **kwargs)

    async def post(self, route: str, **kwargs: Any) -> httpx.Response:
        return await self.client.post(f"{self.prefix}/{route.lstrip('/')}", **kwargs)


WeightAdminClient = AsyncClient | RelayPeerClient


@dataclass(frozen=True)
class EndpointOperationResult:
    endpoint: str
    operation: str
    duration_s: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class EndpointOperationError(RuntimeError):
    def __init__(self, operation: str, results: list[EndpointOperationResult]):
        self.operation = operation
        self.results = results
        failures = [result for result in results if not result.ok]
        message = "; ".join(f"{failure.endpoint}: {failure.error}" for failure in failures)
        super().__init__(f"{operation} failed on {len(failures)}/{len(results)} inference endpoint(s): {message}")


@dataclass
class EndpointLeaseRuntime:
    state: Literal["healthy", "retired", "recovering"] = "healthy"
    retry_after: float | None = None
    retired_count: int = 0
    recovery_attempt_count: int = 0
    recovery_success_count: int = 0
    replayed_delta_count: int = 0
    reload_count: int = 0
    last_reason: str | None = None


@dataclass(frozen=True)
class DeltaReplayEntry:
    weight_path: Path
    version: str
    base_version: str
    upload: bool
    upload_method: Literal["multipart", "chunked", "streaming"]
    done_path: Path | None
    sha256: str


def _endpoint_key(admin_client: WeightAdminClient) -> str:
    return str(admin_client.base_url).rstrip("/").removesuffix("/v1")


def _delta_artifact_path(weight_path: Path, upload_method: str) -> Path:
    if weight_path.is_file():
        return weight_path
    preferred = "delta.stream" if upload_method == "streaming" else "delta.safetensors"
    fallback = "delta.safetensors" if upload_method == "streaming" else "delta.stream"
    path = weight_path / preferred
    return path if path.exists() else weight_path / fallback


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(STAGE_UPLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _format_exception(exception: BaseException) -> str:
    message = str(exception).strip()
    if message:
        return f"{exception.__class__.__name__}: {message}"
    return exception.__class__.__name__


def _is_retryable_stage_error(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return bool(error.exceptions) and all(_is_retryable_stage_error(exception) for exception in error.exceptions)
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {408, 425, 429} or error.response.status_code >= 500
    return isinstance(error, httpx.RequestError)


async def _run_endpoint_operations(
    admin_clients: list[WeightAdminClient],
    operation: str,
    call: Callable[[WeightAdminClient], Awaitable[None]],
) -> list[EndpointOperationResult]:
    async def _run(admin_client: WeightAdminClient) -> EndpointOperationResult:
        start = time.perf_counter()
        try:
            await call(admin_client)
        except Exception as exc:
            return EndpointOperationResult(
                endpoint=_endpoint_key(admin_client),
                operation=operation,
                duration_s=time.perf_counter() - start,
                error=_format_exception(exc),
            )
        return EndpointOperationResult(
            endpoint=_endpoint_key(admin_client),
            operation=operation,
            duration_s=time.perf_counter() - start,
        )

    results = await asyncio.gather(*[_run(admin_client) for admin_client in admin_clients])
    if any(not result.ok for result in results):
        raise EndpointOperationError(operation, results)
    return results


async def _pause_engine(client: WeightAdminClient) -> None:
    response = await client.post("/pause", params={"mode": "keep", "clear_cache": "false"})
    response.raise_for_status()


async def _resume_engine(client: WeightAdminClient) -> None:
    response = await client.post("/resume")
    response.raise_for_status()


async def stage_weights(
    admin_clients: list[WeightAdminClient],
    weight_path: Path,
    version: str,
    mode: str = "full",
    base_version: str | None = None,
    upload: bool = False,
    upload_method: Literal["multipart", "chunked", "streaming"] = "multipart",
    chunk_size_bytes: int = STAGE_UPLOAD_CHUNK_BYTES,
    num_streams: int = 1,
    chunk_retries: int = 3,
    retry_base_delay_s: float = 0.25,
    stage_retries: int = 1,
    stage_retry_base_delay_s: float = 1.0,
    done_path: Path | None = None,
    poll_interval_s: float = 0.1,
    relay: bool = True,
) -> list[EndpointOperationResult]:
    """Stage weights on static inference servers.

    By default this records a path that is already visible to the inference
    server, matching the shared-filesystem delta path used by local migration
    runs. Set ``upload=True`` to stream a single file to the server's staging
    directory. ``upload_method="chunked"`` uses offset-based chunk upload plus
    final size/hash verification.
    """
    if mode not in {"full", "delta"}:
        raise ValueError(f"unsupported weight update mode: {mode}")
    if upload_method not in {"multipart", "chunked", "streaming"}:
        raise ValueError(f"unsupported stage upload method: {upload_method}")
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive")
    if num_streams <= 0:
        raise ValueError("num_streams must be positive")
    if chunk_retries < 0:
        raise ValueError("chunk_retries must be non-negative")
    if retry_base_delay_s < 0:
        raise ValueError("retry_base_delay_s must be non-negative")
    if stage_retries < 0:
        raise ValueError("stage_retries must be non-negative")
    if stage_retry_base_delay_s < 0:
        raise ValueError("stage_retry_base_delay_s must be non-negative")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    data = {"version": version, "mode": mode}
    if not relay:
        data["relay"] = "false"
    if base_version is not None:
        data["base_version"] = base_version

    async def _post_with_retries(
        admin_client: WeightAdminClient,
        route: str,
        **kwargs: Any,
    ) -> httpx.Response:
        for attempt in range(chunk_retries + 1):
            try:
                response = await admin_client.post(route, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as error:
                retryable = error.response.status_code in {408, 425, 429} or error.response.status_code >= 500
                if not retryable or attempt == chunk_retries:
                    raise
            except httpx.RequestError:
                if attempt == chunk_retries:
                    raise

            delay = retry_base_delay_s * 2**attempt
            if delay:
                await asyncio.sleep(delay + random.uniform(0, delay))

        raise AssertionError("retry loop exited unexpectedly")

    async def _retry_complete_stage(call: Callable[[], Awaitable[None]]) -> None:
        for attempt in range(stage_retries + 1):
            try:
                await call()
                return
            except Exception as error:
                if not _is_retryable_stage_error(error) or attempt == stage_retries:
                    raise

            delay = stage_retry_base_delay_s * 2**attempt
            if delay:
                await asyncio.sleep(delay)

        raise AssertionError("stage retry loop exited unexpectedly")

    def _upload_path() -> Path:
        upload_path = weight_path
        if upload_path.is_dir():
            if mode != "delta":
                raise ValueError("upload=True requires a file path for full checkpoint staging")
            preferred_name = "delta.stream" if upload_method == "streaming" else "delta.safetensors"
            fallback_name = "delta.safetensors" if upload_method == "streaming" else "delta.stream"
            preferred_path = upload_path / preferred_name
            fallback_path = upload_path / fallback_name
            upload_path = preferred_path if preferred_path.exists() or not fallback_path.exists() else fallback_path
        return upload_path

    async def _stage_path(admin_client: WeightAdminClient) -> None:
        response = await admin_client.post("/stage", data={**data, "path": weight_path.as_posix()})
        response.raise_for_status()

    async def _stage_multipart_upload(admin_client: WeightAdminClient, upload_path: Path) -> None:
        with upload_path.open("rb") as f:
            files = {"file": (upload_path.name, f, "application/octet-stream")}
            response = await admin_client.post("/stage", data=data, files=files)
        response.raise_for_status()

    async def _stage_chunked_upload(
        admin_client: WeightAdminClient,
        upload_path: Path,
        total_size: int,
        expected_sha256: str,
    ) -> None:
        chunk_data = {
            **data,
            "filename": upload_path.name,
            "total_size": str(total_size),
            "sha256": expected_sha256,
        }
        ranges = [
            (offset, min(chunk_size_bytes, total_size - offset)) for offset in range(0, total_size, chunk_size_bytes)
        ]
        if not ranges:
            ranges = [(0, 0)]

        queue: asyncio.Queue[tuple[int, int] | None] = asyncio.Queue()
        for item in ranges:
            queue.put_nowait(item)
        worker_count = min(num_streams, len(ranges))
        for _ in range(worker_count):
            queue.put_nowait(None)

        async def _upload_chunks() -> None:
            while item := await queue.get():
                offset, size = item
                with upload_path.open("rb") as f:
                    f.seek(offset)
                    chunk = f.read(size)
                if len(chunk) != size:
                    raise RuntimeError(f"short read at offset {offset}: expected {size} bytes, got {len(chunk)}")
                files = {"file": (upload_path.name, chunk, "application/octet-stream")}
                await _post_with_retries(
                    admin_client,
                    "/stage_chunk",
                    data={**chunk_data, "offset": str(offset)},
                    files=files,
                )

        async with asyncio.TaskGroup() as group:
            for _ in range(worker_count):
                group.create_task(_upload_chunks())

        await _post_with_retries(admin_client, "/stage_finalize", data=chunk_data)

    async def _stage_streaming_upload(
        admin_client: WeightAdminClient,
        upload_path: Path,
        done_path: Path | None,
        upload_id: str,
    ) -> None:
        if mode != "delta":
            raise ValueError("streaming upload currently supports delta mode only")
        if done_path is None and not upload_path.exists():
            raise FileNotFoundError(upload_path)

        init_data = {
            **data,
            "filename": upload_path.name,
            "upload_id": upload_id,
        }
        response = await _post_with_retries(admin_client, "/stage_stream_init", json=init_data)
        response_data = response.json()
        response_upload_id = response_data.get("upload_id")
        if not response_upload_id:
            raise RuntimeError("stage_stream_init response did not include upload_id")
        if response_upload_id != upload_id:
            raise RuntimeError(f"stage_stream_init returned unexpected upload_id: {response_upload_id}")

        queue: asyncio.Queue[tuple[int, int, int] | None] = asyncio.Queue(maxsize=num_streams * 2)

        async def _produce_chunks() -> None:
            chunk_index = 0
            offset = 0
            while True:
                if upload_path.exists():
                    size = upload_path.stat().st_size
                    if size > offset:
                        to_read = min(chunk_size_bytes, size - offset)
                        await queue.put((chunk_index, offset, to_read))
                        offset += to_read
                        chunk_index += 1
                        continue

                if done_path is None:
                    if upload_path.exists() and offset >= upload_path.stat().st_size:
                        break
                elif done_path.exists():
                    final_size = upload_path.stat().st_size if upload_path.exists() else 0
                    if offset >= final_size:
                        break

                await asyncio.sleep(poll_interval_s)

            for _ in range(num_streams):
                await queue.put(None)

        async def _upload_chunks() -> None:
            while item := await queue.get():
                chunk_index, offset, size = item
                with upload_path.open("rb") as f:
                    f.seek(offset)
                    chunk = f.read(size)
                if len(chunk) != size:
                    raise RuntimeError(f"short read at offset {offset}: expected {size} bytes, got {len(chunk)}")
                await _post_with_retries(
                    admin_client,
                    "/stage_stream_chunk",
                    params={"upload_id": upload_id, "chunk_index": chunk_index, "offset": offset},
                    content=chunk,
                )

        if not response_data.get("finalized", False):
            async with asyncio.TaskGroup() as group:
                group.create_task(_produce_chunks())
                for _ in range(num_streams):
                    group.create_task(_upload_chunks())

        if not upload_path.exists():
            raise FileNotFoundError(upload_path)
        final_size = upload_path.stat().st_size
        finalize_data = {
            "upload_id": upload_id,
            "final_size": str(final_size),
            "sha256": await asyncio.to_thread(_sha256_file, upload_path),
        }
        await _post_with_retries(admin_client, "/stage_stream_finalize", data=finalize_data)

    if upload:
        upload_path = _upload_path()
        if upload_method == "multipart":
            return await _run_endpoint_operations(
                admin_clients,
                "stage_weights",
                lambda admin_client: _retry_complete_stage(lambda: _stage_multipart_upload(admin_client, upload_path)),
            )
        if upload_method == "chunked":
            total_size = upload_path.stat().st_size
            expected_sha256 = await asyncio.to_thread(_sha256_file, upload_path)
            return await _run_endpoint_operations(
                admin_clients,
                "stage_weights",
                lambda admin_client: _retry_complete_stage(
                    lambda: _stage_chunked_upload(admin_client, upload_path, total_size, expected_sha256)
                ),
            )

        async def _stage_streaming_with_retries(admin_client: WeightAdminClient) -> None:
            upload_id = uuid4().hex
            await _retry_complete_stage(
                lambda: _stage_streaming_upload(admin_client, upload_path, done_path, upload_id)
            )

        return await _run_endpoint_operations(
            admin_clients,
            "stage_weights",
            _stage_streaming_with_retries,
        )
    return await _run_endpoint_operations(
        admin_clients,
        "stage_weights",
        lambda admin_client: _retry_complete_stage(lambda: _stage_path(admin_client)),
    )


async def commit_weights(
    admin_clients: list[WeightAdminClient],
    version: str,
    mode: str | None = None,
    *,
    resume: bool = True,
    relay: bool = True,
) -> list[EndpointOperationResult]:
    """Commit a staged weight version on static inference servers."""
    data = {"version": version}
    if mode is not None:
        data["mode"] = mode
    if not resume:
        data["resume"] = "false"
    if not relay:
        data["relay"] = "false"

    async def _commit(admin_client: WeightAdminClient) -> None:
        response = await admin_client.post("/commit", data=data)
        response.raise_for_status()

    async def _commit_with_pause(admin_client: WeightAdminClient) -> None:
        await _pause_engine(admin_client)
        try:
            await _commit(admin_client)
        finally:
            if resume:
                await _resume_engine(admin_client)

    return await _run_endpoint_operations(admin_clients, "commit_weights", _commit_with_pause)


async def reload_weights(
    admin_clients: list[WeightAdminClient], *, resume: bool = True, relay: bool = True
) -> list[EndpointOperationResult]:
    """Reload base model weights on static inference servers."""

    data = {}
    if not resume:
        data["resume"] = "false"
    if not relay:
        data["relay"] = "false"

    async def _reload(admin_client: WeightAdminClient) -> None:
        response = await admin_client.post("/reload_weights", data=data)
        response.raise_for_status()

    async def _reload_with_pause(admin_client: WeightAdminClient) -> None:
        await _pause_engine(admin_client)
        try:
            await _reload(admin_client)
        finally:
            if resume:
                await _resume_engine(admin_client)

    return await _run_endpoint_operations(admin_clients, "reload_weights", _reload_with_pause)


async def set_weight_serving(
    client: WeightAdminClient, *, enabled: bool, version: str | None = None, timeout_s: float = 5.0
) -> None:
    response = await client.post(
        "/weight_serving",
        params={"timeout_s": timeout_s},
        json={"enabled": enabled, "version": version},
        timeout=timeout_s,
    )
    response.raise_for_status()


class DeltaEndpointPool:
    """Tracks transactional delta state across static inference endpoints."""

    def __init__(
        self,
        clients: list[AsyncClient],
        *,
        lease_enabled: bool,
        recovery_enabled: bool,
        cooldown_s: float,
        health_timeout_s: float,
        stage_num_streams: int = 1,
        stage_chunk_size_bytes: int = STAGE_UPLOAD_CHUNK_BYTES,
        stage_chunk_retries: int = 3,
        stage_retries: int = 1,
    ) -> None:
        self.clients = clients
        self.lease_enabled = lease_enabled
        self.recovery_enabled = recovery_enabled
        self.cooldown_s = cooldown_s
        self.health_timeout_s = health_timeout_s
        self.stage_num_streams = stage_num_streams
        self.stage_chunk_size_bytes = stage_chunk_size_bytes
        self.stage_chunk_retries = stage_chunk_retries
        self.stage_retries = stage_retries
        self.runtime = {_endpoint_key(client): EndpointLeaseRuntime() for client in clients}
        self.replay_entries: dict[str, DeltaReplayEntry] = {}
        self.staged_endpoints: dict[str, set[str]] = {}
        self.active_version = "base"
        self.lock = asyncio.Lock()
        self._serving_initialized = False
        self._recovery_tasks: dict[str, asyncio.Task] = {}
        self._closed = False

    async def stage(
        self,
        weight_path: Path,
        *,
        version: str,
        base_version: str,
        upload: bool,
        upload_method: Literal["multipart", "chunked", "streaming"],
        done_path: Path | None,
    ) -> None:
        async with self.lock:
            if self._closed:
                raise RuntimeError("delta endpoint pool is closed")
            if self.lease_enabled and not self._serving_initialized:
                await self._record(
                    _run_endpoint_operations(
                        self.clients,
                        "quarantine_weights",
                        lambda client: set_weight_serving(client, enabled=False, timeout_s=self.health_timeout_s),
                    ),
                    allow_partial=True,
                )
                self._serving_initialized = True
            clients = self._healthy_clients("stage_weights")
            results = await self._record(
                stage_weights(
                    clients,
                    weight_path,
                    version=version,
                    mode="delta",
                    base_version=base_version,
                    upload=upload,
                    upload_method=upload_method,
                    chunk_size_bytes=self.stage_chunk_size_bytes,
                    num_streams=self.stage_num_streams,
                    chunk_retries=self.stage_chunk_retries,
                    stage_retries=self.stage_retries,
                    done_path=done_path,
                ),
                allow_partial=self.lease_enabled,
            )
            self.staged_endpoints[version] = {result.endpoint for result in results if result.ok}
            self.replay_entries[version] = DeltaReplayEntry(
                weight_path=weight_path,
                version=version,
                base_version=base_version,
                upload=upload,
                upload_method=upload_method,
                done_path=done_path,
                sha256=await asyncio.to_thread(_sha256_file, _delta_artifact_path(weight_path, upload_method)),
            )

    async def commit(self, version: str) -> None:
        async with self.lock:
            entry = self.replay_entries[version]
            healthy = self._healthy_clients("commit_weights")
            staged = self.staged_endpoints.get(version, set())
            missing = [client for client in healthy if _endpoint_key(client) not in staged]
            if missing:
                results = await self._record(
                    stage_weights(
                        missing,
                        entry.weight_path,
                        version=entry.version,
                        mode="delta",
                        base_version=entry.base_version,
                        upload=entry.upload,
                        upload_method=entry.upload_method,
                        chunk_size_bytes=self.stage_chunk_size_bytes,
                        num_streams=self.stage_num_streams,
                        chunk_retries=self.stage_chunk_retries,
                        stage_retries=self.stage_retries,
                        done_path=entry.done_path,
                    ),
                    allow_partial=self.lease_enabled,
                )
                staged.update(result.endpoint for result in results if result.ok)

            commit_clients = [
                client for client in self._healthy_clients("commit_weights") if _endpoint_key(client) in staged
            ]
            if not commit_clients:
                raise RuntimeError(f"No staged inference endpoint is available for delta version {version}.")
            results = await self._record(
                commit_weights(commit_clients, version=version, mode="delta", resume=not self.lease_enabled),
                allow_partial=self.lease_enabled,
            )
            self.active_version = version
            if self.lease_enabled:
                successful = {result.endpoint for result in results if result.ok}
                await self._record(
                    _run_endpoint_operations(
                        [client for client in commit_clients if _endpoint_key(client) in successful],
                        "enable_weight_serving",
                        lambda client: set_weight_serving(
                            client, enabled=True, version=version, timeout_s=self.health_timeout_s
                        ),
                    ),
                    allow_partial=True,
                )

    def metrics(self) -> dict[str, float]:
        state_values = {"healthy": 0.0, "retired": 2.0, "recovering": 3.0}
        metrics: dict[str, float] = {}
        for index, client in enumerate(self.clients):
            runtime = self.runtime[_endpoint_key(client)]
            prefix = f"inference_endpoint/{index}"
            metrics[f"{prefix}/state"] = state_values[runtime.state]
            metrics[f"{prefix}/retired_count"] = float(runtime.retired_count)
            metrics[f"{prefix}/recovery_attempts"] = float(runtime.recovery_attempt_count)
            metrics[f"{prefix}/recovery_successes"] = float(runtime.recovery_success_count)
            metrics[f"{prefix}/replayed_deltas"] = float(runtime.replayed_delta_count)
            metrics[f"{prefix}/reloads"] = float(runtime.reload_count)
        return metrics

    def _healthy_clients(self, operation: str) -> list[AsyncClient]:
        if not self.lease_enabled:
            return self.clients
        clients = [client for client in self.clients if self.runtime[_endpoint_key(client)].state == "healthy"]
        if not clients:
            raise RuntimeError(f"No healthy inference endpoints are available for {operation}.")
        return clients

    async def _record(
        self,
        operation: Awaitable[list[EndpointOperationResult]],
        *,
        allow_partial: bool,
    ) -> list[EndpointOperationResult]:
        try:
            return await operation
        except EndpointOperationError as error:
            failed = {result.endpoint for result in error.results if not result.ok}
            for result in error.results:
                if not result.ok:
                    self._retire(result.endpoint, f"{result.operation} failed: {result.error}")
            if self.lease_enabled:
                await asyncio.gather(
                    *(self._quarantine(client) for client in self.clients if _endpoint_key(client) in failed)
                )
            self._schedule_recoveries()
            if allow_partial and any(result.ok for result in error.results):
                return error.results
            raise

    def _retire(self, endpoint: str, reason: str) -> None:
        runtime = self.runtime[endpoint]
        runtime.state = "retired"
        runtime.retry_after = time.monotonic() + self.cooldown_s
        runtime.retired_count += 1
        runtime.last_reason = reason

    async def _quarantine(self, client: AsyncClient) -> None:
        try:
            await set_weight_serving(client, enabled=False, timeout_s=self.health_timeout_s)
        except httpx.HTTPError as error:
            get_logger().warning(f"Could not quarantine {_endpoint_key(client)}: {error}")

    def _schedule_recoveries(self) -> None:
        if not self.recovery_enabled or self._closed:
            return
        for client in self.clients:
            endpoint = _endpoint_key(client)
            task = self._recovery_tasks.get(endpoint)
            if self.runtime[endpoint].state != "retired" or (task is not None and not task.done()):
                continue
            task = asyncio.create_task(self._recovery_loop(client), name=f"delta_recovery:{endpoint}")
            self._recovery_tasks[endpoint] = task
            task.add_done_callback(lambda done, key=endpoint: self._recovery_finished(key, done))

    def _recovery_finished(self, endpoint: str, task: asyncio.Task) -> None:
        if self._recovery_tasks.get(endpoint) is task:
            self._recovery_tasks.pop(endpoint)

    async def _recovery_loop(self, client: AsyncClient) -> None:
        endpoint = _endpoint_key(client)
        while not self._closed:
            runtime = self.runtime[endpoint]
            if runtime.retry_after is not None:
                await asyncio.sleep(max(0, runtime.retry_after - time.monotonic()))
            async with self.lock:
                if runtime.state != "retired":
                    return
                runtime.state = "recovering"
                runtime.recovery_attempt_count += 1
                runtime.retry_after = None
                for staged in self.staged_endpoints.values():
                    staged.discard(endpoint)
            try:
                await self._recover(client)
            except asyncio.CancelledError:
                await asyncio.shield(self._quarantine(client))
                raise
            except Exception as error:
                async with self.lock:
                    self._retire(endpoint, f"delta recovery failed: {error}")
                get_logger().warning(f"Delta recovery failed on {endpoint}: {error}")
                await asyncio.sleep(self.health_timeout_s)
            else:
                return

    async def _recover(self, client: AsyncClient) -> None:
        endpoint = _endpoint_key(client)
        runtime = self.runtime[endpoint]
        await set_weight_serving(client, enabled=False, timeout_s=self.health_timeout_s)
        response = await client.get(
            "/weight_status", params={"timeout_s": self.health_timeout_s}, timeout=self.health_timeout_s
        )
        response.raise_for_status()
        status = response.json()
        nodes: list[tuple[WeightAdminClient, dict[str, Any]]] = [(client, status)]
        for index, peer in enumerate(status["peers"]):
            if "error" in peer:
                raise RuntimeError(f"relay peer {peer['url']} is unavailable: {peer['error']}")
            nodes.append((RelayPeerClient(client, index), peer))

        async with self.lock:
            target_version = self.active_version
        versions: list[str] = []
        for node, node_status in nodes:
            version = self._trusted_version(node_status, target_version)
            if version is None:
                await reload_weights([node], resume=False, relay=False)
                runtime.reload_count += 1
                version = "base"
            versions.append(version)

        while True:
            async with self.lock:
                target_version = self.active_version
            for index, (node, _) in enumerate(nodes):
                for entry in self._replay_plan(versions[index], target_version):
                    await self._replay(node, entry)
                    versions[index] = entry.version
                    runtime.replayed_delta_count += 1
            await set_weight_serving(client, enabled=True, version=target_version, timeout_s=self.health_timeout_s)
            async with self.lock:
                if self.active_version == target_version:
                    runtime.state = "healthy"
                    runtime.recovery_success_count += 1
                    runtime.last_reason = f"caught up to version {target_version}"
                    self.staged_endpoints.setdefault(target_version, set()).add(endpoint)
                    return
            await set_weight_serving(client, enabled=False, timeout_s=self.health_timeout_s)

    async def _replay(self, client: WeightAdminClient, entry: DeltaReplayEntry) -> None:
        await stage_weights(
            [client],
            entry.weight_path,
            version=entry.version,
            mode="delta",
            base_version=entry.base_version,
            upload=entry.upload,
            upload_method=entry.upload_method,
            chunk_size_bytes=self.stage_chunk_size_bytes,
            num_streams=self.stage_num_streams,
            chunk_retries=self.stage_chunk_retries,
            stage_retries=self.stage_retries,
            done_path=entry.done_path,
            relay=False,
        )
        await commit_weights([client], version=entry.version, mode="delta", resume=False, relay=False)

    def _trusted_version(self, status: dict[str, Any], target_version: str) -> str | None:
        if status["weights_dirty"]:
            return None
        version = status["active_version"]
        if version == "base":
            return "base" if status["active_sha256"] is None else None
        entry = self.replay_entries.get(version)
        if entry is None or status["active_sha256"] != entry.sha256:
            return None
        try:
            self._replay_plan(version, target_version)
        except RuntimeError:
            return None
        return version

    def _replay_plan(self, base_version: str = "base", target_version: str | None = None) -> list[DeltaReplayEntry]:
        version = self.active_version if target_version is None else target_version
        plan: list[DeltaReplayEntry] = []
        seen: set[str] = set()
        while version != base_version:
            if version in seen:
                raise RuntimeError(f"delta replay chain contains a cycle at version {version}")
            seen.add(version)
            entry = self.replay_entries.get(version)
            if entry is None:
                raise RuntimeError(f"missing delta replay entry for version {version} (base={base_version})")
            plan.append(entry)
            version = entry.base_version
        return list(reversed(plan))

    async def aclose(self) -> None:
        self._closed = True
        tasks = list(self._recovery_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._recovery_tasks.clear()
