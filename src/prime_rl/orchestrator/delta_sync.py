from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from httpx import AsyncClient

STAGE_UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024


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
    last_reason: str | None = None


@dataclass(frozen=True)
class DeltaReplayEntry:
    weight_path: Path
    version: str
    base_version: str
    upload: bool
    upload_method: Literal["multipart", "chunked", "streaming"]
    done_path: Path | None


def _endpoint_key(admin_client: AsyncClient) -> str:
    return str(admin_client.base_url).rstrip("/").removesuffix("/v1")


def _format_exception(exception: BaseException) -> str:
    message = str(exception).strip()
    if message:
        return f"{exception.__class__.__name__}: {message}"
    return exception.__class__.__name__


async def _run_endpoint_operations(
    admin_clients: list[AsyncClient],
    operation: str,
    call: Callable[[AsyncClient], Awaitable[None]],
) -> list[EndpointOperationResult]:
    async def _run(admin_client: AsyncClient) -> EndpointOperationResult:
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


async def _pause_engine(client: AsyncClient) -> None:
    response = await client.post("/pause", params={"mode": "keep", "clear_cache": "false"})
    response.raise_for_status()


async def _resume_engine(client: AsyncClient) -> None:
    response = await client.post("/resume")
    response.raise_for_status()


async def stage_weights(
    admin_clients: list[AsyncClient],
    weight_path: Path,
    version: str,
    mode: str = "full",
    base_version: str | None = None,
    upload: bool = False,
    upload_method: Literal["multipart", "chunked", "streaming"] = "multipart",
    chunk_size_bytes: int = STAGE_UPLOAD_CHUNK_BYTES,
    done_path: Path | None = None,
    poll_interval_s: float = 0.1,
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
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    data = {"version": version, "mode": mode}
    if base_version is not None:
        data["base_version"] = base_version

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

    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(STAGE_UPLOAD_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()

    async def _stage_path(admin_client: AsyncClient) -> None:
        response = await admin_client.post("/stage", data={**data, "path": weight_path.as_posix()})
        response.raise_for_status()

    async def _stage_multipart_upload(admin_client: AsyncClient, upload_path: Path) -> None:
        with upload_path.open("rb") as f:
            files = {"file": (upload_path.name, f, "application/octet-stream")}
            response = await admin_client.post("/stage", data=data, files=files)
        response.raise_for_status()

    async def _stage_chunked_upload(
        admin_client: AsyncClient,
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
        with upload_path.open("rb") as f:
            offset = 0
            sent_chunk = False
            while chunk := f.read(chunk_size_bytes):
                sent_chunk = True
                files = {"file": (upload_path.name, chunk, "application/octet-stream")}
                response = await admin_client.post(
                    "/stage_chunk",
                    data={**chunk_data, "offset": str(offset)},
                    files=files,
                )
                response.raise_for_status()
                offset += len(chunk)

            if not sent_chunk:
                files = {"file": (upload_path.name, b"", "application/octet-stream")}
                response = await admin_client.post("/stage_chunk", data={**chunk_data, "offset": "0"}, files=files)
                response.raise_for_status()

        response = await admin_client.post("/stage_finalize", data=chunk_data)
        response.raise_for_status()

    async def _stage_streaming_upload(
        admin_client: AsyncClient,
        upload_path: Path,
        done_path: Path | None,
    ) -> None:
        if mode != "delta":
            raise ValueError("streaming upload currently supports delta mode only")
        if done_path is None and not upload_path.exists():
            raise FileNotFoundError(upload_path)

        init_data = {
            **data,
            "filename": upload_path.name,
        }
        response = await admin_client.post("/stage_stream_init", json=init_data)
        response.raise_for_status()
        upload_id = response.json().get("upload_id")
        if not upload_id:
            raise RuntimeError("stage_stream_init response did not include upload_id")

        chunk_index = 0
        offset = 0
        while True:
            if upload_path.exists():
                size = upload_path.stat().st_size
                if size > offset:
                    to_read = min(chunk_size_bytes, size - offset)
                    with upload_path.open("rb") as f:
                        f.seek(offset)
                        chunk = f.read(to_read)
                    if chunk:
                        response = await admin_client.post(
                            "/stage_stream_chunk",
                            params={"upload_id": upload_id, "chunk_index": chunk_index, "offset": offset},
                            content=chunk,
                        )
                        response.raise_for_status()
                        offset += len(chunk)
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

        if not upload_path.exists():
            raise FileNotFoundError(upload_path)
        final_size = upload_path.stat().st_size
        finalize_data = {
            "upload_id": upload_id,
            "final_size": str(final_size),
            "sha256": _sha256_file(upload_path),
        }
        response = await admin_client.post("/stage_stream_finalize", data=finalize_data)
        response.raise_for_status()

    if upload:
        upload_path = _upload_path()
        if upload_method == "multipart":
            return await _run_endpoint_operations(
                admin_clients,
                "stage_weights",
                lambda admin_client: _stage_multipart_upload(admin_client, upload_path),
            )
        if upload_method == "chunked":
            total_size = upload_path.stat().st_size
            expected_sha256 = _sha256_file(upload_path)
            return await _run_endpoint_operations(
                admin_clients,
                "stage_weights",
                lambda admin_client: _stage_chunked_upload(admin_client, upload_path, total_size, expected_sha256),
            )
        return await _run_endpoint_operations(
            admin_clients,
            "stage_weights",
            lambda admin_client: _stage_streaming_upload(admin_client, upload_path, done_path),
        )
    return await _run_endpoint_operations(admin_clients, "stage_weights", _stage_path)


async def commit_weights(
    admin_clients: list[AsyncClient], version: str, mode: str | None = None
) -> list[EndpointOperationResult]:
    """Commit a staged weight version on static inference servers."""
    data = {"version": version}
    if mode is not None:
        data["mode"] = mode

    async def _commit(admin_client: AsyncClient) -> None:
        response = await admin_client.post("/commit", data=data)
        response.raise_for_status()

    async def _commit_with_pause(admin_client: AsyncClient) -> None:
        await _pause_engine(admin_client)
        try:
            await _commit(admin_client)
        finally:
            await _resume_engine(admin_client)

    return await _run_endpoint_operations(admin_clients, "commit_weights", _commit_with_pause)


async def reload_weights(admin_clients: list[AsyncClient]) -> list[EndpointOperationResult]:
    """Reload base model weights on static inference servers."""

    async def _reload(admin_client: AsyncClient) -> None:
        response = await admin_client.post("/reload_weights")
        response.raise_for_status()

    async def _reload_with_pause(admin_client: AsyncClient) -> None:
        await _pause_engine(admin_client)
        try:
            await _reload(admin_client)
        finally:
            await _resume_engine(admin_client)

    return await _run_endpoint_operations(admin_clients, "reload_weights", _reload_with_pause)


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
    ) -> None:
        self.clients = clients
        self.lease_enabled = lease_enabled
        self.recovery_enabled = recovery_enabled
        self.cooldown_s = cooldown_s
        self.health_timeout_s = health_timeout_s
        self.runtime = {_endpoint_key(client): EndpointLeaseRuntime() for client in clients}
        self.replay_entries: dict[str, DeltaReplayEntry] = {}
        self.staged_endpoints: dict[str, set[str]] = {}
        self.active_version = "base"
        self.lock = asyncio.Lock()

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
            await self._recover_eligible()
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
            await self._record(
                commit_weights(commit_clients, version=version, mode="delta"),
                allow_partial=self.lease_enabled,
            )
            self.active_version = version
            await self._recover_eligible()

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
            for result in error.results:
                if not result.ok:
                    self._retire(result.endpoint, f"{result.operation} failed: {result.error}")
            if allow_partial and any(result.ok for result in error.results):
                return error.results
            raise

    def _retire(self, endpoint: str, reason: str) -> None:
        runtime = self.runtime[endpoint]
        runtime.state = "retired"
        runtime.retry_after = time.monotonic() + self.cooldown_s
        runtime.retired_count += 1
        runtime.last_reason = reason

    async def _recover_eligible(self) -> None:
        if not self.recovery_enabled:
            return
        now = time.monotonic()
        for client in self.clients:
            endpoint = _endpoint_key(client)
            runtime = self.runtime[endpoint]
            if runtime.state != "retired" or (runtime.retry_after is not None and now < runtime.retry_after):
                continue
            if not await self._healthy(client):
                continue
            await self._recover(client)

    async def _healthy(self, client: AsyncClient) -> bool:
        try:
            response = await client.get("/health", timeout=self.health_timeout_s)
            if response.status_code != 404:
                response.raise_for_status()
        except Exception:
            return False
        return True

    async def _recover(self, client: AsyncClient) -> None:
        endpoint = _endpoint_key(client)
        runtime = self.runtime[endpoint]
        runtime.state = "recovering"
        runtime.recovery_attempt_count += 1
        runtime.retry_after = None
        for staged in self.staged_endpoints.values():
            staged.discard(endpoint)
        try:
            await reload_weights([client])
            for entry in self._replay_plan():
                await stage_weights(
                    [client],
                    entry.weight_path,
                    version=entry.version,
                    mode="delta",
                    base_version=entry.base_version,
                    upload=entry.upload,
                    upload_method=entry.upload_method,
                    done_path=entry.done_path,
                )
                await commit_weights([client], version=entry.version, mode="delta")
                self.staged_endpoints.setdefault(entry.version, set()).add(endpoint)
        except Exception as error:
            self._retire(endpoint, f"delta replay recovery failed: {error}")
            return
        runtime.state = "healthy"
        runtime.recovery_success_count += 1
        runtime.last_reason = f"replayed delta chain to version {self.active_version}"

    def _replay_plan(self) -> list[DeltaReplayEntry]:
        if self.active_version == "base":
            return []
        entries = sorted(self.replay_entries.values(), key=lambda entry: int(entry.version))
        plan: list[DeltaReplayEntry] = []
        expected_base = "base"
        for entry in entries:
            if entry.base_version != expected_base:
                raise RuntimeError(
                    f"delta replay chain is broken at version {entry.version}: "
                    f"expected base {expected_base}, got {entry.base_version}"
                )
            plan.append(entry)
            expected_base = entry.version
            if entry.version == self.active_version:
                return plan
        raise RuntimeError(f"missing delta replay entry for active version {self.active_version}")
