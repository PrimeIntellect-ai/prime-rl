"""Dynamo worker discovery and engine administration."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from httpx import AsyncClient
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger

NCCL_READY_MARKER = "NCCL_READY"
ADMIN_TIMEOUT_S = 300.0
UPDATE_WEIGHTS_TIMEOUT_S = 720.0
_REQUIRED_ROUTES = frozenset(
    {
        "init_weights_update_group",
        "pause_generation",
        "resume_generation",
        "update_weights_from_disk",
        "update_weights_from_distributed",
    }
)


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


async def discover_worker_urls(
    discovery_clients: list[AsyncClient],
    timeout: int,
    model_name: str | None = None,
) -> list[str]:
    """Wait for a complete Dynamo worker set and return stable system URLs."""
    logger = get_logger()
    last_error: Exception | None = None
    deadline = asyncio.get_running_loop().time() + timeout

    while asyncio.get_running_loop().time() < deadline:
        try:
            results = await asyncio.gather(*(client.get("/v1/rl/workers") for client in discovery_clients))
            workers: list[dict] = []
            for response in results:
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("workers"), list):
                    raise ValueError("Dynamo worker discovery returned an invalid response")
                workers.extend(payload["workers"])

            urls: list[str] = []
            for worker in workers:
                if worker.get("error"):
                    raise RuntimeError(
                        f"Dynamo worker {worker.get('component', '<unknown>')} is unhealthy: {worker['error']}"
                    )
                system_url = worker.get("system_url")
                if not system_url:
                    raise ValueError("Dynamo worker discovery response is missing system_url")
                worker_model = worker.get("model")
                if model_name is not None and worker_model not in (None, model_name):
                    raise ValueError(f"Dynamo worker {system_url} serves {worker_model!r}, expected {model_name!r}")
                missing = _REQUIRED_ROUTES - set(worker.get("routes", []))
                if missing:
                    raise ValueError(f"Dynamo worker {system_url} is missing RL routes: {sorted(missing)}")
                urls.append(_root_url(system_url))

            if urls:
                if len(set(urls)) != len(urls):
                    raise ValueError("Dynamo worker discovery returned duplicate system URLs")
                resolved = sorted(urls)
                logger.info(f"Discovered {len(resolved)} Dynamo inference worker(s)")
                return resolved
            last_error = RuntimeError("Dynamo worker discovery returned no workers")
        except Exception as exc:
            last_error = exc
        await asyncio.sleep(1)

    raise TimeoutError(f"Dynamo workers were not ready after {timeout} seconds: {last_error}")


def validate_worker_membership(expected: tuple[str, ...], discovered: list[str]) -> None:
    if tuple(discovered) != expected:
        raise RuntimeError(
            f"Dynamo worker membership changed after initialization: expected {list(expected)}, discovered {discovered}"
        )


class DynamoAdminAPI:
    """Typed adapter for Dynamo's per-worker ``/engine`` endpoints."""

    def __init__(self) -> None:
        self._distributed_updates = False

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
    ) -> dict:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(self._retryable),
            stop=stop_after_delay(2 * timeout_s) | stop_after_attempt(10),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        ):
            with attempt:
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
        raise AssertionError("unreachable")

    async def initialize_nccl(
        self,
        clients: list[AsyncClient],
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int | None,
        quantize_in_weight_transfer: bool,
    ) -> None:
        world_size = inference_world_size or len(clients)
        if not clients or world_size % len(clients) != 0:
            raise ValueError(f"inference_world_size={world_size} must be divisible by {len(clients)} Dynamo workers")
        gpus_per_worker = world_size // len(clients)
        await asyncio.gather(
            *(
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
            )
        )
        self._distributed_updates = True

    async def update_weights(self, clients: list[AsyncClient], weight_dir: Path | None, step: int) -> None:
        try:
            await asyncio.gather(
                *(self._post(client, "pause_generation", {"mode": "keep", "clear_cache": False}) for client in clients)
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

            await asyncio.gather(
                *(self._post(client, method, body, timeout_s=UPDATE_WEIGHTS_TIMEOUT_S) for client in clients)
            )
        finally:
            await asyncio.gather(*(self._post(client, "resume_generation") for client in clients))
