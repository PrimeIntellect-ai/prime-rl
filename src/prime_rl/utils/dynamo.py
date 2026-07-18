from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import httpx
from httpx import AsyncClient
from tenacity import AsyncRetrying, retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import (
    LORA_LOAD_READ_TIMEOUT_S,
    LORA_LOAD_TOTAL_TIMEOUT_S,
    StaticInferencePool,
    _is_retryable_lora_error,
    _pause_engines,
    _resume_engines,
    init_nccl_broadcast,
    setup_admin_clients,
    update_weights,
)

DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class DiscoveredDynamoWorker:
    component: str
    instance_id: int
    admin_base_url: str
    world_size: int
    system_url: str | None
    system_routes: tuple[str, ...]


class DynamoDiscoveryPending(ValueError):
    """A well-formed discovery snapshot that is not ready yet."""


def _is_retryable_dynamo_error(exception: BaseException) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code == 429 or exception.response.status_code >= 500
    return isinstance(exception, (DynamoDiscoveryPending, httpx.TransportError))


def _parse_dynamo_worker(raw_worker: object, model_name: str) -> DiscoveredDynamoWorker:
    if not isinstance(raw_worker, dict):
        raise ValueError("Dynamo RL discovery returned a malformed worker")
    if raw_worker.get("error"):
        raise DynamoDiscoveryPending(f"Dynamo RL worker probe is not ready: {raw_worker['error']}")
    if raw_worker.get("model") != model_name:
        raise ValueError(f"Dynamo RL worker model {raw_worker.get('model')!r} does not match {model_name!r}")

    component = raw_worker.get("component")
    instance_id = raw_worker.get("instance_id")
    admin_base_url = raw_worker.get("admin_base_url")
    system_url = raw_worker.get("system_url")
    world_size = raw_worker.get("world_size")
    system_routes = raw_worker.get("system_routes", [])
    if not isinstance(component, str) or not component:
        raise ValueError("Dynamo RL worker is missing component identity")
    if not isinstance(instance_id, int) or isinstance(instance_id, bool) or instance_id < 0:
        raise ValueError("Dynamo RL worker has an invalid instance_id")
    if not isinstance(admin_base_url, str) or not admin_base_url:
        raise ValueError("Dynamo RL worker is missing admin_base_url")
    if system_url is not None and (not isinstance(system_url, str) or not system_url):
        raise ValueError("Dynamo RL worker has an invalid system_url")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("Dynamo RL worker has an invalid world_size")
    if not isinstance(system_routes, list) or any(not isinstance(route, str) or not route for route in system_routes):
        raise ValueError("Dynamo RL worker has an invalid system_routes list")
    admin_base_url = _normalize_control_url(admin_base_url, "admin_base_url")
    if system_url is not None:
        system_url = _normalize_control_url(system_url, "system_url")
    return DiscoveredDynamoWorker(
        component=component,
        instance_id=instance_id,
        admin_base_url=admin_base_url,
        world_size=world_size,
        system_url=system_url,
        system_routes=tuple(sorted(set(system_routes))),
    )


def _validate_dynamo_snapshot(workers: tuple[DiscoveredDynamoWorker, ...]) -> None:
    if not workers:
        raise DynamoDiscoveryPending("Dynamo RL discovery returned no workers yet")
    identities = [(worker.component, worker.instance_id) for worker in workers]
    admin_urls = [worker.admin_base_url for worker in workers]
    system_urls = [worker.system_url for worker in workers if worker.system_url is not None]
    if len(set(identities)) != len(identities):
        raise ValueError("Dynamo RL discovery returned duplicate worker identities")
    if len(set(admin_urls)) != len(admin_urls):
        raise ValueError("Dynamo RL discovery returned duplicate admin endpoints")
    if len(set(system_urls)) != len(system_urls):
        raise ValueError("Dynamo RL discovery returned duplicate system endpoints")
    lora_route_presence = ["update/load_lora" in worker.system_routes for worker in workers]
    if any(lora_route_presence) and not all(lora_route_presence):
        raise ValueError("Dynamo RL discovery returned a partial update/load_lora capability snapshot")
    if all(lora_route_presence) and any(worker.system_url is None for worker in workers):
        raise ValueError("Dynamo RL discovery returned update/load_lora without a system_url")


def _parse_dynamo_workers(payload: object, model_name: str) -> tuple[DiscoveredDynamoWorker, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("workers"), list):
        raise ValueError("Dynamo RL discovery response must contain a workers list")
    protocol_version = payload.get("protocol_version")
    if type(protocol_version) is not int or protocol_version != DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION:
        raise ValueError(
            "Dynamo RL discovery returned an unsupported protocol version; "
            f"expected {DYNAMO_RL_DISCOVERY_PROTOCOL_VERSION}"
        )
    selected_workers = [
        worker
        for worker in payload["workers"]
        if not isinstance(worker, dict) or not isinstance(worker.get("model"), str) or worker["model"] == model_name
    ]
    workers = tuple(_parse_dynamo_worker(worker, model_name) for worker in selected_workers)
    _validate_dynamo_snapshot(workers)
    return tuple(sorted(workers, key=lambda worker: (worker.component, worker.instance_id)))


def _setup_control_clients(urls: list[str]) -> list[AsyncClient]:
    return [
        AsyncClient(
            base_url=url.rstrip("/"),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(None),
        )
        for url in urls
    ]


def _normalize_control_url(value: str, field: str) -> str:
    try:
        url = httpx.URL(value)
    except httpx.InvalidURL as error:
        raise ValueError(f"Dynamo RL worker has an invalid {field}") from error
    if (
        url.scheme != "http"
        or not url.host
        or url.userinfo
        or url.query
        or url.fragment
        or url.path not in ("/", "/v1", "/v1/")
    ):
        raise ValueError(f"Dynamo RL worker has an invalid {field}")
    return str(url.copy_with(path="/", query=None, fragment=None)).rstrip("/")


async def _load_lora_adapter(update_clients: list[AsyncClient], lora_name: str, lora_path: Path) -> None:
    timeout = httpx.Timeout(connect=10.0, read=LORA_LOAD_READ_TIMEOUT_S, write=60.0, pool=10.0)
    payload = {
        "lora_name": lora_name,
        "source": {"uri": lora_path.resolve().as_uri()},
        "load_inplace": True,
    }

    @retry(
        retry=retry_if_exception(_is_retryable_lora_error),
        stop=stop_after_delay(LORA_LOAD_TOTAL_TIMEOUT_S) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _load(update_client: AsyncClient) -> None:
        response = await update_client.post("/v1/loras", json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and result.get("status") == "error":
            raise RuntimeError(result.get("message") or "Dynamo LoRA update failed")

    await asyncio.gather(*(_load(update_client) for update_client in update_clients))


async def _wait_for_model(clients: list[AsyncClient], model_name: str, timeout: int) -> None:
    async for attempt in AsyncRetrying(
        stop=stop_after_delay(timeout),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
        retry=retry_if_exception(_is_retryable_dynamo_error),
        reraise=True,
    ):
        with attempt:
            responses = await asyncio.gather(*(client.get("/v1/models") for client in clients))
            for response in responses:
                response.raise_for_status()
                models = response.json().get("data", [])
                if not any(model.get("id") == model_name for model in models):
                    raise DynamoDiscoveryPending(f"Dynamo frontend has not published model {model_name!r}")


class DynamoInferencePool(StaticInferencePool):
    """Static request pool whose direct vLLM admin clients come from Dynamo discovery."""

    def __init__(self, client_config: ClientConfig, workers: tuple[DiscoveredDynamoWorker, ...], **kwargs):
        admin_clients = _setup_control_clients([worker.admin_base_url for worker in workers])
        super().__init__(
            client_config,
            admin_clients=admin_clients,
            **kwargs,
        )
        self._admin_world_sizes = [worker.world_size for worker in workers]
        self._lora_update_clients = []
        if workers and all("update/load_lora" in worker.system_routes for worker in workers):
            system_urls = [worker.system_url for worker in workers if worker.system_url is not None]
            self._lora_update_clients = _setup_control_clients(system_urls)
        frontend_config = client_config.model_copy(update={"admin_base_url": None})
        self._frontend_model_clients = setup_admin_clients(frontend_config)

    async def init_nccl_broadcast(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int | None,
        quantize_in_weight_transfer: bool,
    ) -> None:
        await init_nccl_broadcast(
            self._admin_clients,
            host=host,
            port=port,
            timeout=timeout,
            inference_world_size=inference_world_size,
            engine_world_sizes=self._admin_world_sizes,
            quantize_in_weight_transfer=quantize_in_weight_transfer,
            use_native_collective_rpc=True,
        )

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        if lora_name is not None and weight_dir is not None:
            if not self._lora_update_clients:
                raise RuntimeError(
                    "Dynamo LoRA update requires every worker to advertise system_url and update/load_lora"
                )
            try:
                await _pause_engines(self._admin_clients, step=step)
                await _load_lora_adapter(self._lora_update_clients, lora_name, weight_dir)
                await _wait_for_model(
                    self._frontend_model_clients,
                    lora_name,
                    timeout=self._wait_for_ready_timeout,
                )
            finally:
                await _resume_engines(self._admin_clients)
            return
        await update_weights(
            self._admin_clients,
            weight_dir,
            step=step,
            use_native_collective_rpc=True,
        )

    async def stop(self) -> None:
        await super().stop()
        await asyncio.gather(
            *(client.aclose() for client in [*self._lora_update_clients, *self._frontend_model_clients])
        )

    @classmethod
    async def from_config(
        cls,
        client_config: ClientConfig,
        model_name: str,
        expected_inference_world_size: int | None = None,
        **kwargs,
    ) -> DynamoInferencePool:
        if client_config.dynamo_discovery_url is None:
            raise ValueError("Dynamo inference pool requires dynamo_discovery_url")
        discovery_url = client_config.dynamo_discovery_url.rstrip("/").removesuffix("/v1")
        async with AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            workers = None
            async for attempt in AsyncRetrying(
                stop=stop_after_delay(client_config.wait_for_ready_timeout),
                wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
                retry=retry_if_exception(_is_retryable_dynamo_error),
                reraise=True,
            ):
                with attempt:
                    response = await client.get(f"{discovery_url}/v1/rl/workers")
                    response.raise_for_status()
                    workers = _parse_dynamo_workers(response.json(), model_name)
                    discovered_world_size = sum(worker.world_size for worker in workers)
                    if (
                        expected_inference_world_size is not None
                        and discovered_world_size != expected_inference_world_size
                    ):
                        raise DynamoDiscoveryPending(
                            "Dynamo RL discovery returned "
                            f"inference_world_size={discovered_world_size}; "
                            f"waiting for expected inference_world_size={expected_inference_world_size}"
                        )
        assert workers is not None
        return cls(client_config, workers, model_name=model_name, **kwargs)
