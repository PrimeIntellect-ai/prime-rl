from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from itertools import cycle
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx
import verifiers as vf
from httpx import AsyncClient
from openai import NotFoundError
from renderers import RendererConfig
from tenacity import retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger

# Identity tuple used by ``select_train_client`` to key load counts. ``api_base_url``
# distinguishes servers; ``X-data-parallel-rank`` distinguishes DP shards within a
# server, since the router uses that header to route to specific GPU ranks.
ClientIdentity = tuple[str, str | None]


def client_identity(client: vf.ClientConfig) -> ClientIdentity:
    """Stable identity for load balancing across inference clients."""
    return (client.api_base_url, client.extra_headers.get("X-data-parallel-rank"))


class AdminAPI(Protocol):
    """Admin endpoints for an inference backend.

    Per-method: construct one HTTP call. Per-server parallelism, retry, and
    raise-for-status policy live in the caller.
    """

    async def health(self, client: AsyncClient) -> None: ...
    async def list_models(self, client: AsyncClient) -> list[dict]: ...
    async def pause(self, client: AsyncClient) -> None: ...
    async def resume(self, client: AsyncClient) -> None: ...
    async def update_weights(self, client: AsyncClient, weight_dir: str | None) -> None: ...
    async def load_lora_adapter(
        self,
        client: AsyncClient,
        lora_name: str,
        lora_path: str,
        *,
        timeout: httpx.Timeout,
    ) -> None: ...
    async def init_broadcaster(
        self,
        client: AsyncClient,
        *,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool,
    ) -> None: ...


class VLLMAdminAPI:
    """vLLM admin endpoints."""

    async def health(self, client: AsyncClient) -> None:
        # No raise_for_status: any HTTP response means the server is up.
        # Only transport errors mean "not ready yet" (caller retries).
        await client.get("/health")

    async def list_models(self, client: AsyncClient) -> list[dict]:
        response = await client.get("/v1/models")
        return response.json()["data"]

    async def pause(self, client: AsyncClient) -> None:
        response = await client.post("/pause", params={"mode": "keep", "clear_cache": "false"})
        response.raise_for_status()

    async def resume(self, client: AsyncClient) -> None:
        response = await client.post("/resume")
        response.raise_for_status()

    async def update_weights(self, client: AsyncClient, weight_dir: str | None) -> None:
        response = await client.post("/update_weights", json={"weight_dir": weight_dir})
        response.raise_for_status()

    async def load_lora_adapter(
        self,
        client: AsyncClient,
        lora_name: str,
        lora_path: str,
        *,
        timeout: httpx.Timeout,
    ) -> None:
        response = await client.post(
            "/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
            timeout=timeout,
        )
        response.raise_for_status()

    async def init_broadcaster(
        self,
        client: AsyncClient,
        *,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool,
    ) -> None:
        response = await client.post(
            "/init_broadcaster",
            json={
                "host": host,
                "port": port,
                "rank_offset": rank_offset,
                "inference_world_size": inference_world_size,
                "timeout": timeout,
                "quantize_in_weight_transfer": quantize_in_weight_transfer,
            },
        )
        response.raise_for_status()


class DynamoAdminAPI(VLLMAdminAPI):
    """NVIDIA Dynamo worker admin endpoints via ``POST /engine/<method>``.

    Each Dynamo worker exposes engine routes on its system status server
    (``DYN_SYSTEM_PORT``, default 8081). Multi-worker deployments are handled by
    iterating over ``admin_clients``.

    Args:
        engine_rpc: The ``collective_rpc`` target forwarded by
            ``update_weights_from_disk``.  Use ``"reload_weights"`` for plain
            vLLM / dynamo.vllm without a worker extension (default).  Use
            ``"update_weights_from_path"`` only when
            FileSystemWeightUpdateWorker / NCCLWeightUpdateWorker is loaded via
            ``--worker-extension-cls``.
    """

    def __init__(self, engine_rpc: str = "reload_weights", weight_broadcast_type: str = "filesystem") -> None:
        self._engine_rpc = engine_rpc
        # Determines which engine method is called per step: "update_weights_from_distributed"
        # for NCCL (trainer broadcasts; worker just needs to receive) vs
        # "update_weights_from_disk" for filesystem. Set externally by the orchestrator
        # once weight_broadcast config is resolved. Defaults to filesystem (run #35 behaviour).
        self._weight_broadcast_type = weight_broadcast_type

    async def health(self, client: AsyncClient) -> None:
        await client.get("/health")

    async def _post_engine(
        self,
        client: AsyncClient,
        method: str,
        body: dict | None = None,
        *,
        timeout: httpx.Timeout | None = None,
    ) -> dict:
        response = await client.post(f"/engine/{method}", json=body or {}, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(data.get("message", f"Dynamo /engine/{method} failed"))
        return data

    async def pause(self, client: AsyncClient) -> None:
        await self._post_engine(client, "pause_generation", {"mode": "keep", "clear_cache": False})

    async def resume(self, client: AsyncClient) -> None:
        await self._post_engine(client, "resume_generation")

    async def update_weights(self, client: AsyncClient, weight_dir: str | None) -> None:
        if weight_dir is None:
            return
        if self._weight_broadcast_type == "nccl":
            # NCCL path: trainer has already broadcast weights via the NCCL group;
            # this RPC tells the inference worker to call receive_state_dict().
            # NCCLWeightUpdateWorker exposes "update_weights_from_path", not "reload_weights".
            await self._post_engine(
                client,
                "update_weights_from_distributed",
                {
                    "weight_version": Path(weight_dir).name,
                    "weight_dir": weight_dir,
                    "engine_rpc": "update_weights_from_path",
                },
                timeout=httpx.Timeout(180.0),
            )
        else:
            # Resolve to absolute path so the inference worker (which may run in a
            # different working directory) can find the checkpoint on the shared NFS.
            abs_path = str(Path(weight_dir).resolve())
            await self._post_engine(
                client,
                "update_weights_from_disk",
                {
                    "model_path": abs_path,
                    "weight_version": Path(weight_dir).name,
                    "engine_rpc": self._engine_rpc,
                },
                timeout=httpx.Timeout(180.0),
            )

    async def load_lora_adapter(
        self,
        client: AsyncClient,
        lora_name: str,
        lora_path: str,
        *,
        timeout: httpx.Timeout,
    ) -> None:
        await self._post_engine(
            client,
            "load_lora",
            {
                "lora_name": lora_name,
                "source": {"uri": Path(lora_path).absolute().as_uri()},
            },
            timeout=timeout,
        )

    async def init_broadcaster(
        self,
        client: AsyncClient,
        *,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        timeout: int,
        quantize_in_weight_transfer: bool,
    ) -> None:
        await self._post_engine(
            client,
            "init_weights_update_group",
            {
                "host": host,
                "port": port,
                "rank_offset": rank_offset,
                "inference_world_size": inference_world_size,
                "timeout": timeout,
                "quantize_in_weight_transfer": quantize_in_weight_transfer,
                "engine_rpc": "init_broadcaster",
            },
        )


def setup_admin_api(client_config: ClientConfig) -> AdminAPI:
    """Pick the AdminAPI implementation that matches ``client_config.backend``."""
    if client_config.backend == "dynamo":
        return DynamoAdminAPI()
    return VLLMAdminAPI()


_DEFAULT_ADMIN: AdminAPI = VLLMAdminAPI()


@runtime_checkable
class InferencePool(Protocol):
    """Protocol for inference pools (static or elastic)."""

    @property
    def model_name(self) -> str:
        """Get current model name for inference requests."""
        ...

    @property
    def train_clients(self) -> list[vf.ClientConfig]:
        """Get inference clients."""
        ...

    @property
    def admin_clients(self) -> list[AsyncClient]:
        """Get admin clients."""
        ...

    def update_model_name(self, model_name: str) -> None:
        """Update the model name."""
        ...

    async def get_eval_client(self) -> vf.ClientConfig:
        """Get next eval client in round-robin fashion."""
        ...

    async def select_train_client(self, load: Mapping[ClientIdentity, int]) -> vf.ClientConfig:
        """Pick the train client with lowest in-flight load.

        Waits for at least one train client to be available, then returns
        the one with the smallest ``load[client_identity(client)]``. The
        caller owns the in-flight counter; the pool just picks against it.
        """
        ...

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        """Wait for inference pool to be ready."""
        ...

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        """Update weights on all inference servers."""
        ...

    def get_metrics(self) -> dict[str, float]:
        """Get pool metrics."""
        ...

    async def stop(self) -> None:
        """Stop the inference pool."""
        ...


class StaticInferencePool:
    """Static inference pool with fixed client list."""

    def __init__(
        self,
        client_config: ClientConfig,
        model_name: str,
        train_client_type: str = "openai_chat_completions",
        eval_client_type: str = "openai_chat_completions",
        renderer_config: RendererConfig | None = None,
        pool_size: int | None = None,
    ):
        renderer_model_name = model_name if train_client_type == "renderer" else None
        self._train_clients = setup_clients(
            client_config,
            client_type=train_client_type,
            renderer_config=renderer_config,
            renderer_model_name=renderer_model_name,
            pool_size=pool_size,
        )
        self._eval_clients = setup_clients(client_config, client_type=eval_client_type)
        self._admin_clients = setup_admin_clients(client_config)
        self._model_clients = (
            setup_admin_clients(client_config, use_admin_base_url=False)
            if client_config.backend == "dynamo" or client_config.admin_base_url
            else self._admin_clients
        )
        self._admin_api = setup_admin_api(client_config)
        self._skip_model_check = client_config.skip_model_check
        self._wait_for_ready_timeout = client_config.wait_for_ready_timeout
        self._eval_cycle = cycle(self._eval_clients)
        self.model_name = model_name

    @property
    def train_clients(self) -> list[vf.ClientConfig]:
        return self._train_clients

    @property
    def admin_clients(self) -> list[AsyncClient]:
        return self._admin_clients

    def update_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def eval_clients(self) -> list[vf.ClientConfig]:
        return self._eval_clients

    async def get_eval_client(self) -> vf.ClientConfig:
        return next(self._eval_cycle)

    async def select_train_client(self, load: Mapping[ClientIdentity, int]) -> vf.ClientConfig:
        while not self.train_clients:
            await asyncio.sleep(0.5)
        return min(self.train_clients, key=lambda c: load[client_identity(c)])

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        await check_health(
            self._admin_clients,
            timeout=timeout if timeout is not None else self._wait_for_ready_timeout,
            admin=self._admin_api,
        )
        await maybe_check_has_model(
            self._model_clients, model_name, skip_model_check=self._skip_model_check, admin=self._admin_api
        )

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        await update_weights(self._admin_clients, weight_dir, lora_name=lora_name, step=step, admin=self._admin_api)

    def get_metrics(self) -> dict[str, float]:
        return {}

    async def stop(self) -> None:
        pass


async def setup_inference_pool(
    client_config: ClientConfig,
    model_name: str,
    train_client_type: str = "openai_chat_completions",
    eval_client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    pool_size: int | None = None,
) -> InferencePool:
    """Create an inference pool from config (static or elastic)."""
    if client_config.is_elastic:
        from prime_rl.utils.elastic import ElasticInferencePool

        return await ElasticInferencePool.from_config(
            client_config,
            model_name=model_name,
            train_client_type=train_client_type,
            eval_client_type=eval_client_type,
            renderer_config=renderer_config,
            pool_size=pool_size,
        )

    return StaticInferencePool(
        client_config,
        model_name=model_name,
        train_client_type=train_client_type,
        eval_client_type=eval_client_type,
        renderer_config=renderer_config,
        pool_size=pool_size,
    )


def setup_clients(
    client_config: ClientConfig,
    client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    renderer_model_name: str | None = None,
    pool_size: int | None = None,
) -> list[vf.ClientConfig]:
    # Pick the verifiers wire-shape selector based on client_config.backend.
    # When backend == "dynamo", both RendererClient and
    # OpenAIChatCompletionsTokenClient route through Dynamo's nvext path:
    #   - request:  nvext.token_data carries pre-tokenized prompt
    #   - response: nvext.engine_data carries completion_token_ids + logprobs
    # Default backend keeps the legacy vLLM TITO surface.
    renderer_transport = "dynamo_chat_nvext" if client_config.backend == "dynamo" else "prime_vllm_generate"
    clients = []
    client_idx = 0
    # Only forward the renderer config when the client actually uses a
    # renderer — MITO/TITO clients ignore it.
    renderer_extra: dict = {}
    if client_type == "renderer":
        renderer_extra = {
            "renderer_config": renderer_config,
            "renderer_model_name": renderer_model_name,
            "renderer_pool_size": pool_size,
        }
    env_headers = {
        k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
    }
    for base_url in client_config.base_url:
        for dp_rank in range(client_config.dp_rank_count):
            headers = {**client_config.headers, **env_headers}
            if client_config.dp_rank_count > 1:
                headers["X-data-parallel-rank"] = str(dp_rank)
            clients.append(
                vf.ClientConfig(
                    client_idx=client_idx,
                    client_type=client_type,
                    # Dynamo backend routes both renderer and token clients through
                    # the nvext path; default backend keeps the legacy vLLM TITO surface.
                    renderer_transport=renderer_transport,
                    api_base_url=base_url,
                    api_key_var=client_config.api_key_var,
                    timeout=client_config.timeout,
                    connect_timeout=client_config.connect_timeout,
                    max_connections=8192,
                    max_keepalive_connections=8192,
                    max_retries=10,
                    extra_headers=headers,
                    extra_headers_from_state=client_config.extra_headers_from_state,
                    **renderer_extra,
                )
            )
            client_idx += 1
    return clients


def setup_admin_clients(client_config: ClientConfig, *, use_admin_base_url: bool = True) -> list[AsyncClient]:
    """Create dedicated admin clients for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    When admin_base_url is set and use_admin_base_url is true, uses those URLs
    instead of base_url, allowing weight updates to bypass routers in
    disaggregated P/D deployments. For Dynamo, if admin_base_url is unset,
    discover worker-advertised system URLs from GET /v1/rl/workers.
    """
    if use_admin_base_url and client_config.admin_base_url:
        urls = client_config.admin_base_url
    elif use_admin_base_url and client_config.backend == "dynamo":
        urls = discover_dynamo_admin_base_urls(client_config)
    else:
        urls = client_config.base_url

    def _setup_admin_client(base_url: str) -> httpx.AsyncClient:
        env_headers = {
            k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
        }
        headers = {**client_config.headers, **env_headers}
        api_key = os.getenv(client_config.api_key_var, "EMPTY")
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        # Strip /v1 suffix since admin endpoints are at root level
        base_url = base_url.rstrip("/").removesuffix("/v1")

        return AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(None),
        )

    return [_setup_admin_client(base_url) for base_url in urls]


def discover_dynamo_admin_base_urls(client_config: ClientConfig) -> list[str]:
    urls: list[str] = []
    headers = client_config.headers.copy()
    api_key = os.getenv(client_config.api_key_var, "EMPTY")
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    for base_url in _dynamo_rl_discovery_base_urls(client_config):
        discovery_base = base_url.rstrip("/").removesuffix("/v1")
        with httpx.Client(
            base_url=discovery_base,
            headers=headers,
            timeout=httpx.Timeout(connect=client_config.connect_timeout, read=30.0, write=30.0, pool=10.0),
        ) as client:
            response = client.get("/v1/rl/workers")
            response.raise_for_status()
            for worker in response.json().get("workers", []):
                system_url = worker.get("system_url")
                if system_url:
                    urls.append(system_url)

    deduped = list(dict.fromkeys(urls))
    if not deduped:
        raise ValueError(
            "Dynamo backend did not discover any worker system URLs from /v1/rl/workers. "
            "Set client.admin_base_url explicitly, set client.rl_base_url to the Dynamo "
            "RL discovery listener, and make sure Dynamo workers run with DYN_ENABLE_RL "
            "and a system status server enabled."
        )
    return deduped


def _dynamo_rl_discovery_base_urls(client_config: ClientConfig) -> list[str]:
    configured = getattr(client_config, "rl_base_url", None)
    if configured:
        return configured

    rl_port = int(os.getenv("DYN_RL_PORT", "8001"))
    return [_replace_url_port(base_url, rl_port) for base_url in client_config.base_url]


def _replace_url_port(base_url: str, port: int) -> str:
    parsed = urlsplit(base_url.rstrip("/").removesuffix("/v1"))
    scheme = parsed.scheme or "http"
    host = parsed.hostname or parsed.netloc
    if not host:
        raise ValueError(f"Cannot derive Dynamo RL discovery URL from base_url={base_url!r}")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}"
    return urlunsplit((scheme, netloc, "", "", ""))


async def maybe_check_has_model(
    admin_clients: list[AsyncClient],
    model_name: str,
    skip_model_check: bool = False,
    *,
    admin: AdminAPI = _DEFAULT_ADMIN,
) -> None:
    if skip_model_check:
        return
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(*[admin.list_models(admin_client) for admin_client in admin_clients])
    for admin_client, models in zip(admin_clients, results):
        if not any(model["id"] == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {admin_client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def check_health(
    admin_clients: list[AsyncClient],
    interval: int = 1,
    log_interval: int = 10,
    timeout: int = 1800,
    *,
    admin: AdminAPI = _DEFAULT_ADMIN,
) -> None:
    logger = get_logger()

    async def _check_health(admin_client: AsyncClient) -> None:
        wait_time = 0
        logger.debug("Starting pinging /health to check health")
        while wait_time < timeout:
            try:
                await admin.health(admin_client)
                logger.debug(f"Inference pool is ready after {wait_time} seconds")
                return
            except NotFoundError:
                logger.warning("The route /health does not exist. Skipping health check.")
                return
            except Exception as e:
                if wait_time % log_interval == 0 and wait_time > 0:
                    logger.warning(
                        f"Inference server was not reached after {wait_time} seconds (Error: {e}) on {admin_client.base_url}"
                    )
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await asyncio.gather(*[_check_health(admin_client) for admin_client in admin_clients])


NCCL_READY_MARKER = "NCCL_READY"


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | None,
    lora_name: str | None = None,
    step: int = 0,
    *,
    admin: AdminAPI = _DEFAULT_ADMIN,
) -> None:
    """Update weights on static inference servers.

    Pauses all engines to drain in-flight requests, performs the weight update,
    then resumes. Ensures all DP workers are idle and can participate in the
    collective weight transfer. The server-side ``/update_weights`` endpoint
    resets the prefix cache to invalidate any KV states computed with the old
    weights.
    """
    logger = get_logger()

    if lora_name is not None and weight_dir is not None:
        await load_lora_adapter(admin_clients, lora_name, weight_dir, admin=admin)
        return

    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

    logger.info("Pausing inference engines for weight update")
    await asyncio.gather(*[admin.pause(c) for c in admin_clients])
    try:
        # NCCL_READY marker is created before servers enter the receive path
        if weight_dir is not None:
            nccl_ready_file = weight_dir / NCCL_READY_MARKER
            nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
            nccl_ready_file.touch()
            logger.debug(f"Created NCCL_READY marker at {nccl_ready_file}")

        await asyncio.gather(*[admin.update_weights(c, weight_dir_posix) for c in admin_clients])
    finally:
        await asyncio.gather(*[admin.resume(c) for c in admin_clients])
        logger.info("Inference engines resumed")


def _is_retryable_lora_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for LoRA loading."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 404 (adapter not found) or 500 (server error during loading)
        return exception.response.status_code in (404, 500)
    # Retry on transport-level failures (timeouts, connection resets, etc.) so
    # the per-call read timeout below turns a stuck server into a bounded retry
    # loop instead of propagating as a hard failure on the first hiccup.
    if isinstance(exception, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# Per-attempt and total bounds for `/load_lora_adapter`. A LoRA load is fast
# (small adapter file + KV cache reset, single-digit seconds in practice) but
# the global admin AsyncClient uses `timeout=None`, so a stuck server hangs
# the orchestrator forever inside `ElasticInferencePool._sync_server_adapter`.
# `_PER_ATTEMPT` converts a hang into a TimeoutException so tenacity retries;
# `_TOTAL` is the wall-clock budget across all retries — pick whichever
# stop condition fires first.
LORA_LOAD_READ_TIMEOUT_S = 30.0
LORA_LOAD_TOTAL_TIMEOUT_S = 120.0


async def load_lora_adapter(
    admin_clients: list[AsyncClient],
    lora_name: str,
    lora_path: Path,
    *,
    admin: AdminAPI = _DEFAULT_ADMIN,
) -> None:
    """Make a HTTP post request to the vLLM server to load a LoRA adapter.

    Uses our wrapper endpoint that also resets the prefix cache to invalidate
    KV states computed with old weights.

    Retries with exponential backoff if the adapter files are not found,
    which can happen due to NFS propagation delays.
    """
    logger = get_logger()
    lora_path_posix = lora_path.as_posix()
    per_attempt_timeout = httpx.Timeout(connect=10.0, read=LORA_LOAD_READ_TIMEOUT_S, write=60.0, pool=10.0)

    @retry(
        retry=retry_if_exception(_is_retryable_lora_error),
        stop=stop_after_delay(LORA_LOAD_TOTAL_TIMEOUT_S) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _load_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to load LoRA adapter {lora_name} from {lora_path}")
        await admin.load_lora_adapter(admin_client, lora_name, lora_path_posix, timeout=per_attempt_timeout)

    await asyncio.gather(*[_load_lora_adapter(admin_client) for admin_client in admin_clients])


async def init_nccl_broadcast(
    admin_clients: list[AsyncClient],
    host: str,
    port: int,
    timeout: int,
    inference_world_size: int | None = None,
    quantize_in_weight_transfer: bool = False,
    *,
    admin: AdminAPI = _DEFAULT_ADMIN,
) -> None:
    """Initialize NCCL broadcast on all inference servers.

    Each admin client represents one vLLM server. The function computes
    per-server rank_offset and gpus_per_server so that every inference GPU
    gets a unique rank in the NCCL broadcast group.
    """
    logger = get_logger()

    if inference_world_size is None:
        inference_world_size = len(admin_clients)
        logger.warning(
            f"inference_world_size not provided, defaulting to {inference_world_size} (one GPU per admin client)"
        )

    gpus_per_server = inference_world_size // len(admin_clients)

    logger.info(
        f"Initializing NCCL broadcast: {len(admin_clients)} servers, "
        f"inference_world_size={inference_world_size}, gpus_per_server={gpus_per_server}"
    )

    async def _init_nccl_broadcast(admin_client: AsyncClient, rank_offset: int) -> None:
        try:
            await admin.init_broadcaster(
                admin_client,
                host=host,
                port=port,
                rank_offset=rank_offset,
                inference_world_size=inference_world_size,
                timeout=timeout,
                quantize_in_weight_transfer=quantize_in_weight_transfer,
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization.")
                return

    await asyncio.gather(
        *[
            _init_nccl_broadcast(admin_client, client_num * gpus_per_server)
            for client_num, admin_client in enumerate(admin_clients)
        ]
    )
