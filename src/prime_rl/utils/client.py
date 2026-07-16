from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
import verifiers.v1 as vf
from httpx import AsyncClient
from openai import AsyncOpenAI, NotFoundError
from renderers import RendererConfig
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)
from verifiers.v1.clients.config import EvalClientConfig, TrainClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger

# Identity tuple used by ``select_train_client`` to key load counts. ``base_url``
# distinguishes servers; ``X-data-parallel-rank`` distinguishes DP shards within a
# server, since the router uses that header to route to specific GPU ranks.
ClientIdentity = tuple[str, str | None]


def client_identity(client: vf.ClientConfig) -> ClientIdentity:
    """Stable identity for load balancing across inference clients."""
    return (client.base_url, client.headers.get("X-data-parallel-rank"))


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

    @property
    def admin_world_sizes(self) -> list[int] | None:
        """Per-admin-client inference world sizes, when discovered."""
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

    async def score(self, token_ids: list[int]) -> list[float]:
        """Prefill-score ``token_ids`` under the pool's model — one logprob per token."""
        ...

    async def stop(self) -> None:
        """Stop the inference pool."""
        ...


class PrefillScorer:
    """Prefill-scores token ids against a pool's *current* endpoints. Resolves one
    client per endpoint, cached by endpoint identity — so it fills once for a
    static pool and tolerates churn for an elastic one (a departed endpoint is
    simply never selected again; its client is closed at stop). Round-robins over
    the live endpoints."""

    def __init__(self) -> None:
        self._clients: dict = {}  # client_identity -> AsyncOpenAI, one per endpoint
        self._rr = 0

    async def score(self, configs: list[vf.ClientConfig], model: str, token_ids: list[int]) -> list[float]:
        if not configs:
            raise RuntimeError("no inference endpoints available to prefill-score")
        cfg = configs[self._rr % len(configs)]
        self._rr += 1
        key = client_identity(cfg)
        openai = self._clients.get(key)
        if openai is None:
            # Build the OpenAI client straight from the config fields — works for any
            # ClientConfig type; resolve_client would hand back an EvalClient (no `.openai`)
            # for these chat-completions teacher configs.
            openai = self._clients[key] = AsyncOpenAI(
                base_url=cfg.base_url,
                api_key=os.environ.get(cfg.api_key_var) or "EMPTY",
                default_headers=cfg.headers or None,
            )
        return await prefill_logprobs(openai, model, token_ids)

    async def aclose(self) -> None:
        await asyncio.gather(*(c.close() for c in self._clients.values()))


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
        self._skip_model_check = client_config.skip_model_check
        self._wait_for_ready_timeout = client_config.wait_for_ready_timeout
        self._eval_cycle = cycle(self._eval_clients)
        self._scorer = PrefillScorer()
        self.model_name = model_name

    @property
    def train_clients(self) -> list[vf.ClientConfig]:
        return self._train_clients

    @property
    def admin_clients(self) -> list[AsyncClient]:
        return self._admin_clients

    @property
    def admin_world_sizes(self) -> list[int] | None:
        return None

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
            self._admin_clients, timeout=timeout if timeout is not None else self._wait_for_ready_timeout
        )
        await maybe_check_has_model(self._admin_clients, model_name, skip_model_check=self._skip_model_check)

    async def update_weights(self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0) -> None:
        await update_weights(self._admin_clients, weight_dir, lora_name=lora_name, step=step)

    async def score(self, token_ids: list[int]) -> list[float]:
        """Prefill-score ``token_ids`` under this pool's model (one logprob per
        token, 0.0 for the leading token). Delegates to the shared scorer."""
        return await self._scorer.score(self.train_clients, self.model_name, token_ids)

    async def stop(self) -> None:
        await self._scorer.aclose()


@dataclass(frozen=True)
class DiscoveredDynamoWorker:
    component: str
    instance_id: int
    admin_base_url: str
    world_size: int


class DynamoDiscoveryPending(ValueError):
    """A well-formed discovery snapshot that is not ready yet."""


def _parse_dynamo_workers(payload: object, model_name: str) -> tuple[DiscoveredDynamoWorker, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("workers"), list):
        raise ValueError("Dynamo RL discovery response must contain a workers list")

    discovered: list[DiscoveredDynamoWorker] = []
    seen_admin_urls: set[str] = set()
    seen_worker_ids: set[tuple[str, int]] = set()
    for raw_worker in payload["workers"]:
        if not isinstance(raw_worker, dict):
            raise ValueError("Dynamo RL discovery returned a malformed worker")
        if raw_worker.get("error"):
            raise DynamoDiscoveryPending(f"Dynamo RL worker probe is not ready: {raw_worker['error']}")
        if raw_worker.get("model") != model_name:
            raise ValueError(
                f"Dynamo RL worker model {raw_worker.get('model')!r} does not match {model_name!r}"
            )

        component = raw_worker.get("component")
        instance_id = raw_worker.get("instance_id")
        admin_base_url = raw_worker.get("admin_base_url")
        world_size = raw_worker.get("world_size")
        if not isinstance(component, str) or not component:
            raise ValueError("Dynamo RL worker is missing component identity")
        if not isinstance(instance_id, int) or isinstance(instance_id, bool) or instance_id < 0:
            raise ValueError("Dynamo RL worker has an invalid instance_id")
        if not isinstance(admin_base_url, str):
            raise ValueError("Dynamo RL worker is missing a valid admin_base_url")
        try:
            parsed_admin_url = httpx.URL(admin_base_url)
        except httpx.InvalidURL as error:
            raise ValueError("Dynamo RL worker has an invalid admin_base_url") from error
        if (
            parsed_admin_url.scheme not in {"http", "https"}
            or parsed_admin_url.host is None
            or parsed_admin_url.username
            or parsed_admin_url.password
            or parsed_admin_url.path not in {"", "/"}
            or parsed_admin_url.query
            or parsed_admin_url.fragment
        ):
            raise ValueError("Dynamo RL worker has an invalid admin_base_url")
        admin_base_url = admin_base_url.rstrip("/")
        worker_id = (component, instance_id)
        if worker_id in seen_worker_ids:
            raise ValueError(f"Dynamo RL discovery returned duplicate worker identity {worker_id}")
        if admin_base_url in seen_admin_urls:
            raise ValueError(f"Dynamo RL discovery returned duplicate admin endpoint {admin_base_url}")
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
            raise ValueError("Dynamo RL worker has an invalid world_size")

        seen_admin_urls.add(admin_base_url)
        seen_worker_ids.add(worker_id)
        discovered.append(
            DiscoveredDynamoWorker(
                component=component,
                instance_id=instance_id,
                admin_base_url=admin_base_url,
                world_size=world_size,
            )
        )

    if not discovered:
        raise DynamoDiscoveryPending("Dynamo RL discovery returned no workers yet")
    return tuple(sorted(discovered, key=lambda worker: (worker.component, worker.instance_id)))


class DynamoInferencePool(StaticInferencePool):
    """Static request pool whose direct vLLM admin clients come from Dynamo discovery."""

    def __init__(self, client_config: ClientConfig, workers: tuple[DiscoveredDynamoWorker, ...], **kwargs):
        discovered_config = client_config.model_copy(
            update={"admin_base_url": [worker.admin_base_url for worker in workers]}
        )
        super().__init__(discovered_config, **kwargs)
        self._admin_world_sizes = [worker.world_size for worker in workers]

    @property
    def admin_world_sizes(self) -> list[int]:
        return list(self._admin_world_sizes)

    @classmethod
    async def from_config(cls, client_config: ClientConfig, model_name: str, **kwargs) -> DynamoInferencePool:
        if client_config.rl_base_url is None:
            raise ValueError("Dynamo inference pool requires rl_base_url")
        discovery_url = client_config.rl_base_url.rstrip("/").removesuffix("/v1")
        async with AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            workers = None
            async for attempt in AsyncRetrying(
                stop=stop_after_delay(client_config.wait_for_ready_timeout),
                wait=wait_exponential(multiplier=0.1, min=0.1, max=1),
                retry=retry_if_exception_type((DynamoDiscoveryPending, httpx.TransportError)),
                reraise=True,
            ):
                with attempt:
                    response = await client.get(f"{discovery_url}/v1/rl/workers")
                    response.raise_for_status()
                    workers = _parse_dynamo_workers(response.json(), model_name)
        assert workers is not None
        return cls(client_config, workers, model_name=model_name, **kwargs)


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

    if client_config.rl_base_url is not None and client_config.admin_base_url is None:
        return await DynamoInferencePool.from_config(
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
    """Build v1 client configs (one per base_url × DP rank). ``client_type``
    ``renderer`` → token-in/out (``TrainClientConfig``, with the renderer the env
    server should use forwarded as a serialized config so it doesn't fall back to the
    default renderer); otherwise plain chat-completions (``EvalClientConfig``)."""
    is_renderer = client_type == "renderer"
    config_cls = TrainClientConfig if is_renderer else EvalClientConfig
    renderer_extra: dict = {}
    if is_renderer:
        renderer_extra = {
            "renderer": renderer_config,
            "pool_size": pool_size or 1,
            "renderer_model_name": renderer_model_name,
        }
    env_headers = {
        k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
    }
    clients: list[vf.ClientConfig] = []
    for base_url in client_config.base_url:
        for dp_rank in range(client_config.dp_rank_count):
            headers = {**client_config.headers, **env_headers}
            if client_config.dp_rank_count > 1:
                headers["X-data-parallel-rank"] = str(dp_rank)
            clients.append(
                config_cls(base_url=base_url, api_key_var=client_config.api_key_var, headers=headers, **renderer_extra)
            )
    return clients


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    """Create dedicated admin clients for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    When admin_base_url is set, uses those URLs instead of base_url, allowing
    weight updates to bypass routers in disaggregated P/D deployments.
    """
    urls = client_config.admin_base_url if client_config.admin_base_url else client_config.base_url

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


async def maybe_check_has_model(
    admin_clients: list[AsyncClient], model_name: str, skip_model_check: bool = False
) -> None:
    if skip_model_check:
        return
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(*[admin_client.get("/v1/models") for admin_client in admin_clients])
    for admin_client, result in zip(admin_clients, results):
        models = result.json()["data"]
        if not any(model["id"] == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {admin_client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def check_health(
    admin_clients: list[AsyncClient], interval: int = 1, log_interval: int = 10, timeout: int = 1800
) -> None:
    logger = get_logger()

    async def _check_health(admin_client: AsyncClient) -> None:
        wait_time = 0
        logger.debug("Starting pinging /health to check health")
        while wait_time < timeout:
            try:
                await admin_client.get("/health")
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


def _is_retryable_admin_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for an admin op (pause/resume/update_weights)."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on transient server errors (5xx, e.g. engine briefly unresponsive);
        # client errors (4xx) won't fix themselves on retry.
        return exception.response.status_code >= 500
    # Retry on transport-level failures (timeouts, connection resets, etc.) so the
    # per-attempt read timeout below turns a stuck server into a bounded retry loop
    # instead of hanging forever on the global timeout=None admin client.
    if isinstance(exception, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# Per-attempt read timeout for admin ops, overridable per call. The admin
# AsyncClient uses `timeout=None`, so without this a stuck server would hang the
# weight update forever: the read timeout converts a hang into a TimeoutException
# that tenacity retries. Sized for `/pause`, which drains in-flight requests
# (mode="keep") and so can legitimately take a while.
ADMIN_TIMEOUT_S = 300.0
# `/update_weights` runs a collective NCCL receive across all DP workers, which
# can take longer than the other admin ops.
UPDATE_WEIGHTS_TIMEOUT_S = 720.0


async def _admin_post(client: AsyncClient, path: str, *, timeout_s: float = ADMIN_TIMEOUT_S, **kwargs) -> None:
    """POST an admin op with a bounded per-attempt timeout, retrying transient errors.

    The total wall-clock budget across all retries is twice the per-attempt timeout.
    """
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_is_retryable_admin_error),
        stop=stop_after_delay(2 * timeout_s) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    ):
        with attempt:
            response = await client.post(
                path,
                timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=10.0),
                **kwargs,
            )
            response.raise_for_status()


async def _pause_engines(admin_clients: list[AsyncClient], *, step: int) -> None:
    """Pause all inference engines, waiting for in-flight requests to drain."""
    logger = get_logger()
    logger.info(f"Updating policy in-flight to v{step}")
    await asyncio.gather(
        *[_admin_post(client, "/pause", params={"mode": "keep", "clear_cache": "false"}) for client in admin_clients]
    )
    logger.debug("All inference engines paused")


async def _resume_engines(admin_clients: list[AsyncClient]) -> None:
    """Resume all inference engines after weight update.

    Resuming is idempotent (it just clears the paused flag), so retrying transient
    failures is safe; a dropped /resume would leave engines paused indefinitely.
    """
    logger = get_logger()
    await asyncio.gather(*[_admin_post(client, "/resume") for client in admin_clients])
    logger.debug("All inference engines resumed")


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | None,
    lora_name: str | None = None,
    step: int = 0,
) -> None:
    """Update weights on static inference servers.

    Pauses all engines first to drain in-flight requests, then performs the
    weight update, then resumes. This ensures all DP workers are idle and can
    participate in the collective weight transfer.

    Note: the prefix cache is intentionally not reset on weight update. The orchestrator
    salts the prefix cache per weight version (``cache_salt`` in the sampling request, see
    ``orchestrator/envs.py``), so KV computed under old weights is never reused.
    """
    logger = get_logger()

    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

    if lora_name is not None and weight_dir is not None:
        await load_lora_adapter(admin_clients, lora_name, weight_dir)
    else:
        # Pause engines so all DP workers drain in-flight work and can join the NCCL broadcast
        await _pause_engines(admin_clients, step=step)

        try:
            # Create ready marker before servers enter receive path (used by NCCL broadcast)
            if weight_dir is not None:
                nccl_ready_file = weight_dir / NCCL_READY_MARKER
                nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
                nccl_ready_file.touch()
                logger.debug(f"Created NCCL_READY marker at {nccl_ready_file}")

            await asyncio.gather(
                *[
                    _admin_post(
                        admin_client,
                        "/update_weights",
                        json={"weight_dir": weight_dir_posix},
                        timeout_s=UPDATE_WEIGHTS_TIMEOUT_S,
                    )
                    for admin_client in admin_clients
                ]
            )
        finally:
            await _resume_engines(admin_clients)


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


async def load_lora_adapter(admin_clients: list[AsyncClient], lora_name: str, lora_path: Path) -> None:
    """Make a HTTP post request to the vLLM server to load a LoRA adapter.

    Uses our wrapper around vLLM's /v1/load_lora_adapter. The prefix cache is not reset
    here; the orchestrator salts it per weight version (see ``orchestrator/envs.py``) so
    KV computed under old weights is never reused.

    Retries with exponential backoff if the adapter files are not found,
    which can happen due to NFS propagation delays.
    """
    logger = get_logger()
    lora_path_posix = lora_path.as_posix()

    @retry(
        retry=retry_if_exception(_is_retryable_lora_error),
        stop=stop_after_delay(LORA_LOAD_TOTAL_TIMEOUT_S) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _load_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to load LoRA adapter {lora_name} from {lora_path}")
        response = await admin_client.post(
            "/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path_posix},
            timeout=httpx.Timeout(connect=10.0, read=LORA_LOAD_READ_TIMEOUT_S, write=60.0, pool=10.0),
        )
        response.raise_for_status()

    await asyncio.gather(*[_load_lora_adapter(admin_client) for admin_client in admin_clients])


async def unload_lora_adapter(admin_clients: list[AsyncClient], lora_name: str) -> None:
    """Make a HTTP post request to the vLLM server to unload a LoRA adapter."""
    logger = get_logger()

    async def _unload_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to unload LoRA adapter {lora_name}")
        await admin_client.post("/v1/unload_lora_adapter", json={"lora_name": lora_name})
        # TODO: The first one can fail, but subsequent ones should succeed.
        # response.raise_for_status()

    await asyncio.gather(*[_unload_lora_adapter(admin_client) for admin_client in admin_clients])


async def init_nccl_broadcast(
    admin_clients: list[AsyncClient],
    host: str,
    port: int,
    timeout: int,
    inference_world_size: int | None = None,
    engine_world_sizes: list[int] | None = None,
    quantize_in_weight_transfer: bool = False,
) -> None:
    """Initialize NCCL broadcast on all inference servers.

    Each admin client represents one vLLM server. The function computes
    per-server rank_offset and gpus_per_server so that every inference GPU
    gets a unique rank in the NCCL broadcast group.
    """
    logger = get_logger()

    if inference_world_size is None:
        if engine_world_sizes is not None:
            inference_world_size = sum(engine_world_sizes)
            logger.info(f"Using discovered inference_world_size={inference_world_size}")
        else:
            inference_world_size = len(admin_clients)
            logger.warning(
                f"inference_world_size not provided, defaulting to {inference_world_size} "
                "(one GPU per admin client)"
            )

    if engine_world_sizes is None:
        if inference_world_size % len(admin_clients) != 0:
            raise ValueError("inference_world_size must be divisible by the number of admin clients")
        engine_world_sizes = [inference_world_size // len(admin_clients)] * len(admin_clients)
    if len(engine_world_sizes) != len(admin_clients):
        raise ValueError("one engine world size is required for each admin client")
    rank_offsets = _rank_offsets(engine_world_sizes, inference_world_size)

    logger.info(
        f"Initializing NCCL broadcast: {len(admin_clients)} servers, "
        f"inference_world_size={inference_world_size}, engine_world_sizes={engine_world_sizes}"
    )

    async def _init_nccl_broadcast(admin_client: AsyncClient, rank_offset: int) -> None:
        try:
            response = await admin_client.post(
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
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization.")
                return

    await asyncio.gather(
        *[
            _init_nccl_broadcast(admin_client, rank_offset)
            for admin_client, rank_offset in zip(admin_clients, rank_offsets, strict=True)
        ]
    )


def _rank_offsets(engine_world_sizes: list[int], inference_world_size: int) -> list[int]:
    if not engine_world_sizes or any(isinstance(size, bool) or size <= 0 for size in engine_world_sizes):
        raise ValueError("engine world sizes must be positive integers")
    if sum(engine_world_sizes) != inference_world_size:
        raise ValueError("discovered engine world sizes do not match inference_world_size")
    offsets: list[int] = []
    offset = 0
    for world_size in engine_world_sizes:
        offsets.append(offset)
        offset += world_size
    return offsets


async def prefill_logprobs(openai: AsyncOpenAI, model: str, token_ids: list[int]) -> list[float]:
    """Prefill-score ``token_ids`` under ``model`` via ``/inference/v1/generate``
    + ``prompt_logprobs`` (the prime-rl server-side extension in
    ``inference/vllm/serving_tokens.py``). Returns one logprob per token (0.0 for
    the leading token, which has no preceding context)."""
    from vllm.entrypoints.serve.disagg.protocol import GenerateResponse

    # `/inference/v1/generate` is mounted at server root, not under `/v1`: pass an
    # absolute URL so the SDK skips the base-url merge. vLLM's `GenerateResponse`
    # isn't an `openai.BaseModel`, so the SDK parse layer rejects it as `cast_to`;
    # `cast_to=httpx.Response` lets the SDK still build the request (auth, retries,
    # timeouts) and hand back the raw response for us to validate.
    base = str(openai.base_url).rstrip("/").removesuffix("/v1")
    http_response = await openai.post(
        f"{base}/inference/v1/generate",
        cast_to=httpx.Response,
        body={
            "model": model,
            "token_ids": token_ids,
            "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "prompt_logprobs": 1},
        },
    )
    response = GenerateResponse.model_validate_json(http_response.content)
    # `prompt_logprobs[i]` is a `{token_id: Logprob}` dict, or `None` for the
    # leading token (no preceding context). Flatten to `list[float]`.
    flat: list[float] = []
    for entry in response.prompt_logprobs or []:
        if not entry:
            flat.append(0.0)
            continue
        first = next(iter(entry.values()))
        lp = first.logprob if hasattr(first, "logprob") else first.get("logprob")
        flat.append(float(lp) if lp is not None else 0.0)
    return flat
