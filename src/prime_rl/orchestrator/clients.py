from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

import httpx
import verifiers.v1 as vf
from httpx import AsyncClient
from openai import AsyncOpenAI
from renderers import RendererConfig
from tenacity import AsyncRetrying, retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential
from verifiers.v1.clients.base import build_async_openai
from verifiers.v1.configs.client import (
    BaseClientConfig,
    EvalClientConfig,
    TrainClientConfig,
    resolve_api_key,
    resolve_headers,
)

from prime_rl.configs.algorithm import FrozenModelConfig
from prime_rl.configs.shared import ClientConfig, PolicyClientConfig
from prime_rl.utils.logger import get_logger


class PrefillScorer:
    """Prefill-scores token ids against a pool's endpoint, lazily resolving
    a single OpenAI client from the pool's train client config."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    async def score(self, config: vf.ClientConfig, model: str, token_ids: list[int]) -> list[float]:
        if self._client is None:
            # Prefill scoring uses the SDK even when generation uses the eval relay.
            self._client = build_async_openai(config)
        return await prefill_logprobs(self._client, model, token_ids)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


# How long the policy deployment may have no engine serving the current weights. A restarted
# engine gets them from the trainer's next weight update, which needs a batch of rollouts.
NO_SERVICE_TIMEOUT_S = 1800

# Concurrent session releases per endpoint. The release client has one connection per slot, so
# requests never wait in httpx's pool, where a pool timeout can leak connections for good
# (encode/httpcore#1099). A release that waits longer than the queue timeout for a slot is
# dropped, which bounds how long a slow router can delay episode completion.
SESSION_RELEASE_CONCURRENCY = 32
SESSION_RELEASE_QUEUE_TIMEOUT_S = 30.0


class InferenceClient:
    """Data-plane clients for one inference endpoint (the router of the policy
    deployment, or an external API for frozen models)."""

    def __init__(
        self,
        client_config: ClientConfig,
        model_name: str,
        train_client_type: str = "openai_chat_completions",
        eval_client_type: str = "openai_chat_completions",
        renderer_config: RendererConfig | None = None,
    ):
        renderer_model_name = model_name if train_client_type == "renderer" else None
        self.train_client = setup_client(
            client_config,
            client_type=train_client_type,
            renderer_config=renderer_config,
            renderer_model_name=renderer_model_name,
        )
        self.eval_client = setup_client(client_config, client_type=eval_client_type)
        self._scorer = PrefillScorer()
        # Managed routed deployments set admin_base_url so engine admin traffic
        # bypasses the client-facing router. External and frozen clients do not.
        self._session_client_config = (
            client_config.model_copy(update={"admin_base_url": None})
            if client_config.admin_base_url is not None
            else None
        )
        self._session_client = (
            _setup_session_client(self._session_client_config) if self._session_client_config is not None else None
        )
        self._session_release_slots = asyncio.Semaphore(SESSION_RELEASE_CONCURRENCY)
        self.sessions_released = 0
        self.session_releases_failed = 0
        self.model_name = model_name

    async def score(self, token_ids: list[int]) -> list[float]:
        """Prefill-score ``token_ids`` under this endpoint's model (one logprob
        per token, 0.0 for the leading token)."""
        return await self._scorer.score(self.train_client, self.model_name, token_ids)

    async def aclose(self) -> None:
        await self._scorer.aclose()
        if self._session_client is not None:
            await self._session_client.aclose()

    async def finish_sessions(self, session_ids: list[str]) -> None:
        """Release completed sessions when the client-facing router supports it."""
        if self._session_client is None or not session_ids:
            return

        async def finish_session(session_id: str) -> Exception | None:
            try:
                await asyncio.wait_for(self._session_release_slots.acquire(), SESSION_RELEASE_QUEUE_TIMEOUT_S)
            except TimeoutError as error:
                return error
            client = self._session_client
            try:
                await _admin_post(
                    client,
                    "/finish_session",
                    timeout_s=5.0,
                    params={"session_id": session_id},
                )
            except httpx.PoolTimeout as error:
                # Releases never outnumber the pool's connections, so a pool timeout means httpcore
                # leaked connections some other way. Only a new client recovers.
                if client is self._session_client:
                    get_logger().warning(f"Replacing the inference session release client after {error!r}")
                    self._session_client = _setup_session_client(self._session_client_config)
                    await client.aclose()
                return error
            except Exception as error:
                return error
            finally:
                self._session_release_slots.release()

        errors = await asyncio.gather(*(finish_session(session_id) for session_id in session_ids))
        failed = [error for error in errors if error is not None]
        failed_before = self.session_releases_failed
        self.sessions_released += len(session_ids) - len(failed)
        self.session_releases_failed += len(failed)
        # Log each time the total doubles, so a burst of failures logs a handful of lines.
        if self.session_releases_failed.bit_length() > failed_before.bit_length():
            get_logger().warning(
                f"{self.session_releases_failed} inference session releases have failed in total; the router "
                f"expires unreleased sessions after its session TTL. Latest error: {failed[0]!r}"
            )

    def session_release_gauges(self) -> dict[str, float]:
        return {
            "sessions/released": float(self.sessions_released),
            "sessions/release_failed": float(self.session_releases_failed),
        }


class AdminPlane:
    """Admin plane of the policy inference deployment: one httpx client per
    engine process. The router serves no admin routes (pause/resume,
    update_weights, init_broadcaster, load_lora_adapter live on the engines),
    so these clients bypass it via ``admin_base_url``.

    The client order is load-bearing: ``admin_base_url`` order must match the
    GPU rank order used by NCCL/NIXL initialization and the metrics collector's
    role list index."""

    roles: list[str | None] | None = None
    """P/D role of each client, in client order. Prefill engines pause before decode engines."""

    # Per-engine times of the last update_weights phases (pause, transfer, resume).
    # Replaced, never mutated.
    update_times: dict[str, list[float]] = {}

    def __init__(
        self, client_config: ClientConfig, roles: list[str | None] | None = None, tolerate_restarts: bool = False
    ):
        self.clients = setup_admin_clients(client_config)
        self.roles = roles
        # When admin URLs bypass a router, also health-check the client-facing
        # (router) endpoint - it only starts serving once its workers are healthy.
        self._router_clients = (
            setup_admin_clients(client_config.model_copy(update={"admin_base_url": None}))
            if client_config.admin_base_url
            else []
        )
        self._skip_model_check = client_config.skip_model_check
        self._wait_for_ready_timeout = client_config.wait_for_ready_timeout
        self._init_membership(tolerate_restarts)

    def _init_membership(self, tolerate_restarts: bool) -> None:
        # Serializes weight updates and changes to which engines are up
        self._update_lock = asyncio.Lock()
        self.tolerate_restarts = tolerate_restarts
        """Mark an engine that stops answering down instead of failing the admin op (see ``watch``)."""
        self.down: set[int] = set()
        """Engines that stopped answering. They get no admin ops and no traffic until they answer again."""
        self.rejoining: set[int] = set()
        """Restarted engines. They take traffic once a weight update has reached them."""
        self.on_rejoin: Callable[[list[int]], Awaitable[None]] | None = None
        """Prepares restarted engines to receive weights; set by the weight transport."""
        self.on_down: Callable[[int, int], None] | None = None
        """Called with the number of engines up before and after one goes down."""
        self._misses = [0] * len(self.clients)

    def up(self) -> list[int]:
        """Indices of the engines that answer."""
        return [index for index in range(len(self.clients)) if index not in self.down]

    async def watch(self, interval: float = 10.0) -> None:
        """Poll every engine's /health. An engine that misses two polls goes down (and leaves the
        router); a down engine that answers again rejoins: the weight transport prepares it, the
        next weight update reaches it, and then it takes traffic again."""
        no_service_since: float | None = None
        while True:
            await asyncio.sleep(interval)
            if self.down | self.rejoining != set(range(len(self.clients))):
                no_service_since = None
            elif no_service_since is None:
                no_service_since = time.monotonic()
            elif time.monotonic() - no_service_since > NO_SERVICE_TIMEOUT_S:
                raise RuntimeError(
                    f"No inference server has served the current weights for {NO_SERVICE_TIMEOUT_S}s: restarted "
                    "servers get them from the trainer's next weight update, which needs rollouts to train on"
                )
            answers = await asyncio.gather(*(_answers(client) for client in self.clients))
            async with self._update_lock:
                answering_again = []
                for index, answer in enumerate(answers):
                    self._misses[index] = 0 if answer else self._misses[index] + 1
                    if index in self.down and answer:
                        answering_again.append(index)
                    elif index not in self.down and self._misses[index] >= 2:
                        await self.mark_down(index, "missed two health checks")
                if answering_again:
                    await self._rejoin(answering_again)

    async def mark_down(self, index: int, reason: object) -> None:
        if index in self.down:
            return
        up = len(self.up())
        self.down.add(index)
        self.rejoining.discard(index)
        get_logger().warning(
            f"Inference server {self.clients[index].base_url} is down ({reason}); it gets no traffic and no weight "
            "updates until it answers again"
        )
        await self._route(index, "remove")
        if self.on_down is not None:
            self.on_down(up, up - 1)

    async def _rejoin(self, indices: list[int]) -> None:
        urls = ", ".join(str(self.clients[index].base_url) for index in indices)
        if self.on_rejoin is not None:
            try:
                await self.on_rejoin(indices)
            except Exception as error:
                get_logger().warning(f"Restarted inference servers {urls} are not ready to rejoin yet: {error!r}")
                return
        self.down.difference_update(indices)
        self.rejoining.update(indices)
        get_logger().info(f"Inference servers {urls} answer again; they take traffic after the next weight update")

    async def _admit(self, targets: list[int]) -> None:
        """Route traffic to the rejoining engines the last weight update reached."""
        for index in sorted(self.rejoining.intersection(targets) - self.down):
            self.rejoining.discard(index)
            await self._route(index, "add")
            get_logger().info(f"Inference server {self.clients[index].base_url} has the current weights and rejoined")

    async def _route(self, index: int, action: Literal["add", "remove"]) -> None:
        if not self._router_clients:
            return
        url = str(self.clients[index].base_url).rstrip("/")
        try:
            response = await self._router_clients[0].post(f"/{action}_worker", params={"url": url}, timeout=60)
            response.raise_for_status()
        except httpx.HTTPError as error:
            get_logger().warning(f"Could not {action} {url} on the router: {error!r}")

    async def _post(self, targets: list[int], path: str, *, retry: bool = True, **kwargs) -> list[float]:
        """POST ``path`` to the ``targets`` engines concurrently; returns each answering engine's
        request time. With ``tolerate_restarts``, an engine that does not answer goes down instead
        of failing the op."""

        async def post(index: int) -> float:
            start = time.perf_counter()
            await _admin_post(
                self.clients[index],
                path,
                attempts=10 if retry else 1,
                retry_connect=not self.tolerate_restarts,
                **kwargs,
            )
            return time.perf_counter() - start

        results = await asyncio.gather(*(post(index) for index in targets), return_exceptions=True)
        times = []
        for index, result in zip(targets, results, strict=True):
            if isinstance(result, BaseException):
                if self.tolerate_restarts and isinstance(result, httpx.TransportError):
                    await self.mark_down(index, repr(result))
                    continue
                raise result
            times.append(result)
        return times

    async def _pause(self, targets: list[int], step: int) -> list[float]:
        """Pause the engines, waiting for in-flight requests to drain. Returns each engine's pause time.

        Prefill engines pause first. A prefill engine's pause waits until decode has pulled the KV
        blocks it still holds, and a paused decode engine never pulls them: the prefill would wait
        for vLLM's NIXL KV lease to expire (30 s by default) and decode would later read the
        released blocks."""
        get_logger().debug(f"Pausing inference engines to update weights to policy v{step}")
        roles = self.roles or [None] * len(self.clients)
        prefill = [index for index in targets if roles[index] == "prefill"]
        rest = [index for index in targets if roles[index] != "prefill"]
        pause_times: list[float] = []
        for indices in (prefill, rest):
            pause_times += await self._post(indices, "/pause", params={"mode": "keep", "clear_cache": "false"})
        get_logger().debug("Inference engines paused")
        return pause_times

    async def _resume(self, targets: list[int]) -> list[float]:
        """Resume the engines (idempotent, so retried on transient failures; a dropped /resume would
        leave an engine paused indefinitely). Returns each engine's resume time."""
        resume_times = await self._post([index for index in targets if index not in self.down], "/resume")
        get_logger().debug("Inference engines resumed")
        return resume_times

    async def wait_for_ready(self, model_name: str) -> None:
        # The engines are waited on even when a router fronts them: the llm-d
        # router (Envoy) 404s /health, which check_health treats as "no health
        # route", so the router alone is not a readiness signal. The router
        # owns the info-level waiting log; the per-engine waits log at debug.
        await asyncio.gather(
            check_health(self.clients, timeout=self._wait_for_ready_timeout, quiet=bool(self._router_clients)),
            check_health(self._router_clients, timeout=self._wait_for_ready_timeout),
        )
        await maybe_check_has_model(self.clients, model_name, skip_model_check=self._skip_model_check)

    async def initialize_nccl(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
    ) -> None:
        gpus_per_server = inference_world_size // len(self.clients)
        get_logger().info(
            f"Initializing NCCL broadcast: {len(self.clients)} servers, "
            f"inference_world_size={inference_world_size}, gpus_per_server={gpus_per_server}"
        )

        async def initialize_client(admin_client: AsyncClient, rank_offset: int) -> None:
            try:
                response = await admin_client.post(
                    "/init_broadcaster",
                    json={
                        "host": host,
                        "port": port,
                        "rank_offset": rank_offset,
                        "inference_world_size": inference_world_size,
                        "timeout": timeout,
                    },
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 404:
                    get_logger().warning(
                        "The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization."
                    )

        await asyncio.gather(
            *(
                initialize_client(admin_client, client_num * gpus_per_server)
                for client_num, admin_client in enumerate(self.clients)
            )
        )

    async def update_weights(
        self,
        weight_dir: Path | None,
        *,
        transport: Literal["filesystem", "nccl", "nixl"],
        step: int = 0,
        on_paused: Callable[[], None] | None = None,
    ) -> None:
        """Update every inference engine that is up through its configured weight transport."""
        weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

        async with self._update_lock:
            targets = self.up()
            pause_times = await self._pause(targets, step)
            try:
                if on_paused is not None:
                    on_paused()
                transfer_times = await self._post(
                    targets,
                    "/update_weights",
                    json={"weight_dir": weight_dir_posix, "weight_version": step},
                    timeout_s=UPDATE_WEIGHTS_TIMEOUT_S,
                )
            finally:
                resume_times = await self._resume(targets)
            await self._admit(targets)
        self.update_times = {"pause": pause_times, "transfer": transfer_times, "resume": resume_times}

    async def update_modelexpress_weights(
        self,
        *,
        version_uid: str,
        step: int = 0,
        on_paused: Callable[[], None] | None = None,
    ) -> None:
        """Install an exact MX version on every engine that is up; keep inference paused if
        installation is uncertain."""
        async with self._update_lock:
            if not version_uid:
                raise ValueError("modelexpress requires version_uid")
            targets = self.up()
            pause_times = await self._pause(targets, step)
            if on_paused is not None:
                on_paused()
            transfer_times = await self._post(
                targets,
                "/update_weights_from_modelexpress",
                retry=False,
                json={"version_uid": version_uid, "weight_version": step},
                timeout_s=UPDATE_WEIGHTS_TIMEOUT_S,
            )
            resume_times = await self._resume(targets)
            await self._admit(targets)
            self.update_times = {"pause": pause_times, "transfer": transfer_times, "resume": resume_times}

    async def initialize_modelexpress(
        self,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        staging_buffer_bytes: int | None,
        staging_buffers_count: int,
        indices: list[int] | None = None,
    ) -> None:
        """Start the ModelExpress clients of the ``indices`` engines (all by default) under a new session."""
        workers_per_server = inference_world_size // len(self.clients)
        if workers_per_server < 1 or inference_world_size % len(self.clients):
            raise ValueError("modelexpress requires an equal number of workers per inference server")
        session_id = uuid.uuid4().hex

        async def initialize(index, client):
            response = await client.post(
                "/init_broadcaster",
                json={
                    "host": host,
                    "port": port,
                    "timeout": timeout,
                    "rank_offset": index * workers_per_server,
                    "inference_world_size": inference_world_size,
                    "session_id": session_id,
                    "staging_buffer_bytes": staging_buffer_bytes,
                    "staging_buffers_count": staging_buffers_count,
                },
                timeout=max(ADMIN_TIMEOUT_S, timeout),
            )
            response.raise_for_status()

        targets = range(len(self.clients)) if indices is None else indices
        await asyncio.gather(*(initialize(index, self.clients[index]) for index in targets))

    async def aclose(self) -> None:
        for client in self.clients + self._router_clients:
            await client.aclose()


def setup_admin_plane(
    client_config: PolicyClientConfig,
    model_name: str,
    roles: list[str | None] | None = None,
    tolerate_restarts: bool = False,
) -> AdminPlane:
    dynamo = client_config.dynamo
    if dynamo is not None and dynamo.enabled:
        from prime_rl.inference.dynamo import DynamoAdminPlane

        return DynamoAdminPlane(client_config, model_name)
    return AdminPlane(client_config, roles, tolerate_restarts)


async def check_inference_ready(client_config: ClientConfig, model_name: str) -> None:
    """One-shot readiness check of an inference endpoint (health + model
    listing) with transient clients — for frozen endpoints that never need a
    persistent admin plane."""
    admin = AdminPlane(client_config)
    try:
        await admin.wait_for_ready(model_name)
    finally:
        await admin.aclose()


async def connect_frozen_client(
    config: FrozenModelConfig, *, renderer_config: RendererConfig | None = None
) -> InferenceClient:
    """Connect to an externally hosted frozen model and wait for it."""
    get_logger().info(f"Initializing frozen model pool (model={config.name}, base_url={config.base_url})")
    if renderer_config is not None:
        clients = InferenceClient(
            config, model_name=config.name, train_client_type="renderer", renderer_config=renderer_config
        )
    else:
        clients = InferenceClient(config, model_name=config.name)
    await check_inference_ready(config, config.name)
    return clients


def setup_client(
    client_config: ClientConfig,
    client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    renderer_model_name: str | None = None,
) -> vf.ClientConfig:
    """Build a v1 client config for the base URL. ``client_type``
    ``renderer`` → token-in/out (``TrainClientConfig``, with the renderer the env
    server should use forwarded as a serialized config so it doesn't fall back to the
    default renderer); otherwise plain chat-completions (``EvalClientConfig``)."""
    is_renderer = client_type == "renderer"
    config_cls = TrainClientConfig if is_renderer else EvalClientConfig
    renderer_extra: dict = {}
    if is_renderer:
        renderer_extra = {
            "renderer": renderer_config,
            "renderer_model_name": renderer_model_name,
        }
    return config_cls(**client_config.model_dump(include=set(BaseClientConfig.model_fields)), **renderer_extra)


def _setup_session_client(client_config: ClientConfig) -> AsyncClient:
    """Create the client that releases router sessions, with a connection per release slot."""
    limits = httpx.Limits(
        max_connections=SESSION_RELEASE_CONCURRENCY, max_keepalive_connections=SESSION_RELEASE_CONCURRENCY
    )
    return setup_admin_clients(client_config, limits=limits)[0]


def setup_admin_clients(
    client_config: ClientConfig, limits: httpx.Limits = httpx.Limits(max_connections=4, max_keepalive_connections=1)
) -> list[AsyncClient]:
    """Create dedicated admin clients for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    When admin_base_url is set, uses those URLs instead of base_url, allowing
    weight updates to bypass routers in disaggregated P/D deployments.
    """
    urls = client_config.admin_base_url if client_config.admin_base_url else [client_config.base_url]

    def _setup_admin_client(base_url: str) -> httpx.AsyncClient:
        config = client_config.model_copy(update={"base_url": base_url})
        headers = resolve_headers(config)
        api_key = resolve_api_key(config)
        if api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        # Strip /v1 suffix since admin endpoints are at root level
        base_url = base_url.rstrip("/").removesuffix("/v1")

        return AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=limits,
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
        body = result.json() if result.headers.get("content-type", "").startswith("application/json") else {}
        if result.status_code != 200 or "data" not in body:
            raise RuntimeError(
                f"Listing the models of {admin_client.base_url} failed with status {result.status_code}: "
                f"{result.text[:300]}"
            )
        models = body["data"]
        # A LoRA server lists the model name only once the adapter is loaded; until then
        # the base model (served as ``<model>-base``) carries it as its ``root``.
        if not any(model_name in (model["id"], model.get("root")) for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {admin_client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def check_health(
    admin_clients: list[AsyncClient],
    interval: int = 1,
    log_interval: int = 30,
    timeout: int = 3600,
    quiet: bool = False,
) -> None:
    """Wait until every client's /health responds. With ``quiet``, the periodic
    waiting lines log at debug instead of info - used for engines fronted by a
    router, so startup logs one waiting line instead of one per engine."""
    logger = get_logger()

    async def _check_health(admin_client: AsyncClient) -> None:
        wait_time = 0
        logger.debug("Pinging /health until the inference server is ready")
        while wait_time < timeout:
            try:
                response = await admin_client.get("/health")
                if response.status_code == 404:
                    logger.warning("The route /health does not exist. Skipping health check.")
                    return
                response.raise_for_status()
                logger.debug(f"Inference pool is ready after {wait_time} seconds")
                return
            except Exception as e:
                if wait_time % log_interval == 0 and wait_time > 0:
                    log = logger.debug if quiet else logger.info
                    log(f"Waiting for inference server at {admin_client.base_url} to start up ({wait_time}s elapsed)")
                    logger.debug(f"Inference server at {admin_client.base_url} not reachable: {e!r}")
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await asyncio.gather(*[_check_health(admin_client) for admin_client in admin_clients])


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


async def _admin_post(
    client: AsyncClient,
    path: str,
    *,
    timeout_s: float = ADMIN_TIMEOUT_S,
    attempts: int = 10,
    retry_connect: bool = True,
    **kwargs,
) -> None:
    """POST an admin op with a bounded per-attempt timeout, retrying transient errors. Without
    ``retry_connect``, a refused connection (a dead engine) fails at once.

    The total wall-clock budget across all retries is twice the per-attempt timeout.
    """

    def retryable(error: BaseException) -> bool:
        return _is_retryable_admin_error(error) and (retry_connect or not isinstance(error, httpx.ConnectError))

    async for attempt in AsyncRetrying(
        retry=retry_if_exception(retryable),
        stop=stop_after_delay(2 * timeout_s) | stop_after_attempt(attempts),
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


async def _answers(client: AsyncClient) -> bool:
    """Whether the engine's /health answers OK within 10 s."""
    try:
        response = await client.get("/health", timeout=10)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


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
# the global admin AsyncClient uses `timeout=None`, so a stuck server would
# hang the orchestrator forever.
# `_PER_ATTEMPT` converts a hang into a TimeoutException so tenacity retries;
# `_TOTAL` is the wall-clock budget across all retries — pick whichever
# stop condition fires first.
LORA_LOAD_READ_TIMEOUT_S = 30.0
LORA_LOAD_TOTAL_TIMEOUT_S = 120.0


async def load_lora_adapter(admin_plane: AdminPlane, lora_name: str, lora_path: Path, step: int) -> None:
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
            params={"weight_version": step},
            timeout=httpx.Timeout(connect=10.0, read=LORA_LOAD_READ_TIMEOUT_S, write=60.0, pool=10.0),
        )
        response.raise_for_status()

    await asyncio.gather(*[_load_lora_adapter(client) for client in admin_plane.clients])


async def init_nixl_broadcast(
    admin_plane: AdminPlane,
    host: str,
    port: int,
    timeout: int,
    inference_world_size: int,
    session_id: str,
) -> None:
    """Configure every vLLM worker for NIXL + ModelExpress pulls."""
    admin_clients = admin_plane.clients
    workers_per_server = inference_world_size // len(admin_clients)

    async def initialize(admin_client: AsyncClient, rank_offset: int) -> None:
        await _admin_post(
            admin_client,
            "/init_broadcaster",
            timeout_s=max(ADMIN_TIMEOUT_S, timeout),
            json={
                "host": host,
                "port": port,
                "rank_offset": rank_offset,
                "inference_world_size": inference_world_size,
                "timeout": timeout,
                "session_id": session_id,
            },
        )

    await asyncio.gather(
        *[initialize(admin_client, index * workers_per_server) for index, admin_client in enumerate(admin_clients)]
    )


async def prefill_logprobs(openai: AsyncOpenAI, model: str, token_ids: list[int]) -> list[float]:
    """Prefill-score ``token_ids`` under ``model`` via ``/inference/v1/generate``
    + ``prompt_logprobs`` (the prime-rl server-side extension in
    ``inference/vllm/serving_tokens.py``). Returns one logprob per token (0.0 for
    the leading token, which has no preceding context)."""
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateTokensResponse

    # `/inference/v1/generate` is mounted at server root, not under `/v1`: pass an
    # absolute URL so the SDK skips the base-url merge. vLLM's `GenerateTokensResponse`
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
    response = GenerateTokensResponse.model_validate_json(http_response.content)
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
