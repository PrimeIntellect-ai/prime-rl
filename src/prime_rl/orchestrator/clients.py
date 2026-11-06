from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
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


@dataclass(eq=False)
class Engine:
    """One inference server of the admin plane. With inference recovery a server is ``live``, ``lost``
    (left out of weight updates until it answers again) or ``joining`` (answers again and takes part
    in the next weight update, but takes requests only once that update succeeded)."""

    client: AsyncClient
    role: str | None = None
    status: Literal["live", "lost", "joining"] = "live"
    instance_id: str | None = None
    """The server process's ID from ``/ready``; a new one means the server was replaced."""
    failed_probes: int = 0
    calls: set[asyncio.Task] = field(default_factory=set)
    """Admin calls in flight, cancelled when the server is lost."""
    losses: int = 0
    """How often the server was lost. A call it was lost during sees this change even if the server
    already rejoins (e.g. a replacement process answered the next probe)."""

    @property
    def url(self) -> str:
        return str(self.client.base_url)


def loss_counts(engines: list[Engine]) -> list[tuple[Engine, int]]:
    return [(engine, engine.losses) for engine in engines]


def lost_since(counts: list[tuple[Engine, int]]) -> list[Engine]:
    """Servers lost since ``counts`` were taken, including those that already rejoin."""
    return [engine for engine, losses in counts if engine.status == "lost" or engine.losses != losses]


class EngineLost(Exception):
    """Inference servers were lost during a weight update that needs every server taking part."""

    def __init__(self, engines: list[Engine]):
        super().__init__(f"Lost inference servers {', '.join(engine.url for engine in engines)} during a weight update")
        self.engines = engines


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

    def __init__(self, client_config: ClientConfig, roles: list[str | None] | None = None):
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
        self._modelexpress_lock = asyncio.Lock()
        self._engines: list[Engine] = []
        self.abort_transfer: Callable[[], None] | None = None
        """Stops the transfer in flight when a server is lost, if the transport needs every server."""
        self.on_availability: Callable[[bool], None] | None = None
        """Told whether any server takes requests, whenever that changes."""
        self.loss_verdict_s: float | None = None
        """With inference recovery: how long a failed admin call waits for the monitor to judge its
        server, so the failure of a dying server counts as its loss rather than failing the update."""
        self.losses = 0

    @property
    def engines(self) -> list[Engine]:
        """One ``Engine`` per admin client, in client order (rebuilt if the clients are replaced)."""
        if [engine.client for engine in self._engines] != self.clients:
            roles = self.roles or [None] * len(self.clients)
            self._engines = [Engine(client, role) for client, role in zip(self.clients, roles, strict=True)]
        return self._engines

    def members(self) -> list[Engine]:
        """Servers that take part in weight updates."""
        return [engine for engine in self.engines if engine.status != "lost"]

    @property
    def available(self) -> bool:
        return any(engine.status == "live" for engine in self.engines)

    def watch_availability(self, callback: Callable[[bool], None]) -> None:
        """Tell ``callback`` whenever it changes whether any server takes requests, and at once if none
        does: the monitor may have lost every server before the callback was attached."""
        self.on_availability = callback
        if not self.available:
            callback(False)

    def mark_lost(self, engine: Engine, reason: str) -> None:
        """Leave a server out of weight updates and cancel its admin calls in flight."""
        if engine.status == "lost":
            return
        was_available = self.available
        engine.status = "lost"
        engine.failed_probes = 0
        engine.losses += 1
        self.losses += 1
        get_logger().warning(f"Lost inference server {engine.url} ({reason}); training continues without it")
        for call in list(engine.calls):
            call.cancel()
        if self.abort_transfer is not None:
            self.abort_transfer()
        if was_available and not self.available and self.on_availability is not None:
            self.on_availability(False)

    def mark_joining(self, engine: Engine) -> None:
        """Let a server that answers again take part in the next weight update."""
        engine.status = "joining"
        get_logger().info(f"Inference server {engine.url} is back; it rejoins at the next weight update")

    def mark_joined(self, engines: list[Engine]) -> None:
        """Joining servers that took a complete weight update take requests again."""
        was_available = self.available
        for engine in engines:
            if engine.status == "joining":
                engine.status = "live"
                get_logger().info(f"Inference server {engine.url} rejoined and takes requests again")
        if not was_available and self.available and self.on_availability is not None:
            self.on_availability(True)

    async def call(self, engine: Engine, request: Callable[[], Awaitable[None]], collective: bool = False) -> None:
        """Run an admin call to one server, cancelled if the server is lost. A failed call first waits
        for the monitor's verdict on its server (``loss_verdict_s``): the failure of a dying server
        must not pass as a plain error. In a ``collective`` call, where one server's loss fails every
        server's call, any loss ends the wait, and any failure aborts the transfer in progress."""
        task = asyncio.current_task()
        engine.calls.add(task)
        own_losses, losses = engine.losses, self.losses
        try:
            await request()
        except Exception:
            if self.loss_verdict_s is not None:
                deadline = time.monotonic() + self.loss_verdict_s
                while (
                    engine.losses == own_losses
                    and not (collective and self.losses != losses)
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.5)
            if collective and self.abort_transfer is not None:
                # Also without a loss: the publisher and the other servers would wait for this one
                # until their deadlines.
                self.abort_transfer()
            raise
        finally:
            engine.calls.discard(task)

    async def _post_engines(
        self, engines: list[Engine], path: str, collective: bool = False, **kwargs
    ) -> tuple[list[float], list[Engine]]:
        """POST ``path`` to every server concurrently (``collective``: see ``call``). Returns the request
        times and the servers lost meanwhile (their calls are cancelled); raises another failure once
        every request finished."""

        async def post(engine: Engine) -> float:
            start = time.perf_counter()
            await self.call(engine, lambda: _admin_post(engine.client, path, **kwargs), collective)
            return time.perf_counter() - start

        counts = loss_counts(engines)
        results = await asyncio.gather(*(post(engine) for engine in engines), return_exceptions=True)
        lost = lost_since(counts)
        times = []
        for engine, result in zip(engines, results, strict=True):
            if isinstance(result, BaseException):
                if engine not in lost:
                    raise result
            else:
                times.append(result)
        return times, lost

    async def _pause_members(self, engines: list[Engine], step: int) -> list[float]:
        """Pause servers, prefill before decode (see ``_pause_engines``)."""
        get_logger().debug(f"Pausing inference engines to update weights to policy v{step}")
        pause_times: list[float] = []
        for group in (
            [engine for engine in engines if engine.role == "prefill"],
            [engine for engine in engines if engine.role != "prefill"],
        ):
            times, _ = await self._post_engines(group, "/pause", params={"mode": "keep", "clear_cache": "false"})
            pause_times += times
        return pause_times

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
        recoverable: bool = False,
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
                        **({"recoverable": True} if recoverable else {}),
                    },
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 404 and not recoverable:
                    get_logger().warning(
                        "The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization."
                    )
                else:
                    raise

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
        abort_path: Path | None = None,
        before_transfer: Callable[[list[Engine]], Awaitable[None]] | None = None,
        all_or_nothing: bool = False,
        engines: list[Engine] | None = None,
    ) -> None:
        """Update every member server through its configured weight transport. A recoverable
        NCCL transfer stops waiting on its publisher once ``abort_path`` exists.

        ``on_paused`` acknowledges the offer once the members are paused and ``before_transfer``
        prepares the paused members (e.g. forms their transfer group). A server lost meanwhile is left out; with
        ``all_or_nothing`` (one collective transfer) the transfer is stopped and ``EngineLost`` raised,
        leaving the others paused for a retry. ``engines`` limits the update to some members (e.g.
        servers catching up)."""
        weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

        engines = self.members() if engines is None else engines
        counts = loss_counts(engines)
        pause_times = await self._pause_members(engines, step)
        engines = [engine for engine in engines if engine not in lost_since(counts)]
        if on_paused is not None:
            on_paused()
        if all_or_nothing and abort_path is not None:
            self.abort_transfer = abort_path.touch
        try:
            if before_transfer is not None:
                await before_transfer(engines)
            transfer_times, lost = await self._post_engines(
                engines,
                "/update_weights",
                collective=all_or_nothing,
                json={
                    "weight_dir": weight_dir_posix,
                    "weight_version": step,
                    **({"abort_path": abort_path.as_posix()} if abort_path is not None else {}),
                },
                timeout_s=UPDATE_WEIGHTS_TIMEOUT_S,
                retry=False,
            )
        except Exception:
            # The others' transfer fails once a lost server's abort stopped it.
            lost = [engine for engine in engines if engine in lost_since(counts)]
            if not (all_or_nothing and lost):
                raise
            transfer_times = []
        finally:
            self.abort_transfer = None
        if lost and all_or_nothing:
            raise EngineLost(lost)
        # A failed transfer may have overwritten only part of the model. Keep all
        # engines paused until a complete replacement has succeeded everywhere.
        engines = [engine for engine in engines if engine not in lost_since(counts)]
        resume_times, _ = await self._post_engines(engines, "/resume")
        get_logger().debug("All inference engines resumed")
        self.mark_joined(engines)
        self.update_times = {"pause": pause_times, "transfer": transfer_times, "resume": resume_times}

    async def form_nccl_group(
        self,
        engines: list[Engine],
        *,
        host: str,
        port: int,
        timeout: int,
        gpus_per_server: int,
        session_id: str,
        abort_path: Path,
    ) -> None:
        """Join ``engines`` to a fresh NCCL transfer group of ``session_id``, ranked in their order after
        the trainer. Raises ``EngineLost`` if a server is lost before the group formed."""
        world_size = len(engines) * gpus_per_server
        get_logger().info(f"Forming NCCL transfer group {session_id} with {len(engines)} servers ({world_size} GPUs)")

        async def join(engine: Engine, rank_offset: int) -> None:
            response = await engine.client.post(
                "/init_broadcaster",
                json={
                    "host": host,
                    "port": port,
                    "rank_offset": rank_offset,
                    "inference_world_size": world_size,
                    "timeout": timeout,
                    "session_id": session_id,
                    "recoverable": True,
                    "abort_path": abort_path.as_posix(),
                },
                timeout=httpx.Timeout(connect=10.0, read=timeout, write=60.0, pool=10.0),
            )
            response.raise_for_status()

        counts = loss_counts(engines)
        results = await asyncio.gather(
            *(
                self.call(engine, partial(join, engine, index * gpus_per_server), collective=True)
                for index, engine in enumerate(engines)
            ),
            return_exceptions=True,
        )
        lost = lost_since(counts)
        if lost:
            raise EngineLost(lost)
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
        """Install an exact MX version; keep inference paused if installation is uncertain."""
        async with self._modelexpress_lock:
            if not version_uid:
                raise ValueError("modelexpress requires version_uid")
            pause_times = await _pause_engines(self.clients, step=step, roles=self.roles)
            if on_paused is not None:
                on_paused()

            async def install(client) -> float:
                start = time.perf_counter()
                response = await client.post(
                    "/update_weights_from_modelexpress",
                    json={"version_uid": version_uid, "weight_version": step},
                    timeout=httpx.Timeout(connect=10.0, read=UPDATE_WEIGHTS_TIMEOUT_S, write=60.0, pool=10.0),
                )
                response.raise_for_status()
                return time.perf_counter() - start

            results = await asyncio.gather(*(install(client) for client in self.clients), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            resume_times = await _resume_engines(self.clients)
            self.update_times = {"pause": pause_times, "transfer": results, "resume": resume_times}

    async def initialize_modelexpress(
        self,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        staging_buffer_bytes: int | None,
        staging_buffers_count: int,
    ) -> None:
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

        await asyncio.gather(*(initialize(index, client) for index, client in enumerate(self.clients)))

    async def reset_weight_transfer(self, stale_trainer_instances: list[str]) -> None:
        """Make every engine drop its NIXL transfer plan with a replaced trainer. The next update
        pulls from a trainer table whose instance is not listed."""
        await _post_all(
            self.clients,
            "/reset_weight_transfer",
            json={"stale_trainer_instances": stale_trainer_instances},
            timeout_s=ADMIN_TIMEOUT_S,
            retry=False,
        )

    async def aclose(self) -> None:
        for client in self.clients + self._router_clients:
            await client.aclose()


def setup_admin_plane(
    client_config: PolicyClientConfig, model_name: str, roles: list[str | None] | None = None
) -> AdminPlane:
    dynamo = client_config.dynamo
    if dynamo is not None and dynamo.enabled:
        from prime_rl.inference.dynamo import DynamoAdminPlane

        return DynamoAdminPlane(client_config, model_name)
    return AdminPlane(client_config, roles)


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
    client: AsyncClient, path: str, *, timeout_s: float = ADMIN_TIMEOUT_S, retry: bool = True, **kwargs
) -> None:
    """POST an admin op with a bounded per-attempt timeout, retrying transient errors.

    The total wall-clock budget across all retries is twice the per-attempt timeout.
    """
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(lambda error: retry and _is_retryable_admin_error(error)),
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


async def _post_all(admin_clients: list[AsyncClient], path: str, **kwargs) -> list[float]:
    """POST ``path`` to every engine concurrently; returns each engine's request time. Raises the
    first failure only once every request has finished: a caller that moves on after one engine
    failed (e.g. clears a transfer's abort marker) must not race the others."""

    async def post(client: AsyncClient) -> float:
        start = time.perf_counter()
        await _admin_post(client, path, **kwargs)
        return time.perf_counter() - start

    results = await asyncio.gather(*[post(client) for client in admin_clients], return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results


async def _pause_engines(
    admin_clients: list[AsyncClient], *, step: int, roles: list[str | None] | None = None
) -> list[float]:
    """Pause all inference engines, waiting for in-flight requests to drain.
    Returns each engine's pause time.

    Prefill engines pause first. A prefill engine's pause waits until decode has pulled the KV
    blocks it still holds, and a paused decode engine never pulls them: the prefill would wait
    for vLLM's NIXL KV lease to expire (30 s by default) and decode would later read the
    released blocks."""
    logger = get_logger()
    logger.debug(f"Pausing inference engines to update weights to policy v{step}")
    roles = roles or [None] * len(admin_clients)
    prefill = [client for client, role in zip(admin_clients, roles, strict=True) if role == "prefill"]
    rest = [client for client, role in zip(admin_clients, roles, strict=True) if role != "prefill"]
    pause_times: list[float] = []
    for clients in (prefill, rest):
        pause_times += await _post_all(clients, "/pause", params={"mode": "keep", "clear_cache": "false"})
    logger.debug(f"All inference engines paused in {min(pause_times):.2f}-{max(pause_times):.2f}s")
    return pause_times


async def _resume_engines(admin_clients: list[AsyncClient]) -> list[float]:
    """Resume all inference engines after weight update. Returns each engine's resume time.

    Resuming is idempotent (it just clears the paused flag), so retrying transient
    failures is safe; a dropped /resume would leave engines paused indefinitely.
    """
    resume_times = await _post_all(admin_clients, "/resume")
    get_logger().debug("All inference engines resumed")
    return resume_times


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


async def load_lora_adapter(
    admin_plane: AdminPlane, lora_name: str, lora_path: Path, step: int, engines: list[Engine] | None = None
) -> None:
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

    # A server lost meanwhile is left out: it loads the adapter of the update it rejoins at.
    engines = admin_plane.members() if engines is None else engines
    counts = loss_counts(engines)
    results = await asyncio.gather(
        *[admin_plane.call(engine, partial(_load_lora_adapter, engine.client)) for engine in engines],
        return_exceptions=True,
    )
    lost = lost_since(counts)
    for engine, result in zip(engines, results, strict=True):
        if isinstance(result, BaseException) and engine not in lost:
            raise result
    admin_plane.mark_joined([engine for engine in engines if engine not in lost])


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
