from __future__ import annotations

import asyncio
import time
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
from prime_rl.orchestrator.inference_metrics import parse_prometheus_text
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
        self._session_client = (
            setup_admin_clients(client_config.model_copy(update={"admin_base_url": None}))[0]
            if client_config.admin_base_url is not None
            else None
        )
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

        async def finish_session(session_id: str) -> None:
            try:
                await _admin_post(
                    self._session_client,
                    "/finish_session",
                    timeout_s=5.0,
                    params={"session_id": session_id},
                )
            except Exception as error:
                get_logger().debug(f"Failed to release inference session {session_id}: {error!r}")

        await asyncio.gather(*(finish_session(session_id) for session_id in session_ids))


class AdminPlane:
    """Admin plane of the policy inference deployment: one httpx client per
    engine process. The router serves no admin routes (pause/resume,
    update_weights, init_broadcaster, load_lora_adapter live on the engines),
    so these clients bypass it via ``admin_base_url``.

    The client order is load-bearing: ``admin_base_url`` order must match the
    GPU rank order used by NCCL/NIXL initialization and the metrics collector's
    role list index."""

    prefill: set[AsyncClient] = set()
    """The P/D prefill engines. They pause before the other engines."""

    # Per-engine times of the last update_weights phases (pause, transfer, resume).
    # Replaced, never mutated.
    update_times: dict[str, list[float]] = {}

    def __init__(self, client_config: ClientConfig, roles: list[str | None] | None = None):
        self.clients = setup_admin_clients(client_config)
        self.prefill = {client for client, role in zip(self.clients, roles or []) if role == "prefill"}
        # When admin URLs bypass a router, also health-check the client-facing
        # (router) endpoint - it only starts serving once its workers are healthy.
        self._router_clients = (
            setup_admin_clients(client_config.model_copy(update={"admin_base_url": None}))
            if client_config.admin_base_url
            else []
        )
        self._skip_model_check = client_config.skip_model_check
        self._wait_for_ready_timeout = client_config.wait_for_ready_timeout
        self._init_membership()

    def _init_membership(self) -> None:
        # Engine eviction (``monitor_health``): engines that failed a health check or an
        # admin op leave ``clients`` for ``evicted``; ``recovered`` are evicted engines that
        # answer again and rejoin at the next weight update.
        self.evicted: list[AsyncClient] = []
        self.recovered: set[AsyncClient] = set()
        self.evicting = False
        self.updating = False
        # The engines taking part in weight updates and a counter bumped whenever that set
        # changes. Transports that pin their receivers (NCCL) rebuild on a new generation.
        self.members = list(self.clients)
        self.generation = 0

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
        on_membership_change: Callable[[list[AsyncClient], int], Awaitable[None]] | None = None,
        abort: Callable[[AsyncClient], None] | None = None,
        resend: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        """Update every inference engine through its configured weight transport.

        While ``monitor_health`` runs, failing engines are evicted instead of failing the update.
        ``on_membership_change(members, generation)`` runs with the engines paused, before the
        version is acknowledged, whenever the set of updating engines changed; it may evict
        engines it fails to set up. ``abort(engine)`` runs as soon as an engine dies or hangs
        during the transfer, so the trainer and the other engines stop waiting for it.
        ``resend()`` then tells whether the trainer aborted and will send the version again to the
        regrouped survivors (NCCL); without it the survivors finish the transfer (NIXL)."""
        weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None
        rejoining = self._rejoining()
        self.updating = True
        times: dict[str, list[float]] = {}
        try:
            members = await self._pause(self.clients + rejoining, step, times.setdefault("pause", []))
            # A rejoining engine has no communicator (it may have been relaunched), so always regroup
            regroup = bool(rejoining)
            try:
                while True:
                    if regroup or members != self.members:
                        regroup = False
                        self.members = list(members)
                        self.generation += 1
                        if on_membership_change is not None:
                            await on_membership_change(members, self.generation)
                            members = self._standing(members)
                            self.members = list(members)
                    if on_paused is not None:
                        on_paused()
                        on_paused = None
                    times["transfer"] = []
                    members, retry = await self._transfer(members, weight_dir_posix, abort, resend, times["transfer"])
                    if not retry:
                        break
            finally:
                resumed = await self._each(members, "/resume", times=times.setdefault("resume", []))
        finally:
            self.updating = False
        self.update_times = times
        await self._readmit(rejoining, resumed, step)

    def _rejoining(self) -> list[AsyncClient]:
        return [client for client in self.evicted if client in self.recovered]

    async def _readmit(self, rejoining: list[AsyncClient], updated: list[AsyncClient], step: int) -> None:
        """Put the rejoining engines that took the new weights back into ``clients`` and the router."""
        for client in rejoining:
            if client in updated:
                self.evicted.remove(client)
                self.recovered.discard(client)
                self.clients.append(client)
                await self._set_routed(client, True)
                get_logger().info(f"Inference engine {client.base_url} rejoined at policy v{step}")

    async def _transfer(
        self,
        members: list[AsyncClient],
        weight_dir: str | None,
        abort: Callable[[AsyncClient], None] | None,
        resend: Callable[[], Awaitable[bool]] | None,
        times: list[float],
    ) -> tuple[list[AsyncClient], bool]:
        """Run ``/update_weights`` on ``members``. Returns the members still standing and whether
        the transfer was aborted and must run again."""
        if not self.evicting or abort is None:
            json = {"weight_dir": weight_dir}
            survivors = await self._each(
                members, "/update_weights", times=times, json=json, timeout_s=UPDATE_WEIGHTS_TIMEOUT_S
            )
            return survivors, False

        async def update(client: AsyncClient) -> None:
            start = time.perf_counter()
            # One attempt: retrying a collective receive cannot succeed on its own. No read timeout:
            # the deadline below tells a hung engine from the ones waiting behind it.
            response = await client.post(
                "/update_weights",
                json={"weight_dir": weight_dir},
                timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
            )
            response.raise_for_status()
            times.append(time.perf_counter() - start)

        tasks = {asyncio.create_task(update(client)): client for client in members}
        dead: list[AsyncClient] = []
        failed: list[AsyncClient] = []

        def drop(client: AsyncClient, error: object) -> None:
            dead.append(client)
            abort(client)
            get_logger().warning(f"Aborting the weight transfer to {client.base_url} ({error!r})")

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, timeout=UPDATE_WEIGHTS_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                # The engines still in the transfer wait for each other, so they all hit the deadline.
                # The ones whose server does not answer are hung and the others get another round.
                # If every server answers, all of them are dropped.
                answers = await asyncio.gather(*(_answers(tasks[task]) for task in pending))
                hung = {task for task, ok in zip(pending, answers) if not ok} or set(pending)
                for task in hung:
                    task.cancel()
                    drop(tasks[task], f"no answer after {UPDATE_WEIGHTS_TIMEOUT_S:.0f}s")
                pending -= hung
                continue
            for task in done:
                error = task.exception()
                # Gone: the transfer cannot finish while the others wait for it. Without a resend
                # the survivors are not aborted, so any failure means it stopped pulling.
                if isinstance(error, httpx.TransportError) or (error is not None and resend is None):
                    drop(tasks[task], error)
                elif error is not None:
                    failed.append(tasks[task])
        for client in dead:
            await self._evict(client, "weight update failed")
        if dead and resend is not None and await resend():
            return self._standing(members), True
        for client in failed:
            await self._evict(client, "weight update failed")
        return self._standing(members), False

    def _standing(self, members: list[AsyncClient]) -> list[AsyncClient]:
        return [client for client in members if client in self.clients or client in self.recovered]

    async def monitor_health(self, interval: float = 10.0, failures: int = 3, stall_timeout: float = 300.0) -> None:
        """Keep running on the surviving engines when one dies, instead of failing the run.
        While this runs, an engine is evicted and removed from the router when:
        - ``failures`` consecutive ``/liveness`` probes fail (the probe is a worker RPC, so it
          reaches the engine core and its workers, not only the API server);
        - it has running requests but generated no token for ``stall_timeout`` seconds;
        - an admin op fails during a weight update.
        Once an evicted engine answers again (e.g. after a relaunch), it gets the next weight
        update and is re-added."""
        self.evicting = True
        consecutive: dict[AsyncClient, int] = {}
        progress: dict[AsyncClient, tuple[float, float]] = {}
        # Engines evicted for a stall -> generated tokens at eviction. Liveness alone does not
        # clear a stall: they recover once their requests finish, progress, or they restart.
        stalled: dict[AsyncClient, float] = {}
        while True:
            await asyncio.sleep(interval)
            # A weight update keeps the workers busy and pauses generation
            if self.updating:
                progress.clear()
                continue
            await self._sync_router()
            clients = self.clients + self.evicted
            live = await asyncio.gather(*(_is_live(client) for client in clients))
            probed = self.clients + [client for client in stalled if client in self.evicted]
            counters = dict(zip(probed, await asyncio.gather(*(_generation_counters(client) for client in probed))))
            now = time.monotonic()
            # A weight update can readmit an engine while the probes above run
            for client in [client for client in self.clients if client in counters]:
                running, tokens = counters[client]
                last = progress.get(client)
                if running == 0 or last is None or tokens != last[1]:
                    progress[client] = (now, tokens)
                elif now - last[0] > stall_timeout:
                    progress.pop(client)
                    stalled[client] = tokens
                    await self._evict(
                        client, f"no token generated for {now - last[0]:.0f}s with {running:.0f} running requests"
                    )
            for client, ok in zip(clients, live):
                if client in self.evicted:
                    if ok and client in stalled:
                        running, tokens = counters[client]
                        if running and tokens == stalled[client]:
                            continue
                    stalled.pop(client, None)
                    if ok and client not in self.recovered:
                        get_logger().info(
                            f"Evicted inference engine {client.base_url} answers again; it rejoins at the next weight update"
                        )
                        self.recovered.add(client)
                    elif not ok:
                        self.recovered.discard(client)
                    continue
                consecutive[client] = 0 if ok else consecutive.get(client, 0) + 1
                if consecutive[client] >= failures:
                    consecutive[client] = 0
                    await self._evict(client, f"{failures} failed /liveness probes")

    async def _evict(self, client: AsyncClient, reason: str) -> None:
        self.recovered.discard(client)  # a rejoining engine that fails waits for the next recovery
        if client not in self.clients:
            return
        self.clients.remove(client)
        self.evicted.append(client)
        get_logger().warning(f"Evicted inference engine {client.base_url} ({reason}); {len(self.clients)} engines left")
        if not self.clients:
            raise RuntimeError("Every inference engine was evicted")
        await self._set_routed(client, False)

    async def _sync_router(self) -> None:
        """Drop evicted engines that the router lists again, e.g. after it was restarted with
        its original worker list."""
        for router in self._router_clients:
            try:
                response = await router.get("/list_workers", timeout=10.0)
                listed = set(response.json().get("urls", []))
            except Exception:
                continue
            for client in self.evicted:
                if str(client.base_url).rstrip("/") in listed:
                    await self._set_routed(client, False)

    async def _set_routed(self, client: AsyncClient, routed: bool) -> None:
        """Add or remove an engine from the client-facing vllm-router. Best effort: the router's
        own health check also ejects an engine that stops answering."""
        url = str(client.base_url).rstrip("/")
        for router in self._router_clients:
            try:
                await _admin_post(
                    router, "/add_worker" if routed else "/remove_worker", timeout_s=10.0, params={"url": url}
                )
            except Exception as error:
                get_logger().warning(
                    f"Could not {'add' if routed else 'remove'} {url} on router {router.base_url}: {error!r}"
                )

    async def _pause(self, clients: list[AsyncClient], step: int, times: list[float]) -> list[AsyncClient]:
        """Pause ``clients``, waiting for in-flight requests to drain; returns the ones that paused.

        Prefill engines pause first. A prefill engine's pause waits until decode has pulled the KV
        blocks it still holds, and a paused decode engine never pulls them: the prefill would wait
        for vLLM's NIXL KV lease to expire (30 s by default) and decode would later read the
        released blocks."""
        get_logger().debug(f"Pausing inference engines to update weights to policy v{step}")
        params = {"mode": "keep", "clear_cache": "false"}
        paused = await self._each([c for c in clients if c in self.prefill], "/pause", times=times, params=params)
        paused += await self._each([c for c in clients if c not in self.prefill], "/pause", times=times, params=params)
        return [client for client in clients if client in paused]

    async def _each(
        self, clients: list[AsyncClient], op: str, times: list[float] | None = None, **kwargs
    ) -> list[AsyncClient]:
        """Run one admin op on every client; evict the ones that fail and return the rest. Without
        eviction (``monitor_health`` not running), a failure raises. Appends each successful
        request's time to ``times``."""

        async def post(client: AsyncClient) -> None:
            start = time.perf_counter()
            await _admin_post(client, op, **kwargs)
            if times is not None:
                times.append(time.perf_counter() - start)

        if not self.evicting:
            await asyncio.gather(*(post(client) for client in clients))
            return list(clients)
        results = await asyncio.gather(*(post(client) for client in clients), return_exceptions=True)
        succeeded = []
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                if client in self.clients:
                    await self._evict(client, f"{op} failed: {result!r}")
                else:
                    self.recovered.discard(client)
            else:
                succeeded.append(client)
        return succeeded

    async def aclose(self) -> None:
        for client in self.clients + self.evicted + self._router_clients:
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


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
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


async def _generation_counters(client: AsyncClient) -> tuple[float, float]:
    """``(running requests, generated tokens so far)`` from the engine's Prometheus metrics;
    ``(0, 0)`` when they cannot be read (the liveness probe covers unreachable engines)."""
    try:
        response = await client.get("/metrics", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return 0.0, 0.0
    engines = parse_prometheus_text(response.text).values()
    running = sum(engine.gauges.get("num_requests_running", 0.0) for engine in engines)
    tokens = sum(
        engine.counters.get("generation_tokens", engine.counters.get("generation_tokens_total", 0.0))
        for engine in engines
    )
    return running, tokens


async def _answers(client: AsyncClient) -> bool:
    try:
        response = await client.get("/health", timeout=10.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


async def _is_live(client: AsyncClient) -> bool:
    try:
        response = await client.get("/liveness", timeout=60.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


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
# can take longer than the other admin ops. Keep it below the trainer's collective timeout
# and the NIXL weight_broadcast.timeout: it is what detects an engine that hangs during the transfer.
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


async def load_lora_adapter(admin_plane: AdminPlane, lora_name: str, lora_path: Path) -> None:
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
