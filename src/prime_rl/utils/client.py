from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from itertools import cycle
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
import verifiers as vf
from httpx import AsyncClient
from openai import NotFoundError
from renderers import RendererConfig
from tenacity import AsyncRetrying, retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger

# Identity tuple used by ``select_train_client`` to key load counts. ``api_base_url``
# distinguishes servers; ``X-data-parallel-rank`` distinguishes DP shards within a
# server, since the router uses that header to route to specific GPU ranks.
ClientIdentity = tuple[str, str | None]


def client_identity(client: vf.ClientConfig) -> ClientIdentity:
    """Stable identity for load balancing across inference clients."""
    return (client.api_base_url, client.extra_headers.get("X-data-parallel-rank"))


def train_typed_client(
    client: vf.ClientConfig,
    *,
    train_client_type: str,
    renderer_config: RendererConfig | None,
    renderer_model_name: str | None,
    pool_size: int | None,
) -> vf.ClientConfig:
    """Twin of ``client`` carrying the pool's train-path client type.

    Same server, headers, and timeouts; only the client type (and, for the
    renderer type, the renderer fields — mirroring ``setup_clients``) change.
    Identity when ``client`` already speaks the train client type."""
    if client.client_type == train_client_type:
        return client
    update: dict = {"client_type": train_client_type}
    if train_client_type == "renderer":
        update.update(
            renderer_config=renderer_config,
            renderer_model_name=renderer_model_name,
            renderer_pool_size=pool_size,
        )
    return client.model_copy(update=update)


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
    def eval_clients(self) -> list[vf.ClientConfig]:
        """Get eval clients."""
        ...

    def update_model_name(self, model_name: str) -> None:
        """Update the model name."""
        ...

    async def get_eval_client(self) -> vf.ClientConfig:
        """Get next eval client in round-robin fashion."""
        ...

    def as_train_client(self, client: vf.ClientConfig) -> vf.ClientConfig:
        """Re-type ``client`` to the pool's train-path client type (same
        server, headers, and timeouts). Used at eval so trained members
        generate through the same client semantics as training (e.g. the
        renderer client's think-channel splitting)."""
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

    async def update_weights(
        self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0, nccl_lora: bool = False
    ) -> None:
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
        self._train_client_type = train_client_type
        self._renderer_config = renderer_config
        self._renderer_model_name = renderer_model_name
        self._pool_size = pool_size
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

    def as_train_client(self, client: vf.ClientConfig) -> vf.ClientConfig:
        return train_typed_client(
            client,
            train_client_type=self._train_client_type,
            renderer_config=self._renderer_config,
            renderer_model_name=self._renderer_model_name,
            pool_size=self._pool_size,
        )

    async def select_train_client(self, load: Mapping[ClientIdentity, int]) -> vf.ClientConfig:
        while not self.train_clients:
            await asyncio.sleep(0.5)
        return min(self.train_clients, key=lambda c: load[client_identity(c)])

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        await check_health(
            self._admin_clients, timeout=timeout if timeout is not None else self._wait_for_ready_timeout
        )
        await maybe_check_has_model(self._admin_clients, model_name, skip_model_check=self._skip_model_check)

    async def update_weights(
        self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0, nccl_lora: bool = False
    ) -> None:
        await update_weights(self._admin_clients, weight_dir, lora_name=lora_name, step=step, nccl_lora=nccl_lora)

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


class ClientConnectionPolicy(Protocol):
    """Connection/retry policy shared by every config that backs a verifiers client.

    Satisfied by ``prime_rl.configs.shared.ClientConfig`` (inference pool) and
    ``prime_rl.configs.multi_agent.FixedMemberTargetConfig`` (fixed members).
    """

    @property
    def api_key_var(self) -> str: ...
    @property
    def timeout(self) -> float: ...
    @property
    def connect_timeout(self) -> float: ...
    @property
    def max_connections(self) -> int: ...
    @property
    def max_keepalive_connections(self) -> int: ...
    @property
    def max_retries(self) -> int: ...


def build_client(
    policy: ClientConnectionPolicy,
    *,
    api_base_url: str,
    client_type: str = "openai_chat_completions",
    client_idx: int = 0,
    extra_headers: dict[str, str] | None = None,
    extra_headers_from_state: dict[str, str] | None = None,
    renderer_config: RendererConfig | None = None,
    renderer_model_name: str | None = None,
    renderer_pool_size: int | None = None,
) -> vf.ClientConfig:
    """Build one verifiers client config. Single owner of connection/retry policy wiring."""
    return vf.ClientConfig(
        client_idx=client_idx,
        client_type=client_type,
        api_base_url=api_base_url,
        api_key_var=policy.api_key_var,
        timeout=policy.timeout,
        connect_timeout=policy.connect_timeout,
        max_connections=policy.max_connections,
        max_keepalive_connections=policy.max_keepalive_connections,
        max_retries=policy.max_retries,
        extra_headers=extra_headers or {},
        extra_headers_from_state=extra_headers_from_state or {},
        renderer_config=renderer_config,
        renderer_model_name=renderer_model_name,
        renderer_pool_size=renderer_pool_size,
    )


def setup_clients(
    client_config: ClientConfig,
    client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    renderer_model_name: str | None = None,
    pool_size: int | None = None,
) -> list[vf.ClientConfig]:
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
                build_client(
                    client_config,
                    api_base_url=base_url,
                    client_type=client_type,
                    client_idx=client_idx,
                    extra_headers=headers,
                    extra_headers_from_state=client_config.extra_headers_from_state,
                    **renderer_extra,
                )
            )
            client_idx += 1
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
    results = await _gather_admin(
        admin_clients,
        [admin_client.get("/v1/models") for admin_client in admin_clients],
        op_name="GET /v1/models",
    )
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
                    logger.info(
                        f"Inference server was not reached after {wait_time} seconds (Error: {e}) on {admin_client.base_url}"
                    )
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await _gather_admin(
        admin_clients,
        [_check_health(admin_client) for admin_client in admin_clients],
        op_name="health check",
    )


class AdminGatherError(RuntimeError):
    """One or more inference-server peers failed a fan-out admin operation.

    Names the failing peers so an opaque first-exception death (the death
    multiplier behind the weight-sync OOM incident: one bad peer raises and the
    bare ``asyncio.gather`` kills the whole 16-node job with no attribution)
    becomes an attributable, potentially-recoverable error.
    """

    def __init__(self, op_name: str, failures: list[tuple[AsyncClient, BaseException]], total: int) -> None:
        self.op_name = op_name
        self.failures = failures
        self.total = total
        peers = ", ".join(f"{client.base_url} ({type(exc).__name__}: {exc})" for client, exc in failures)
        super().__init__(f"{op_name} failed on {len(failures)}/{total} inference peer(s): {peers}")


async def _gather_admin(
    admin_clients: list[AsyncClient],
    coros: list,
    *,
    op_name: str,
    raise_on_failure: bool = True,
) -> list:
    """Fan-out ``coros`` (one per admin client) and attribute per-peer failures.

    ``return_exceptions=True`` is the *means* to inspect-and-attribute, not to
    swallow: every failed peer is logged loudly with its base URL and error, an
    ``asyncio.CancelledError`` is re-raised verbatim (cancellation is not a peer
    failure), and -- when ``raise_on_failure`` -- an aggregated
    :class:`AdminGatherError` naming the dead peer(s) is raised so the
    orchestrator/watcher can act on *which* peer died instead of an opaque
    first-exception. On full success the result list is returned unchanged so
    callers that consume per-peer payloads (e.g. the model-presence check) keep
    working exactly as before.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    failures: list[tuple[AsyncClient, BaseException]] = []
    for admin_client, result in zip(admin_clients, results):
        if isinstance(result, asyncio.CancelledError):
            # Cancellation is not a peer failure -- never swallow it.
            raise result
        if isinstance(result, BaseException):
            failures.append((admin_client, result))
    if failures:
        logger = get_logger()
        for admin_client, exc in failures:
            logger.error(f"{op_name} failed on inference peer {admin_client.base_url}: {exc!r}")
        if raise_on_failure:
            raise AdminGatherError(op_name, failures, total=len(admin_clients))
    return results


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
    # Compose retry (per-peer _admin_post) with attribution (_gather_admin names
    # the dead peer instead of an opaque first-exception that kills the job).
    await _gather_admin(
        admin_clients,
        [_admin_post(client, "/pause", params={"mode": "keep", "clear_cache": "false"}) for client in admin_clients],
        op_name="pause engines",
    )
    logger.debug("All inference engines paused")


async def _resume_engines(admin_clients: list[AsyncClient]) -> None:
    """Resume all inference engines after weight update.

    Resuming is idempotent (it just clears the paused flag), so retrying transient
    failures is safe; a dropped /resume would leave engines paused indefinitely.
    """
    logger = get_logger()
    await _gather_admin(
        admin_clients,
        [_admin_post(client, "/resume") for client in admin_clients],
        op_name="resume engines",
    )
    logger.debug("All inference engines resumed")


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | None,
    lora_name: str | None = None,
    step: int = 0,
    nccl_lora: bool = False,
) -> None:
    """Update weights on static inference servers.

    Pauses all engines first to drain in-flight requests, then performs the
    weight update, then resumes. This ensures all DP workers are idle and can
    participate in the collective weight transfer.

    Note: The server-side /update_weights endpoint automatically resets the prefix cache
    to invalidate any cached KV states computed with the old weights.
    """
    logger = get_logger()

    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

    if lora_name is not None and nccl_lora:
        if weight_dir is None:
            raise ValueError("NCCL LoRA update requires a broadcast marker directory")
        await update_lora_adapter(admin_clients, lora_name, weight_dir, step)
    elif lora_name is not None and weight_dir is not None:
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

            await _gather_admin(
                admin_clients,
                [
                    _admin_post(
                        admin_client,
                        "/update_weights",
                        json={"weight_dir": weight_dir_posix},
                        timeout_s=UPDATE_WEIGHTS_TIMEOUT_S,
                    )
                    for admin_client in admin_clients
                ],
                op_name="update weights",
            )
        finally:
            await _resume_engines(admin_clients)


LORA_UPDATE_STATUS_TIMEOUT_S = 300.0
LORA_UPDATE_STATUS_INTERVAL_S = 0.2


async def update_lora_adapter(admin_clients: list[AsyncClient], lora_name: str, weight_dir: Path, step: int) -> None:
    """Arm an NCCL LoRA update, release the trainer broadcast, and wait for completion."""
    logger = get_logger()
    adapters = [{"lora_name": lora_name, "lora_int_id": 1}]

    async def _arm_lora_update(admin_client: AsyncClient) -> None:
        # Arm is a pause-class drain op; route through _admin_post for the same
        # bounded-timeout + transient-retry contract as the other admin sites.
        await _admin_post(admin_client, "/update_lora", json={"step": step, "adapters": adapters})

    async def _wait_lora_update(admin_client: AsyncClient) -> None:
        deadline = time.monotonic() + LORA_UPDATE_STATUS_TIMEOUT_S
        while time.monotonic() < deadline:
            response = await admin_client.get(
                "/update_lora/status",
                params={"step": step},
                timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    return
            elif response.status_code >= 500:
                response.raise_for_status()
            await asyncio.sleep(LORA_UPDATE_STATUS_INTERVAL_S)
        raise TimeoutError(f"Timed out waiting for NCCL LoRA update step {step}")

    await _pause_engines(admin_clients, step=step)
    try:
        await _gather_admin(
            admin_clients,
            [_arm_lora_update(admin_client) for admin_client in admin_clients],
            op_name="arm NCCL LoRA update",
        )
        nccl_ready_file = weight_dir / NCCL_READY_MARKER
        nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
        nccl_ready_file.touch()
        logger.debug(f"Created NCCL_READY marker at {nccl_ready_file}")
        await _gather_admin(
            admin_clients,
            [_wait_lora_update(admin_client) for admin_client in admin_clients],
            op_name="await NCCL LoRA update",
        )
    finally:
        await _resume_engines(admin_clients)


def _is_retryable_lora_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for LoRA loading."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 404 (adapter not found) or 500 (server error during loading)
        return exception.response.status_code in (404, 500)
    # Retry only failures where a new POST can plausibly succeed: connection
    # establishment problems (server restarting, transient network). A READ
    # timeout means the server accepted the request and is still working on a
    # serialized load — re-POSTing just queues a duplicate behind it, so let
    # the generous per-attempt read window be the verdict instead.
    if isinstance(exception, (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
        return True
    return False


# Per-attempt and total bounds for `/load_lora_adapter`. Sized for the worst
# real case, not the best: a rank-64 expert-targeted adapter is a multi-GB
# safetensors file, and at weight-update time every replica reads it from
# Lustre simultaneously while still serving decode traffic — 30s reads were
# routinely exceeded at 13 replicas (watcher died, fail-fast killed the run,
# 2026-06-11). The bounds still convert a genuinely stuck server into a
# TimeoutException (tenacity retries per attempt; `_TOTAL` is the wall-clock
# budget across all retries — whichever stop condition fires first).
# Read window per attempt is deliberately ~the whole budget: vLLM serializes
# adapter loads per engine, so re-POSTing a slow-but-progressing load only
# queues a duplicate behind it (observed death spiral on a busy replica,
# 2026-06-11: 240s attempts stacked until the total budget reraised). Retries
# are reserved for connection-class failures where a new POST can help.
LORA_LOAD_READ_TIMEOUT_S = 900.0
LORA_LOAD_TOTAL_TIMEOUT_S = 1200.0


async def load_lora_adapter(admin_clients: list[AsyncClient], lora_name: str, lora_path: Path) -> None:
    """Make a HTTP post request to the vLLM server to load a LoRA adapter.

    Uses our wrapper endpoint that also resets the prefix cache to invalidate
    KV states computed with old weights.

    Retries with exponential backoff if the adapter files are not found,
    which can happen due to NFS propagation delays.
    """
    logger = get_logger()
    lora_path_posix = lora_path.as_posix()
    logger.info(f"Loading LoRA adapter {lora_name} from {lora_path} on {len(admin_clients)} inference server(s)")

    @retry(
        retry=retry_if_exception(_is_retryable_lora_error),
        stop=stop_after_delay(LORA_LOAD_TOTAL_TIMEOUT_S) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _load_lora_adapter(admin_client: AsyncClient) -> None:
        # `gather` surfaces only the first failure; log per-server so a stuck
        # replica in a multi-server fanout is identifiable from the logs.
        logger.debug(f"Sending request to load LoRA adapter {lora_name} from {lora_path} on {admin_client.base_url}")
        try:
            response = await admin_client.post(
                "/load_lora_adapter",
                json={"lora_name": lora_name, "lora_path": lora_path_posix},
                timeout=httpx.Timeout(connect=10.0, read=LORA_LOAD_READ_TIMEOUT_S, write=60.0, pool=10.0),
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning(
                f"Failed to load LoRA adapter {lora_name} from {lora_path} on {admin_client.base_url}: {exc!r}"
            )
            raise

    await _gather_admin(
        admin_clients,
        [_load_lora_adapter(admin_client) for admin_client in admin_clients],
        op_name="load LoRA adapter",
    )
    logger.info(f"Loaded LoRA adapter {lora_name} on {len(admin_clients)} inference server(s)")


async def unload_lora_adapter(admin_clients: list[AsyncClient], lora_name: str) -> None:
    """Make a HTTP post request to the vLLM server to unload a LoRA adapter."""
    logger = get_logger()

    async def _unload_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to unload LoRA adapter {lora_name}")
        await admin_client.post("/v1/unload_lora_adapter", json={"lora_name": lora_name})
        # TODO: The first one can fail, but subsequent ones should succeed.
        # response.raise_for_status()

    await _gather_admin(
        admin_clients,
        [_unload_lora_adapter(admin_client) for admin_client in admin_clients],
        op_name="unload LoRA adapter",
        raise_on_failure=False,
    )


async def init_nccl_broadcast(
    admin_clients: list[AsyncClient],
    host: str,
    port: int,
    timeout: int,
    inference_world_size: int | None = None,
    quantize_in_weight_transfer: bool = False,
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

    if not admin_clients:
        raise ValueError("Cannot initialize NCCL broadcast without inference admin clients")
    if inference_world_size < 1:
        raise ValueError(f"inference_world_size must be positive, got {inference_world_size}")
    if inference_world_size % len(admin_clients) != 0:
        num_servers = len(admin_clients)
        raise ValueError(
            f"inference_world_size ({inference_world_size}) must be divisible by the number of "
            f"inference servers ({num_servers})"
        )

    gpus_per_server = inference_world_size // len(admin_clients)

    logger.info(
        f"Initializing NCCL broadcast: {len(admin_clients)} servers, "
        f"inference_world_size={inference_world_size}, gpus_per_server={gpus_per_server}"
    )

    async def _init_nccl_broadcast(admin_client: AsyncClient, rank_offset: int) -> None:
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

    await _gather_admin(
        admin_clients,
        [
            _init_nccl_broadcast(admin_client, client_num * gpus_per_server)
            for client_num, admin_client in enumerate(admin_clients)
        ],
        op_name="init NCCL broadcaster",
    )
