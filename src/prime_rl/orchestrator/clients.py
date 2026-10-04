from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import httpx
import verifiers.v1 as vf
from httpx import AsyncClient
from openai import AsyncOpenAI
from renderers import RendererConfig
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential
from verifiers.v1.configs.client import EvalClientConfig, TrainClientConfig

from prime_rl.configs.eval import PRIME_INFERENCE_URL
from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger


def resolve_api_key(api_key_var: str) -> str:
    """The API key named by ``api_key_var``; ``PRIME_API_KEY`` also falls back to the prime
    CLI config (``prime login``), like the verifiers client does. ``"EMPTY"`` when unset."""
    api_key = os.environ.get(api_key_var)
    if not api_key and api_key_var == "PRIME_API_KEY":
        from prime_sandboxes import Config as PrimeConfig

        api_key = PrimeConfig().api_key
    return api_key or "EMPTY"


def resolve_headers(client_config: ClientConfig) -> dict[str, str]:
    """The static headers plus those read from the environment. A Prime Inference client
    without a team header gets the team from ``$PRIME_TEAM_ID`` or the prime CLI config,
    like the verifiers client: a team's internal models are served only under it."""
    env_headers = {
        k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
    }
    headers = {**client_config.headers, **env_headers}
    if client_config.base_url.startswith(PRIME_INFERENCE_URL) and "X-Prime-Team-ID" not in headers:
        from prime_sandboxes import Config as PrimeConfig

        if team_id := os.environ.get("PRIME_TEAM_ID") or PrimeConfig().team_id:
            headers["X-Prime-Team-ID"] = team_id
    return headers


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
        self._openai: AsyncOpenAI | None = None
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
        if self._openai is None:
            # Built from the config fields: resolve_client would return an EvalClient
            # (no `.openai`) for chat-completions teacher configs.
            self._openai = AsyncOpenAI(
                base_url=self.train_client.base_url,
                api_key=resolve_api_key(self.train_client.api_key_var),
                default_headers=self.train_client.headers or None,
            )
        return await prefill_logprobs(self._openai, self.model_name, token_ids)

    async def aclose(self) -> None:
        if self._openai is not None:
            await self._openai.close()
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

    def __init__(self, client_config: ClientConfig):
        self.clients = setup_admin_clients(client_config)
        # When admin URLs bypass a router, also health-check the client-facing
        # (router) endpoint - it only starts serving once its workers are healthy.
        self._router_clients = (
            setup_admin_clients(client_config.model_copy(update={"admin_base_url": None}))
            if client_config.admin_base_url
            else []
        )
        self._skip_model_check = client_config.skip_model_check
        self._wait_for_ready_timeout = client_config.wait_for_ready_timeout

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

    async def init_broadcaster(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        session_id: str = "default",
    ) -> None:
        """Join every engine to the trainer's weight broadcast (NCCL or NIXL)."""
        gpus_per_server = inference_world_size // len(self.clients)
        get_logger().info(
            f"Initializing weight broadcast: {len(self.clients)} servers, "
            f"inference_world_size={inference_world_size}, gpus_per_server={gpus_per_server}"
        )
        await asyncio.gather(
            *(
                _admin_post(
                    admin_client,
                    "/init_broadcaster",
                    timeout_s=max(ADMIN_TIMEOUT_S, timeout),
                    json={
                        "host": host,
                        "port": port,
                        "rank_offset": index * gpus_per_server,
                        "inference_world_size": inference_world_size,
                        "timeout": timeout,
                        "session_id": session_id,
                    },
                )
                for index, admin_client in enumerate(self.clients)
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
        """Update every inference engine through its configured weight transport."""
        weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

        await _pause_engines(self.clients, step=step)
        try:
            if on_paused is not None:
                on_paused()
            await asyncio.gather(
                *[
                    _admin_post(
                        admin_client,
                        "/update_weights",
                        json={"weight_dir": weight_dir_posix},
                        timeout_s=UPDATE_WEIGHTS_TIMEOUT_S,
                    )
                    for admin_client in self.clients
                ]
            )
        finally:
            await _resume_engines(self.clients)

    async def load_lora_adapter(self, lora_name: str, lora_path: Path) -> None:
        """Load a LoRA adapter on every engine via our ``/load_lora_adapter`` wrapper.

        The prefix cache is not reset here; the orchestrator salts it per weight
        version (see ``orchestrator/envs.py``) so KV computed under old weights is
        never reused. vLLM answers 404 while the adapter files are not yet visible
        (NFS propagation), so 404 is retried too.
        """
        await asyncio.gather(
            *(
                _admin_post(
                    admin_client,
                    "/load_lora_adapter",
                    timeout_s=LORA_LOAD_TIMEOUT_S,
                    retry_not_found=True,
                    json={"lora_name": lora_name, "lora_path": lora_path.as_posix()},
                )
                for admin_client in self.clients
            )
        )

    async def aclose(self) -> None:
        for client in self.clients + self._router_clients:
            await client.aclose()


def setup_admin_plane(client_config: ClientConfig, model_name: str) -> AdminPlane:
    dynamo = client_config.dynamo
    if dynamo is not None and dynamo.enabled:
        from prime_rl.inference.dynamo import DynamoAdminPlane

        return DynamoAdminPlane(client_config, model_name)
    return AdminPlane(client_config)


async def check_inference_ready(client_config: ClientConfig, model_name: str) -> None:
    """One-shot readiness check of an inference endpoint (health + model
    listing) with transient clients — for frozen endpoints that never need a
    persistent admin plane."""
    admin = AdminPlane(client_config)
    try:
        await admin.wait_for_ready(model_name)
    finally:
        await admin.aclose()


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
    headers = resolve_headers(client_config)
    return config_cls(
        base_url=client_config.base_url, api_key_var=client_config.api_key_var, headers=headers, **renderer_extra
    )


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    """Create dedicated admin clients for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    When admin_base_url is set, uses those URLs instead of base_url, allowing
    weight updates to bypass routers in disaggregated P/D deployments.
    """
    urls = client_config.admin_base_url if client_config.admin_base_url else [client_config.base_url]

    def _setup_admin_client(base_url: str) -> httpx.AsyncClient:
        headers = resolve_headers(client_config)
        api_key = resolve_api_key(client_config.api_key_var)
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
        if not any(model["id"] == model_name for model in models):
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


def _is_retryable_admin_error(exception: BaseException, *, retry_not_found: bool = False) -> bool:
    """Check if an exception should trigger a retry for an admin op."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on transient server errors (5xx, e.g. engine briefly unresponsive);
        # client errors (4xx) won't fix themselves on retry, except an opted-in 404.
        status = exception.response.status_code
        return status >= 500 or (retry_not_found and status == 404)
    # Retry on transport-level failures (timeouts, connection resets, etc.) so the
    # per-attempt read timeout below turns a stuck server into a bounded retry loop
    # instead of hanging forever on the global timeout=None admin client.
    return isinstance(exception, httpx.TransportError)


# Per-attempt read timeout for admin ops, overridable per call. The admin
# AsyncClient uses `timeout=None`, so without this a stuck server would hang the
# weight update forever: the read timeout converts a hang into a TimeoutException
# that tenacity retries. Sized for `/pause`, which drains in-flight requests
# (mode="keep") and so can legitimately take a while.
ADMIN_TIMEOUT_S = 300.0
# `/update_weights` runs a collective NCCL receive across all DP workers, which
# can take longer than the other admin ops.
UPDATE_WEIGHTS_TIMEOUT_S = 720.0
# A LoRA load is a small adapter file plus a KV cache reset: seconds in practice.
LORA_LOAD_TIMEOUT_S = 60.0


async def _admin_post(
    client: AsyncClient,
    path: str,
    *,
    timeout_s: float = ADMIN_TIMEOUT_S,
    retry_not_found: bool = False,
    **kwargs,
) -> None:
    """POST an admin op with a bounded per-attempt timeout, retrying transient errors.

    The total wall-clock budget across all retries is twice the per-attempt timeout.
    """
    async for attempt in AsyncRetrying(
        retry=retry_if_exception(partial(_is_retryable_admin_error, retry_not_found=retry_not_found)),
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
    logger.debug(f"Pausing inference engines to update weights to policy v{step}")
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


async def prefill_logprobs(openai: AsyncOpenAI, model: str, token_ids: list[int]) -> list[float]:
    """Prefill-score ``token_ids`` under ``model`` via ``/inference/v1/generate``
    + ``prompt_logprobs`` (the prime-rl server-side extension in
    ``inference/vllm/serving_tokens.py``). Returns one logprob per token (0.0 for
    the leading token, which has no preceding context)."""
    from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateResponse

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
