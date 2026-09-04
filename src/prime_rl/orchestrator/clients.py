from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import httpx
import verifiers.v1 as vf
from httpx import AsyncClient
from openai import AsyncOpenAI
from renderers import RendererConfig
from tenacity import AsyncRetrying, retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential
from verifiers.v1.configs.client import EvalClientConfig, TrainClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger


class PrefillScorer:
    """Prefill-scores token ids against a pool's endpoint, lazily resolving
    a single OpenAI client from the pool's train client config."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    async def score(self, config: vf.ClientConfig, model: str, token_ids: list[int]) -> list[float]:
        if self._client is None:
            # Build the OpenAI client straight from the config fields — works for any
            # ClientConfig type; resolve_client would hand back an EvalClient (no `.openai`)
            # for these chat-completions teacher configs.
            self._client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=os.environ.get(config.api_key_var) or "EMPTY",
                default_headers=config.headers or None,
            )
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
        self.model_name = model_name

    async def score(self, token_ids: list[int]) -> list[float]:
        """Prefill-score ``token_ids`` under this endpoint's model (one logprob
        per token, 0.0 for the leading token)."""
        return await self._scorer.score(self.train_client, self.model_name, token_ids)

    async def aclose(self) -> None:
        await self._scorer.aclose()


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

    async def initialize_nccl(
        self,
        *,
        host: str,
        port: int,
        timeout: int,
        inference_world_size: int,
        quantize_in_weight_transfer: bool = False,
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
                        "quantize_in_weight_transfer": quantize_in_weight_transfer,
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

    async def aclose(self) -> None:
        for client in self.clients + self._router_clients:
            await client.aclose()


def setup_admin_plane(client_config: ClientConfig, model_name: str) -> AdminPlane:
    if client_config.dynamo is not None:
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
    env_headers = {
        k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
    }
    headers = {**client_config.headers, **env_headers}
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
        env_headers = {
            k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
        }
        headers = {**client_config.headers, **env_headers}
        api_key = os.getenv(client_config.api_key_var, "EMPTY")
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        headers = {name: value for name, value in headers.items() if name.lower() != "accept-encoding"}
        headers["Accept-Encoding"] = "identity"

        # Strip /v1 suffix since admin endpoints are at root level
        base_url = base_url.rstrip("/").removesuffix("/v1")

        return AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(min(30.0, max(1.0, float(client_config.wait_for_ready_timeout)))),
            trust_env=False,
        )

    return [_setup_admin_client(base_url) for base_url in urls]


MAX_ADMIN_RESPONSE_BYTES = 1024 * 1024


async def _bounded_request(
    client: AsyncClient,
    method: str,
    path: str,
    *,
    max_response_bytes: int = MAX_ADMIN_RESPONSE_BYTES,
    **kwargs,
) -> httpx.Response:
    """Send an admin request without eagerly buffering an unbounded response."""

    request_headers = kwargs.pop("headers", {})
    request_headers = {name: value for name, value in request_headers.items() if name.lower() != "accept-encoding"}
    request_headers["Accept-Encoding"] = "identity"
    async with client.stream(method, path, headers=request_headers, **kwargs) as response:
        content_encoding = response.headers.get("Content-Encoding")
        if content_encoding and content_encoding.strip().lower() != "identity":
            raise ValueError(f"Admin endpoint returned unsupported Content-Encoding {content_encoding!r}")
        body = bytearray()
        async for chunk in response.aiter_raw():
            if len(body) + len(chunk) > max_response_bytes:
                raise ValueError(f"Admin response body exceeds {max_response_bytes} bytes")
            body.extend(chunk)
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=bytes(body),
            request=response.request,
        )


async def maybe_check_has_model(
    admin_clients: list[AsyncClient], model_name: str, skip_model_check: bool = False
) -> None:
    if skip_model_check:
        return
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(
        *[_bounded_request(admin_client, "GET", "/v1/models") for admin_client in admin_clients]
    )
    for admin_client, result in zip(admin_clients, results):
        models = result.json()["data"]
        if not any(model["id"] == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {admin_client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def check_health(
    admin_clients: list[AsyncClient],
    interval: int = 1,
    log_interval: int = 30,
    timeout: int = 1800,
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
                response = await _bounded_request(admin_client, "GET", "/health")
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
    # per-attempt read timeout below turns a stuck server into a bounded retry loop.
    if isinstance(exception, (TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# Per-attempt read timeout for admin ops, overridable per call. The read timeout
# converts a stuck server into a TimeoutException that tenacity retries. Sized
# for `/pause`, which drains in-flight requests
# (mode="keep") and so can legitimately take a while.
ADMIN_TIMEOUT_S = 300.0
# `/update_weights` runs a collective NCCL receive across all DP workers, which
# can take longer than the other admin ops.
UPDATE_WEIGHTS_TIMEOUT_S = 720.0


async def _admin_post(
    client: AsyncClient, path: str, *, timeout_s: float = ADMIN_TIMEOUT_S, **kwargs
) -> httpx.Response:
    """POST an admin op with absolute per-attempt and total timeouts, retrying transient errors.

    The total wall-clock budget across all retries is twice the per-attempt timeout.
    """
    async with asyncio.timeout(2 * timeout_s):
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable_admin_error),
            stop=stop_after_delay(2 * timeout_s) | stop_after_attempt(10),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        ):
            with attempt:
                async with asyncio.timeout(timeout_s):
                    response = await _bounded_request(
                        client,
                        "POST",
                        path,
                        timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=60.0, pool=10.0),
                        **kwargs,
                    )
                    response.raise_for_status()
                    return response

    raise RuntimeError("Admin request retry loop exited without a response")


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
# (small adapter file + KV cache reset, single-digit seconds in practice).
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
                "quantize_in_weight_transfer": False,
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
