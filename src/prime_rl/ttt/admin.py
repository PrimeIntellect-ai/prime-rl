"""Shared inference-admin operations for both TTT service engines."""

import asyncio

import httpx

from prime_rl.utils.logger import get_logger


class AdapterUnloadError(RuntimeError):
    """One or more inference replicas did not confirm an adapter unload."""


class AdapterLoadError(RuntimeError):
    """One or more inference replicas did not confirm an adapter load."""


async def post_with_timeout(
    http: httpx.AsyncClient,
    url: str,
    *,
    json: dict,
    timeout_seconds: float,
) -> httpx.Response:
    """POST with a real wall-clock deadline, including custom/mock transports.

    httpx's own timeout protects network phases, but a custom transport (and some stuck
    protocol paths) need an outer asyncio deadline too. The caller owns reconciliation
    after an ambiguous timeout.
    """
    async with asyncio.timeout(timeout_seconds):
        return await http.post(url, json=json, timeout=timeout_seconds)


def _is_known_absent_adapter_response(response: httpx.Response, adapter_name: str) -> bool:
    """Recognize vLLM's idempotent "adapter is absent" response.

    A bare 404 can mean the admin route itself is missing, and a 400 can describe any
    malformed request.  A structured 404 that names this adapter (or carries vLLM's typed
    ``NotFoundError``) proves the desired postcondition (adapter absent) already holds —
    matching on the name rather than an exact message survives vLLM message churn, and a
    route-missing 404 would not mention the adapter name.
    """
    if response.status_code != 404:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error", payload.get("message"))
    if isinstance(error, dict) and error.get("type") == "NotFoundError":
        return True
    return adapter_name in str(error)


async def unload_adapter_from_replicas(
    http: httpx.AsyncClient,
    inference_admin_urls: list[str],
    adapter_name: str,
    timeout_seconds: float = 120.0,
) -> None:
    """Attempt every replica and fail if any retryable unload did not succeed.

    A release retry commonly finds the adapter already absent; vLLM reports that as a
    structured 404 ``NotFoundError``. Transport failures, unrecognized 4xx responses, and
    5xx responses remain errors so a missing admin route or bad request cannot be mistaken
    for a successful unload.
    """

    async def unload_one(url: str) -> str | None:
        try:
            response = await post_with_timeout(
                http,
                f"{url.rstrip('/')}/v1/unload_lora_adapter",
                json={"lora_name": adapter_name},
                timeout_seconds=timeout_seconds,
            )
        except (httpx.HTTPError, TimeoutError) as e:
            return f"{url}: {type(e).__name__}: {e}"
        if response.status_code // 100 == 2 or _is_known_absent_adapter_response(response, adapter_name):
            return None
        return f"{url}: HTTP {response.status_code}: {response.text[:200]}"

    failures = [
        failure
        for failure in await asyncio.gather(*(unload_one(url) for url in inference_admin_urls))
        if failure is not None
    ]
    if failures:
        detail = "; ".join(failures)
        get_logger().warning(f"TTT adapter unload incomplete for {adapter_name}: {detail}")
        raise AdapterUnloadError(detail)


async def load_adapter_into_replicas(
    http: httpx.AsyncClient,
    inference_admin_urls: list[str],
    adapter_name: str,
    ckpt_path: str,
    timeout_seconds: float = 120.0,
) -> None:
    """Load every replica concurrently and report every failure.

    Concurrent fan-out makes the transaction deadline one per-replica timeout rather than
    N times that timeout. The caller reconciles any partial success by unloading from all
    replicas before releasing rollout ownership.
    """

    async def load_one(url: str) -> str | None:
        try:
            response = await post_with_timeout(
                http,
                f"{url.rstrip('/')}/load_lora_adapter",
                json={"lora_name": adapter_name, "lora_path": ckpt_path},
                timeout_seconds=timeout_seconds,
            )
            response.raise_for_status()
        except (httpx.HTTPError, TimeoutError) as e:
            return f"{url}: {type(e).__name__}: {e}"
        return None

    failures = [
        failure
        for failure in await asyncio.gather(*(load_one(url) for url in inference_admin_urls))
        if failure is not None
    ]
    if failures:
        detail = "; ".join(failures)
        get_logger().warning(f"TTT adapter load incomplete for {adapter_name}: {detail}")
        raise AdapterLoadError(detail)
