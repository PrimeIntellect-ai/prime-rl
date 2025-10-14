import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from httpx import Response
from openai import AsyncOpenAI, NotFoundError

from prime_rl.orchestrator.config import ClientConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_weight_ckpt_model_path


def _server_base_from_oai(base_url: str) -> str:
    """Extract server base URL from OpenAI base URL (removes /v1 suffix)"""
    s = str(base_url).strip()
    return s[:-3] if s.endswith("/v1") else s


def _admin_client() -> httpx.AsyncClient:
    """Create a dedicated admin client for weight update operations."""
    # One-off socket; avoids waiting behind streaming keep-alives
    return httpx.AsyncClient(
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        headers={"Connection": "close"},
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=30.0, pool=None),
    )


def _trace_id() -> str:
    """Generate a unique trace ID for request tracking."""
    return uuid.uuid4().hex[:16]


def setup_client(client_config: ClientConfig) -> AsyncOpenAI:
    # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
    timeout = httpx.Timeout(timeout=client_config.timeout, connect=5.0)
    # We use as many concurrent connections as possible, but lower than available ports
    limits = httpx.Limits(
        max_connections=28000,  # OAI default: 1000
        max_keepalive_connections=28000,  # OAI default: 100
    )
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    return AsyncOpenAI(
        base_url=client_config.base_url,
        api_key=os.getenv(client_config.api_key_var, "EMPTY"),
        max_retries=10,  # OAI default: 2 (does exponential backoff and reasonable timeout in between retries)
        http_client=http_client,
    )


async def check_health(client: AsyncOpenAI, interval: int = 1, log_interval: int = 10, timeout: int = 1800) -> None:
    logger = get_logger()
    wait_time = 0
    url = str(client.base_url).strip()[:-4] + "/health"
    logger.debug(f"Starting pinging {url} to check health")
    while wait_time < timeout:
        try:
            await client.get(url, cast_to=Response, options={"max_retries": 0})
            logger.debug(f"Inference pool is ready after {wait_time} seconds")
            return
        except NotFoundError:
            logger.warning(f"The route {url} does not exist. Skipping health check.")
            return
        except Exception as e:
            if wait_time % log_interval == 0 and wait_time > 0:
                logger.warning(f"Inference server was not reached after {wait_time} seconds (Error: {e})")
            await asyncio.sleep(interval)
            wait_time += interval
    msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
    logger.error(msg)
    raise TimeoutError(msg)


async def check_has_model(client: AsyncOpenAI, model_name: str) -> None:
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    models = (await client.models.list()).data
    if not any(model.id == model_name for model in models):
        raise ValueError(f"Model {model_name} was not found in the inference pool")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def update_weights(client: AsyncOpenAI, path: Path, step: int) -> None:
    """Make a HTTP post request to the vLLM server to update the weights."""
    logger = get_logger()
    url = _server_base_from_oai(client.base_url) + "/update_weights"
    tid = _trace_id()
    t0 = time.monotonic()

    try:
        model_path = get_weight_ckpt_model_path(path, step).absolute()
        logger.info(f"[weights][{tid}] client.send url={url} path={model_path} t={t0:.6f}")

        async with _admin_client() as admin:
            r = await admin.post(
                url,
                json={"model_path": model_path.as_posix()},
                headers={"x-trace-id": tid}
            )
            r.raise_for_status()
            payload: dict[str, Any] = r.json()

        t1 = time.monotonic()
        wall_ms = (t1 - t0) * 1000.0
        rpc_ms = float(payload.get("rpc_ms", -1))
        queue_ms = wall_ms - rpc_ms if rpc_ms >= 0 else -1
        logger.info(f"[weights][{tid}] client.done wall_ms={wall_ms:.1f} rpc_ms={rpc_ms:.1f} queue_ms={queue_ms:.1f}")

    except NotFoundError:
        logger.warning(f"The route {url} does not exist. Skipping weight update.")
        return


async def reload_weights(client: AsyncOpenAI) -> None:
    """Make a HTTP post request to the vLLM server to reload weights (reset to base model)."""
    logger = get_logger()
    url = _server_base_from_oai(client.base_url) + "/reload_weights"
    tid = _trace_id()
    t0 = time.monotonic()

    try:
        logger.info(f"[weights][{tid}] client.send url={url} t={t0:.6f}")

        async with _admin_client() as admin:
            r = await admin.post(url, json={}, headers={"x-trace-id": tid})
            r.raise_for_status()
            payload = r.json()

        t1 = time.monotonic()
        wall_ms = (t1 - t0) * 1000.0
        rpc_ms = float(payload.get("rpc_ms", -1))
        queue_ms = wall_ms - rpc_ms if rpc_ms >= 0 else -1
        logger.info(f"[weights][{tid}] client.done wall_ms={wall_ms:.1f} rpc_ms={rpc_ms:.1f} queue_ms={queue_ms:.1f}")

    except NotFoundError:
        logger.warning(f"The route {url} does not exist. Skipping weight reload.")
        return
