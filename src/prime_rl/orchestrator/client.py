import asyncio
import os
from pathlib import Path

import httpx
import torch
from httpx import Response
from openai import AsyncOpenAI, NotFoundError

from prime_rl.orchestrator.config import ClientConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_weight_ckpt_model_path, get_step_path


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
    url = str(client.base_url)[:-4] + "/health"
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


async def update_weights(client: AsyncOpenAI, path: Path, step: int, server_type: str = "vllm") -> None:
    """POST to update backend weights.

    - For vLLM, send the path to the weight file (pytorch_model.bin).
    - For SGLang, send the path to the step directory (contains model files).
    """
    logger = get_logger()
    url = str(client.base_url)[:-4] + "/update_weights"
    try:
        if server_type == "sglang":
            # SGLang expects a directory or repo id; pass the step directory
            model_dir = get_step_path(path, step).absolute()
            if not model_dir.exists():
                raise FileNotFoundError(f"Weight checkpoint directory not found: {model_dir}")
            model_bin = model_dir / "pytorch_model.bin"
            if not model_bin.exists():
                raise FileNotFoundError(f"Weight checkpoint file missing: {model_bin}")
            logger.debug(f"Sending request to {url} to update weights from dir {model_dir}")
            await client.post(url, cast_to=Response, body={"model_path": model_dir.as_posix()})
        else:
            # vLLM accepts a file path to the checkpoint
            model_path = get_weight_ckpt_model_path(path, step).absolute()
            if not model_path.exists():
                raise FileNotFoundError(f"Weight checkpoint file not found: {model_path}")
            logger.debug(f"Sending request to {url} to update weights from file {model_path}")
            await client.post(url, cast_to=Response, body={"model_path": model_path.as_posix()})
    except NotFoundError:
        logger.warning(f"The route {url} does not exist. Skipping weight update.")
        return


async def reload_weights(client: AsyncOpenAI) -> None:
    """POST to reset backend weights."""
    logger = get_logger()
    url = str(client.base_url)[:-4] + "/reload_weights"
    try:
        logger.debug(f"Sending request to {url} to reload weights (reset to base model)")
        await client.post(url, cast_to=Response, body={})
    except NotFoundError:
        logger.warning(f"The route {url} does not exist. Skipping weight reload.")
        return
    await client.post(url, cast_to=Response, body={})


async def flush_cache(client: AsyncOpenAI) -> None:
    """POST to flush backend cache."""
    logger = get_logger()
    url = str(client.base_url)[:-4] + "/flush_cache"
    try:
        logger.debug(f"Sending request to {url} to flush cache")
        await client.post(url, cast_to=Response, body={})
    except NotFoundError:
        logger.warning(f"The route {url} does not exist. Skipping cache flush.")
        return


def apply_sampling_transforms(logits, temperature=1.0, top_p=1.0):
    t = torch.tensor(logits, dtype=torch.float32)
    t = t / max(temperature, 1e-6)
    if top_p < 1.0:
        probs = torch.softmax(t, dim=-1)
        sorted_probs, idx = torch.sort(probs, descending=True)
        cum = sorted_probs.cumsum(dim=-1)
        mask = cum > top_p
        sorted_probs[mask] = 0
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.log(probs + 1e-12).tolist()
    return torch.log_softmax(t, dim=-1).tolist()
