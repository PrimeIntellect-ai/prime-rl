import asyncio
from pathlib import Path

import httpx
from httpx import Response
from openai import AsyncOpenAI, BaseModel
from openai.types.chat import ChatCompletion
from vllm.entrypoints.openai.api_server import TokenizeResponse

from prime_rl.orchestrator.config import ClientConfig, ModelConfig, SamplingConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_weight_ckpt_model_path


def setup_client(client_config: ClientConfig) -> AsyncOpenAI:
    # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
    timeout = httpx.Timeout(timeout=1200, connect=5.0)
    # We use as many concurrent connections as possible, but lower than available ports
    limits = httpx.Limits(
        max_connections=28000,  # OAI default: 1000
        max_keepalive_connections=28000,  # OAI default: 100
    )
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    base_url = f"http://{client_config.host}:{client_config.port}/v1"
    return AsyncOpenAI(
        base_url=base_url,
        api_key=client_config.api_key,
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
        except Exception as e:
            if wait_time % log_interval == 0 and wait_time > 0:
                logger.warning(f"Inference pool was not reached after {wait_time} seconds (Error: {e})")
            await asyncio.sleep(interval)
            wait_time += interval
    msg = f"Inference pool is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
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
    url = str(client.base_url)[:-4] + "/update_weights"
    model_path = get_weight_ckpt_model_path(path, step)
    logger.debug(f"Sending request to {url} to update weights from {model_path}")
    await client.post(url, cast_to=Response, body={"model_path": model_path.as_posix()})


async def reload_weights(client: AsyncOpenAI) -> None:
    """Make a HTTP post request to the vLLM server to reload weights (reset to base model)."""
    logger = get_logger()
    url = str(client.base_url)[:-4] + "/reload_weights"
    logger.debug(f"Sending request to {url} to reload weights (reset to base model)")
    await client.post(url, cast_to=Response, body={})


async def tokenize(client: AsyncOpenAI, model_config: ModelConfig, messages: list[dict[str, str]]) -> list[int]:
    url = str(client.base_url)[:-4] + "/tokenize"

    class OAITokenizeResponse(BaseModel, TokenizeResponse):
        """Make vLLM's TokenizeResponse compatible with OAI client."""

    tokenize_response = await client.post(
        url, cast_to=OAITokenizeResponse, body={"model": model_config.name, "messages": messages}
    )
    return tokenize_response.tokens


async def generate_completion(
    client: AsyncOpenAI,
    model_config: ModelConfig,
    sampling_config: SamplingConfig,
    messages: list[dict[str, str]],
) -> ChatCompletion:
    response = await client.chat.completions.create(
        messages=messages,
        model=model_config.name,
        temperature=sampling_config.temperature,
        max_tokens=sampling_config.max_tokens,
        seed=sampling_config.seed,
        logprobs=True,
        extra_body={
            "min_tokens": sampling_config.min_tokens,
            "return_tokens_as_token_ids": True,
        },
    )
    assert len(response.choices) == 1, "Response should always have one choice"
    return response
