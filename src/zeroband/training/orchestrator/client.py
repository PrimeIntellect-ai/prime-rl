import asyncio
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from zeroband.training.orchestrator.config import ClientConfig, ModelConfig, SamplingConfig
from zeroband.utils.logger import get_logger


def setup_client(client_config: ClientConfig) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=client_config.base_url, api_key=client_config.api_key)


async def check_health(client: AsyncOpenAI, interval: int = 1, log_interval: int = 10, timeout: int = 360) -> None:
    logger = get_logger()
    wait_time = 0
    url = str(client.base_url)[:-4] + "/health"
    logger.debug(f"Starting pinging {url} to check health")
    while wait_time < timeout:
        try:
            await client._client.get(url=url)
            logger.debug(f"Inference pool is ready after {wait_time} seconds")
            return
        except Exception as e:
            if wait_time % log_interval == 0:
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


async def reload_weights(client: AsyncOpenAI, path: Path, step: int) -> None:
    """Make a HTTP post request to the vLLM server to reload the weights."""
    logger = get_logger()
    url = str(client.base_url)[:-4] + "/reload_weights"
    model_path = path / f"step_{step}" / "model.pt"
    logger.debug(f"Sending request to {url} to reload weights from {model_path}")
    await client._client.post(url=url, json={"model_path": model_path.as_posix()})


async def reset_weights(client: AsyncOpenAI) -> None:
    """Make a HTTP post request to the vLLM server to reset weights to the base model."""
    logger = get_logger()
    url = str(client.base_url)[:-4] + "/reset_weights"
    logger.debug(f"Sending request to {url} to reset weights to base model")
    await client._client.post(url=url, json={})


async def tokenize(client: AsyncOpenAI, model_config: ModelConfig, messages: list[dict[str, str]]) -> list[int]:
    url = str(client.base_url)[:-4] + "/tokenize"
    res = await client._client.post(url=url, json={"model": model_config.name, "messages": messages})
    return res.json()["tokens"]


async def generate_completion(
    client: AsyncOpenAI,
    model_config: ModelConfig,
    sampling_config: SamplingConfig,
    messages: list[dict[str, str]],
    num_input_tokens: int,
) -> ChatCompletion:
    if sampling_config.max_seq_len:
        sampling_config.max_tokens = sampling_config.max_seq_len - num_input_tokens

    response = await client.chat.completions.create(
        messages=messages,
        model=model_config.name,
        temperature=sampling_config.temperature,
        top_p=sampling_config.top_p,
        max_tokens=sampling_config.max_tokens,
        seed=sampling_config.seed,
        logprobs=True,
        extra_body={
            "top_k": sampling_config.top_k,
            "min_p": sampling_config.min_p,
            "min_tokens": sampling_config.min_tokens,
            "return_tokens_as_token_ids": True,
        },
    )
    assert len(response.choices) == 1, "Response should always have one choice"
    return response
