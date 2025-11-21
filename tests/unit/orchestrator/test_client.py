import pytest
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from prime_rl.orchestrator.config import SamplingConfig
from prime_rl.orchestrator.utils import get_sampling_args


@pytest.fixture(scope="module")
def client():
    return OpenAI(base_url="http://localhost:8000/v1", api_key="")


@pytest.fixture(scope="module")
def model_name() -> str:
    return "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def prompt() -> list[dict]:
    return [{"role": "user", "content": "Hello, how are you?"}]


@pytest.fixture
def response(client: OpenAI, model_name: str, prompt) -> ChatCompletion:
    sampling_config = SamplingConfig(max_tokens=10)
    sampling_args = get_sampling_args(sampling_config)

    return client.chat.completions.create(
        model=model_name,
        messages=prompt,
        **sampling_args,
    )


def test_token_ids_in_response(response: ChatCompletion):
    response_dict = response.model_dump()
    assert "prompt_token_ids" in response_dict, "prompt_token_ids should be present in the response"
    assert len(response_dict["choices"]) == 1, "response should have one choice"
    assert "token_ids" in response_dict["choices"][0], "token_ids should be present in the response"


def test_logprobs_in_response(response: ChatCompletion):
    response_dict = response.model_dump()
    assert "logprobs" in response_dict["choices"][0], "logprobs should be present in the response"


def token_ids_and_logprobs_match(response: ChatCompletion):
    response_dict = response.model_dump()
    token_ids = response_dict["choices"][0]["token_ids"]
    logprobs = response_dict["choices"][0]["logprobs"]["content"]
    assert len(token_ids) == len(logprobs), "token_ids and logprobs should have the same length"
