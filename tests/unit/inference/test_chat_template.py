import pytest
from vllm import LLM

from zeroband.inference.rewards import LenRewardsConfig
from zeroband.inference.utils import format_prompts

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture
def prompts() -> list[str]:
    return ["What is the capital of France?", "Explain quantum mechanics"]


@pytest.fixture(params=["system_prompt", "instruction"])
def length_rewards_config(request: pytest.FixtureRequest) -> LenRewardsConfig:
    return LenRewardsConfig(length_prompt_location=request.param)


def test_format_prompts(llm: LLM, prompts: list[str]):
    """Test format_prompts with no length rewards configuration."""
    formatted_prompts = format_prompts(prompts=prompts, target_lengths=[-1] * len(prompts), len_rewards_config=None, llm=llm)

    assert formatted_prompts == [
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain quantum mechanics<|im_end|>\n<|im_start|>assistant\n",
    ]


def test_format_prompts_with_length_rewards(llm: LLM, prompts: list[str], length_rewards_config: LenRewardsConfig):
    formatted_prompts = format_prompts(prompts=prompts, target_lengths=[100, 200], len_rewards_config=length_rewards_config, llm=llm)

    if length_rewards_config.length_prompt_location == "system_prompt":
        assert formatted_prompts == [
            "<|im_start|>system\nThink for 100 tokens before giving a response.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>system\nThink for 200 tokens before giving a response.<|im_end|>\n<|im_start|>user\nExplain quantum mechanics<|im_end|>\n<|im_start|>assistant\n",
        ]
    else:
        assert formatted_prompts == [
            "<|im_start|>user\nWhat is the capital of France? Think for 100 tokens before giving a response.<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\nExplain quantum mechanics Think for 200 tokens before giving a response.<|im_end|>\n<|im_start|>assistant\n",
        ]
