import pytest
from transformers import AutoTokenizer

from zeroband.inference.config import LenRewardsConfig
from zeroband.inference.utils import format_prompts


@pytest.fixture
def prompts() -> list[str]:
    return ["What is the capital of France?", "Explain quantum mechanics"]


@pytest.fixture(params=["deepseek-ai/DeepSeek-R1-0528", "Qwen/QwQ-32B", "Qwen/Qwen3-0.6B"])
def tokenizer(request: pytest.FixtureRequest) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(request.param)


@pytest.fixture(params=["system_prompt", "instruction"])
def length_rewards_config(request: pytest.FixtureRequest) -> LenRewardsConfig:
    return LenRewardsConfig(length_prompt_location=request.param)


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_format_prompts(prompts: list[str], tokenizer: AutoTokenizer, enable_thinking: bool):
    """Test format_prompts with no length rewards configuration."""
    formatted_prompts = format_prompts(
        prompts=prompts, target_lengths=[-1] * len(prompts), len_rewards_config=None, tokenizer=tokenizer, enable_thinking=enable_thinking
    )

    match tokenizer.name_or_path:
        case "deepseek-ai/DeepSeek-R1-0528":
            assert formatted_prompts == [
                "<｜User｜>What is the capital of France?<｜Assistant｜>",
                "<｜User｜>Explain quantum mechanics<｜Assistant｜>",
            ]
        case "Qwen/QwQ-32B":
            expected_formatted_prompts = [
                "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n",
                "<|im_start|>user\nExplain quantum mechanics<|im_end|>\n<|im_start|>assistant\n<think>\n",
            ]
            # Does not support "not thinking"
            assert formatted_prompts == expected_formatted_prompts
        case "Qwen/Qwen3-0.6B":
            expected_formatted_prompts = [
                "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nExplain quantum mechanics<|im_end|>\n<|im_start|>assistant\n",
            ]
            if not enable_thinking:  # If thinking disabled, will add closed thinking tag
                expected_formatted_prompts = [x + "<think>\n\n</think>\n\n" for x in expected_formatted_prompts]
            assert formatted_prompts == expected_formatted_prompts
        case _:
            raise ValueError(f"Unknown model: {tokenizer.name_or_path}")


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_format_prompts_with_length_rewards(
    prompts: list[str], length_rewards_config: LenRewardsConfig, tokenizer: AutoTokenizer, enable_thinking: bool
):
    formatted_prompts = format_prompts(
        prompts=prompts,
        target_lengths=[100, 200],
        len_rewards_config=length_rewards_config,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
    )
    print(formatted_prompts)

    match tokenizer.name_or_path:
        case "deepseek-ai/DeepSeek-R1-0528":
            if length_rewards_config.length_prompt_location == "system_prompt":
                expected_formatted_prompts = [
                    "Think for 100 tokens before giving a response.<｜User｜>What is the capital of France?<｜Assistant｜>",
                    "Think for 200 tokens before giving a response.<｜User｜>Explain quantum mechanics<｜Assistant｜>",
                ]
            else:
                expected_formatted_prompts = [
                    "<｜User｜>What is the capital of France? Think for 100 tokens before giving a response.<｜Assistant｜>",
                    "<｜User｜>Explain quantum mechanics Think for 200 tokens before giving a response.<｜Assistant｜>",
                ]
            assert formatted_prompts == expected_formatted_prompts
        case "Qwen/QwQ-32B":
            if length_rewards_config.length_prompt_location == "system_prompt":
                expected_formatted_prompts = [
                    "<|im_start|>system\nThink for 100 tokens before giving a response.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n",
                    "<|im_start|>system\nThink for 200 tokens before giving a response.<|im_end|>\n<|im_start|>user\nExplain quantum mechanics<|im_end|>\n<|im_start|>assistant\n<think>\n",
                ]
            else:
                expected_formatted_prompts = [
                    "<|im_start|>user\nWhat is the capital of France? Think for 100 tokens before giving a response.<|im_end|>\n<|im_start|>assistant\n<think>\n",
                    "<|im_start|>user\nExplain quantum mechanics Think for 200 tokens before giving a response.<|im_end|>\n<|im_start|>assistant\n<think>\n",
                ]
            # Does not support "not thinking"
            assert formatted_prompts == expected_formatted_prompts
        case "Qwen/Qwen3-0.6B":
            if length_rewards_config.length_prompt_location == "system_prompt":
                expected_formatted_prompts = [
                    "<|im_start|>system\nThink for 100 tokens before giving a response.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
                    "<|im_start|>system\nThink for 200 tokens before giving a response.<|im_end|>\n<|im_start|>user\nExplain quantum mechanics<|im_end|>\n<|im_start|>assistant\n",
                ]
            else:
                expected_formatted_prompts = [
                    "<|im_start|>user\nWhat is the capital of France? Think for 100 tokens before giving a response.<|im_end|>\n<|im_start|>assistant\n",
                    "<|im_start|>user\nExplain quantum mechanics Think for 200 tokens before giving a response.<|im_end|>\n<|im_start|>assistant\n",
                ]
            if not enable_thinking:  # If thinking disabled, will add closed thinking tag
                expected_formatted_prompts = [x + "<think>\n\n</think>\n\n" for x in expected_formatted_prompts]
            assert formatted_prompts == expected_formatted_prompts
        case _:
            raise ValueError(f"Unknown model: {tokenizer.name_or_path}")
