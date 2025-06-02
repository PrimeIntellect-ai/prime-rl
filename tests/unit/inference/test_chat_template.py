import pytest
from transformers import AutoTokenizer

from zeroband.inference.rewards import LenRewardsConfig
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


def test_format_prompts(prompts: list[str], tokenizer: AutoTokenizer):
    """Test format_prompts with no length rewards configuration."""
    formatted_prompts = format_prompts(prompts=prompts, target_lengths=[-1] * len(prompts), len_rewards_config=None, tokenizer=tokenizer)

    match tokenizer.name_or_path:
        case "deepseek-ai/DeepSeek-R1-0528":
            assert formatted_prompts == [
                "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France?<｜Assistant｜>",
                "<｜begin▁of▁sentence｜><｜User｜>Explain quantum mechanics<｜Assistant｜>",
            ]
        case "Qwen/QwQ-32B":
            assert formatted_prompts == [
                "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n",
                "<|im_start|>user\nExplain quantum mechanics<|im_end|>\n<|im_start|>assistant\n<think>\n",
            ]
        case "Qwen/Qwen3-0.6B":
            assert formatted_prompts == [
                "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
                "<|im_start|>user\nExplain quantum mechanics<|im_end|>\n<|im_start|>assistant\n",
            ]
        case _:
            raise ValueError(f"Unknown model: {tokenizer.name_or_path}")


def test_format_prompts_with_length_rewards(prompts: list[str], length_rewards_config: LenRewardsConfig, tokenizer: AutoTokenizer):
    formatted_prompts = format_prompts(
        prompts=prompts, target_lengths=[100, 200], len_rewards_config=length_rewards_config, tokenizer=tokenizer
    )

    match tokenizer.name_or_path:
        case "deepseek-ai/DeepSeek-R1-0528":
            if length_rewards_config.length_prompt_location == "system_prompt":
                assert formatted_prompts == [
                    "<｜begin▁of▁sentence｜>Think for 100 tokens before giving a response.<｜User｜>What is the capital of France?<｜Assistant｜>",
                    "<｜begin▁of▁sentence｜>Think for 200 tokens before giving a response.<｜User｜>Explain quantum mechanics<｜Assistant｜>",
                ]
            else:
                assert formatted_prompts == [
                    "<｜begin▁of▁sentence｜><｜User｜>What is the capital of France? Think for 100 tokens before giving a response.<｜Assistant｜>",
                    "<｜begin▁of▁sentence｜><｜User｜>Explain quantum mechanics Think for 200 tokens before giving a response.<｜Assistant｜>",
                ]
        case "Qwen/QwQ-32B":
            if length_rewards_config.length_prompt_location == "system_prompt":
                assert formatted_prompts == [
                    "<|im_start|>system\nThink for 100 tokens before giving a response.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n",
                    "<|im_start|>system\nThink for 200 tokens before giving a response.<|im_end|>\n<|im_start|>user\nExplain quantum mechanics<|im_end|>\n<|im_start|>assistant\n<think>\n",
                ]
            else:
                assert formatted_prompts == [
                    "<|im_start|>user\nWhat is the capital of France? Think for 100 tokens before giving a response.<|im_end|>\n<|im_start|>assistant\n<think>\n",
                    "<|im_start|>user\nExplain quantum mechanics Think for 200 tokens before giving a response.<|im_end|>\n<|im_start|>assistant\n<think>\n",
                ]
        case "Qwen/Qwen3-0.6B":
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
        case _:
            raise ValueError(f"Unknown model: {tokenizer.name_or_path}")
