import pytest

from prime_rl.configs.inference import VLLMConfig
from prime_rl.utils.parsers import resolve_reasoning_parser, resolve_tool_call_parser

# (model_name, expected_tool_call_parser, expected_reasoning_parser)
EXPECTED_PARSERS: list[tuple[str, str | None, str | None]] = [
    # DeepSeek
    ("deepseek-ai/DeepSeek-V3.2", "deepseek_v32", None),
    ("deepseek-ai/DeepSeek-V3.2-Exp", "deepseek_v32", None),
    ("deepseek-ai/DeepSeek-V3.1", "deepseek_v31", None),
    ("deepseek-ai/DeepSeek-V3.1-FP8", "deepseek_v31", None),
    # GLM-4.5
    ("zai-org/GLM-4.5", "glm45", "glm45"),
    ("zai-org/GLM-4.5-FP8", "glm45", "glm45"),
    ("zai-org/GLM-4.5-Base", "glm45", "glm45"),
    ("zai-org/GLM-4.5-Air", "glm45", "glm45"),
    ("zai-org/GLM-4.5-Air-FP8", "glm45", "glm45"),
    ("zai-org/GLM-4.5-Air-Base", "glm45", "glm45"),
    ("zai-org/GLM-4.5V", "glm45", "glm45"),
    ("zai-org/GLM-4.5V-FP8", "glm45", "glm45"),
    # GLM-4.7
    ("zai-org/GLM-4.7", "glm47", "glm45"),
    ("zai-org/GLM-4.7-FP8", "glm47", "glm45"),
    ("zai-org/GLM-4.7-Flash", "glm47", "glm45"),
    # MiniMax M2
    ("MiniMaxAI/MiniMax-M2", "minimax_m2", "minimax_m2"),
    ("MiniMaxAI/MiniMax-M2.1", "minimax_m2", "minimax_m2"),
    ("MiniMaxAI/MiniMax-M2.5", "minimax_m2", "minimax_m2"),
    # INTELLECT-3
    ("PrimeIntellect/INTELLECT-3", "qwen3_coder", "deepseek_r1"),
    ("PrimeIntellect/INTELLECT-3-FP8", "qwen3_coder", "deepseek_r1"),
    ("PrimeIntellect/INTELLECT-3.1", "qwen3_coder", "deepseek_r1"),
    # StepFun
    ("stepfun-ai/Step-3.5-Flash", "step3p5", "step3p5"),
    # Qwen3 dense
    ("Qwen/Qwen3-0.6B", "hermes", None),
    ("Qwen/Qwen3-0.6B-Base", "hermes", None),
    ("Qwen/Qwen3-0.6B-FP8", "hermes", None),
    ("Qwen/Qwen3-1.7B", "hermes", None),
    ("Qwen/Qwen3-1.7B-Base", "hermes", None),
    ("Qwen/Qwen3-1.7B-FP8", "hermes", None),
    ("Qwen/Qwen3-4B", "hermes", None),
    ("Qwen/Qwen3-4B-Base", "hermes", None),
    ("Qwen/Qwen3-4B-FP8", "hermes", None),
    ("Qwen/Qwen3-8B", "hermes", None),
    ("Qwen/Qwen3-8B-Base", "hermes", None),
    ("Qwen/Qwen3-8B-FP8", "hermes", None),
    ("Qwen/Qwen3-14B", "hermes", None),
    ("Qwen/Qwen3-14B-Base", "hermes", None),
    ("Qwen/Qwen3-14B-FP8", "hermes", None),
    ("Qwen/Qwen3-32B", "hermes", None),
    ("Qwen/Qwen3-32B-FP8", "hermes", None),
    # Qwen3 MoE
    ("Qwen/Qwen3-30B-A3B", "hermes", None),
    ("Qwen/Qwen3-30B-A3B-Base", "hermes", None),
    ("Qwen/Qwen3-30B-A3B-FP8", "hermes", None),
    ("Qwen/Qwen3-235B-A22B", "hermes", None),
    ("Qwen/Qwen3-235B-A22B-FP8", "hermes", None),
    # Qwen3 2507
    ("Qwen/Qwen3-4B-Instruct-2507", "hermes", None),
    ("Qwen/Qwen3-4B-Thinking-2507", "hermes", "deepseek_r1"),
    ("Qwen/Qwen3-4B-Instruct-2507-FP8", "hermes", None),
    ("Qwen/Qwen3-4B-Thinking-2507-FP8", "hermes", "deepseek_r1"),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", "hermes", None),
    ("Qwen/Qwen3-30B-A3B-Thinking-2507", "hermes", "deepseek_r1"),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "hermes", None),
    ("Qwen/Qwen3-30B-A3B-Thinking-2507-FP8", "hermes", "deepseek_r1"),
    ("Qwen/Qwen3-235B-A22B-Instruct-2507", "hermes", None),
    ("Qwen/Qwen3-235B-A22B-Thinking-2507", "hermes", "deepseek_r1"),
    ("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "hermes", None),
    ("Qwen/Qwen3-235B-A22B-Thinking-2507-FP8", "hermes", "deepseek_r1"),
    # Qwen3-Next
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", "hermes", None),
    ("Qwen/Qwen3-Next-80B-A3B-Thinking", "hermes", "deepseek_r1"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8", "hermes", None),
    ("Qwen/Qwen3-Next-80B-A3B-Thinking-FP8", "hermes", "deepseek_r1"),
    # Qwen3-Coder
    ("Qwen/Qwen3-Coder-480B-A35B-Instruct", "hermes", None),
    ("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", "hermes", None),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "hermes", None),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "hermes", None),
    # Qwen3-Coder-Next
    ("Qwen/Qwen3-Coder-Next", "hermes", None),
    ("Qwen/Qwen3-Coder-Next-Base", "hermes", None),
    ("Qwen/Qwen3-Coder-Next-FP8", "hermes", None),
    # Qwen3.5 dense
    ("Qwen/Qwen3.5-0.8B", "qwen3_coder", None),
    ("Qwen/Qwen3.5-0.8B-Base", "qwen3_coder", None),
    ("Qwen/Qwen3.5-2B", "qwen3_coder", None),
    ("Qwen/Qwen3.5-2B-Base", "qwen3_coder", None),
    ("Qwen/Qwen3.5-4B", "qwen3_coder", None),
    ("Qwen/Qwen3.5-4B-Base", "qwen3_coder", None),
    ("Qwen/Qwen3.5-9B", "qwen3_coder", None),
    ("Qwen/Qwen3.5-9B-Base", "qwen3_coder", None),
    ("Qwen/Qwen3.5-27B", "qwen3_coder", None),
    ("Qwen/Qwen3.5-27B-FP8", "qwen3_coder", None),
    # Qwen3.5 MoE
    ("Qwen/Qwen3.5-35B-A3B", "qwen3_coder", None),
    ("Qwen/Qwen3.5-35B-A3B-Base", "qwen3_coder", None),
    ("Qwen/Qwen3.5-35B-A3B-FP8", "qwen3_coder", None),
    ("Qwen/Qwen3.5-122B-A10B", "qwen3_coder", None),
    ("Qwen/Qwen3.5-122B-A10B-FP8", "qwen3_coder", None),
    ("Qwen/Qwen3.5-397B-A17B", "qwen3_coder", None),
    ("Qwen/Qwen3.5-397B-A17B-FP8", "qwen3_coder", None),
    # Unknown
    ("some/unknown-model", None, None),
]


@pytest.mark.parametrize("model_name,expected_tool_call,expected_reasoning", EXPECTED_PARSERS)
def test_resolve_tool_call_parser(model_name: str, expected_tool_call: str | None, expected_reasoning: str | None):
    assert resolve_tool_call_parser(model_name) == expected_tool_call


@pytest.mark.parametrize("model_name,expected_tool_call,expected_reasoning", EXPECTED_PARSERS)
def test_resolve_reasoning_parser(model_name: str, expected_tool_call: str | None, expected_reasoning: str | None):
    assert resolve_reasoning_parser(model_name) == expected_reasoning


@pytest.mark.parametrize("model_name,expected_tool_call,expected_reasoning", EXPECTED_PARSERS)
def test_auto_resolve_parsers(model_name: str, expected_tool_call: str | None, expected_reasoning: str | None):
    config = VLLMConfig(model=model_name)
    assert config.enable_auto_tool_choice is True
    assert config.tool_call_parser == expected_tool_call
    assert config.reasoning_parser == expected_reasoning


def test_explicit_parser_not_overridden():
    config = VLLMConfig(model="Qwen/Qwen3-4B", tool_call_parser="my_parser")
    assert config.tool_call_parser == "my_parser"


def test_validator_skips_when_explicitly_set_to_none():
    config = VLLMConfig(model="Qwen/Qwen3-4B", tool_call_parser=None)
    assert config.tool_call_parser is None


def test_validator_skips_when_auto_tool_choice_disabled():
    config = VLLMConfig(model="Qwen/Qwen3-4B", enable_auto_tool_choice=False)
    assert config.tool_call_parser is None


def test_reasoning_parser_auto_resolves():
    config = VLLMConfig(model="Qwen/Qwen3-4B-Thinking-2507")
    assert config.reasoning_parser == "deepseek_r1"


def test_reasoning_parser_explicit_not_overridden():
    config = VLLMConfig(model="Qwen/Qwen3.5-27B", reasoning_parser="deepseek_r1")
    assert config.reasoning_parser == "deepseek_r1"
