import pytest

from prime_rl.configs.inference import VLLMConfig
from prime_rl.utils.parsers import REASONING_PARSER_PATTERNS, TOOL_CALL_PARSER_PATTERNS, resolve_parser


@pytest.mark.parametrize(
    "model_name,expected_parser",
    [
        # DeepSeek
        ("deepseek-ai/DeepSeek-V3.2", "deepseek_v32"),
        ("deepseek-ai/DeepSeek-V3.2-Exp", "deepseek_v32"),
        ("deepseek-ai/DeepSeek-V3.1", "deepseek_v31"),
        ("deepseek-ai/DeepSeek-V3.1-FP8", "deepseek_v31"),
        # GLM-4.5
        ("zai-org/GLM-4.5", "glm45"),
        ("zai-org/GLM-4.5-FP8", "glm45"),
        ("zai-org/GLM-4.5-Base", "glm45"),
        ("zai-org/GLM-4.5-Air", "glm45"),
        ("zai-org/GLM-4.5-Air-FP8", "glm45"),
        ("zai-org/GLM-4.5-Air-Base", "glm45"),
        ("zai-org/GLM-4.5V", "glm45"),
        ("zai-org/GLM-4.5V-FP8", "glm45"),
        # GLM-4.7
        ("zai-org/GLM-4.7", "glm47"),
        ("zai-org/GLM-4.7-FP8", "glm47"),
        ("zai-org/GLM-4.7-Flash", "glm47"),
        # MiniMax M2
        ("MiniMaxAI/MiniMax-M2", "minimax_m2"),
        ("MiniMaxAI/MiniMax-M2.1", "minimax_m2"),
        ("MiniMaxAI/MiniMax-M2.5", "minimax_m2"),
        # INTELLECT-3
        ("PrimeIntellect/INTELLECT-3", "qwen3_coder"),
        ("PrimeIntellect/INTELLECT-3-FP8", "qwen3_coder"),
        ("PrimeIntellect/INTELLECT-3.1", "qwen3_coder"),
        # StepFun
        ("stepfun-ai/Step-3.5-Flash", "step3p5"),
        # Qwen3 dense
        ("Qwen/Qwen3-0.6B", "hermes"),
        ("Qwen/Qwen3-0.6B-Base", "hermes"),
        ("Qwen/Qwen3-0.6B-FP8", "hermes"),
        ("Qwen/Qwen3-1.7B", "hermes"),
        ("Qwen/Qwen3-1.7B-Base", "hermes"),
        ("Qwen/Qwen3-1.7B-FP8", "hermes"),
        ("Qwen/Qwen3-4B", "hermes"),
        ("Qwen/Qwen3-4B-Base", "hermes"),
        ("Qwen/Qwen3-4B-FP8", "hermes"),
        ("Qwen/Qwen3-8B", "hermes"),
        ("Qwen/Qwen3-8B-Base", "hermes"),
        ("Qwen/Qwen3-8B-FP8", "hermes"),
        ("Qwen/Qwen3-14B", "hermes"),
        ("Qwen/Qwen3-14B-Base", "hermes"),
        ("Qwen/Qwen3-14B-FP8", "hermes"),
        ("Qwen/Qwen3-32B", "hermes"),
        ("Qwen/Qwen3-32B-FP8", "hermes"),
        # Qwen3 MoE
        ("Qwen/Qwen3-30B-A3B", "hermes"),
        ("Qwen/Qwen3-30B-A3B-Base", "hermes"),
        ("Qwen/Qwen3-30B-A3B-FP8", "hermes"),
        ("Qwen/Qwen3-235B-A22B", "hermes"),
        ("Qwen/Qwen3-235B-A22B-FP8", "hermes"),
        # Qwen3 2507
        ("Qwen/Qwen3-4B-Instruct-2507", "hermes"),
        ("Qwen/Qwen3-4B-Thinking-2507", "hermes"),
        ("Qwen/Qwen3-4B-Instruct-2507-FP8", "hermes"),
        ("Qwen/Qwen3-4B-Thinking-2507-FP8", "hermes"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "hermes"),
        ("Qwen/Qwen3-30B-A3B-Thinking-2507", "hermes"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", "hermes"),
        ("Qwen/Qwen3-30B-A3B-Thinking-2507-FP8", "hermes"),
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", "hermes"),
        ("Qwen/Qwen3-235B-A22B-Thinking-2507", "hermes"),
        ("Qwen/Qwen3-235B-A22B-Instruct-2507-FP8", "hermes"),
        ("Qwen/Qwen3-235B-A22B-Thinking-2507-FP8", "hermes"),
        # Qwen3-Next
        ("Qwen/Qwen3-Next-80B-A3B-Instruct", "hermes"),
        ("Qwen/Qwen3-Next-80B-A3B-Thinking", "hermes"),
        ("Qwen/Qwen3-Next-80B-A3B-Instruct-FP8", "hermes"),
        ("Qwen/Qwen3-Next-80B-A3B-Thinking-FP8", "hermes"),
        # Qwen3-Coder
        ("Qwen/Qwen3-Coder-480B-A35B-Instruct", "hermes"),
        ("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", "hermes"),
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "hermes"),
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "hermes"),
        # Qwen3-Coder-Next
        ("Qwen/Qwen3-Coder-Next", "hermes"),
        ("Qwen/Qwen3-Coder-Next-Base", "hermes"),
        ("Qwen/Qwen3-Coder-Next-FP8", "hermes"),
        # Qwen3.5 dense
        ("Qwen/Qwen3.5-0.8B", "qwen3_coder"),
        ("Qwen/Qwen3.5-0.8B-Base", "qwen3_coder"),
        ("Qwen/Qwen3.5-2B", "qwen3_coder"),
        ("Qwen/Qwen3.5-2B-Base", "qwen3_coder"),
        ("Qwen/Qwen3.5-4B", "qwen3_coder"),
        ("Qwen/Qwen3.5-4B-Base", "qwen3_coder"),
        ("Qwen/Qwen3.5-9B", "qwen3_coder"),
        ("Qwen/Qwen3.5-9B-Base", "qwen3_coder"),
        ("Qwen/Qwen3.5-27B", "qwen3_coder"),
        ("Qwen/Qwen3.5-27B-FP8", "qwen3_coder"),
        # Qwen3.5 MoE
        ("Qwen/Qwen3.5-35B-A3B", "qwen3_coder"),
        ("Qwen/Qwen3.5-35B-A3B-Base", "qwen3_coder"),
        ("Qwen/Qwen3.5-35B-A3B-FP8", "qwen3_coder"),
        ("Qwen/Qwen3.5-122B-A10B", "qwen3_coder"),
        ("Qwen/Qwen3.5-122B-A10B-FP8", "qwen3_coder"),
        ("Qwen/Qwen3.5-397B-A17B", "qwen3_coder"),
        ("Qwen/Qwen3.5-397B-A17B-FP8", "qwen3_coder"),
    ],
)
def test_auto_detect_tool_call_parser(model_name: str, expected_parser: str):
    assert resolve_parser(model_name, TOOL_CALL_PARSER_PATTERNS) == expected_parser


@pytest.mark.parametrize(
    "model_name,expected_parser",
    [
        # GLM
        ("zai-org/GLM-4.5", "glm45"),
        ("zai-org/GLM-4.5-Air", "glm45"),
        ("zai-org/GLM-4.7", "glm45"),
        ("zai-org/GLM-4.7-Flash", "glm45"),
        # MiniMax M2
        ("MiniMaxAI/MiniMax-M2", "minimax_m2"),
        ("MiniMaxAI/MiniMax-M2.1", "minimax_m2"),
        ("MiniMaxAI/MiniMax-M2.5", "minimax_m2"),
        # INTELLECT-3
        ("PrimeIntellect/INTELLECT-3", "deepseek_r1"),
        ("PrimeIntellect/INTELLECT-3-FP8", "deepseek_r1"),
        ("PrimeIntellect/INTELLECT-3.1", "deepseek_r1"),
        # StepFun
        ("stepfun-ai/Step-3.5-Flash", "step3p5"),
        # Qwen3 Thinking → deepseek_r1
        ("Qwen/Qwen3-4B-Thinking-2507", "deepseek_r1"),
        ("Qwen/Qwen3-30B-A3B-Thinking-2507", "deepseek_r1"),
        ("Qwen/Qwen3-235B-A22B-Thinking-2507", "deepseek_r1"),
        # Qwen3 (non-thinking) → qwen3
        ("Qwen/Qwen3-0.6B", "qwen3"),
        ("Qwen/Qwen3-4B-Instruct-2507", "qwen3"),
        ("Qwen/Qwen3-235B-A22B", "qwen3"),
        ("Qwen/Qwen3-Coder-Next", "qwen3"),
        # Qwen3.5
        ("Qwen/Qwen3.5-27B", "qwen3"),
        ("Qwen/Qwen3.5-397B-A17B", "qwen3"),
    ],
)
def test_auto_detect_reasoning_parser(model_name: str, expected_parser: str | None):
    assert resolve_parser(model_name, REASONING_PARSER_PATTERNS) == expected_parser


def test_explicit_parser_not_overridden():
    config = VLLMConfig(model="Qwen/Qwen3-4B", tool_call_parser="qwen3_xml")
    assert config.tool_call_parser == "qwen3_xml"


def test_unknown_model_returns_none():
    assert resolve_parser("some/unknown-model", TOOL_CALL_PARSER_PATTERNS) is None


def test_validator_auto_resolves_when_not_set():
    config = VLLMConfig(model="Qwen/Qwen3-4B")
    assert config.tool_call_parser == "hermes"
    assert config.reasoning_parser == "qwen3"
    assert config.enable_auto_tool_choice is True


def test_validator_skips_when_explicitly_set_to_none():
    config = VLLMConfig(model="Qwen/Qwen3-4B", tool_call_parser=None)
    assert config.tool_call_parser is None


def test_validator_skips_when_auto_tool_choice_disabled():
    config = VLLMConfig(model="Qwen/Qwen3-4B", enable_auto_tool_choice=False)
    assert config.tool_call_parser is None


def test_reasoning_parser_auto_resolves():
    config = VLLMConfig(model="Qwen/Qwen3.5-27B")
    assert config.reasoning_parser == "qwen3"


def test_reasoning_parser_explicit_not_overridden():
    config = VLLMConfig(model="Qwen/Qwen3.5-27B", reasoning_parser="deepseek_r1")
    assert config.reasoning_parser == "deepseek_r1"
