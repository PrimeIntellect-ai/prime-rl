import pytest

from prime_rl.inference.config import ModelConfig


@pytest.mark.parametrize(
    "model_name,expected_parser",
    [
        # GLM-4.5
        ("zai-org/GLM-4.5", "glm45"),
        ("zai-org/GLM-4.5-Air", "glm45"),
        ("zai-org/GLM-4.5V", "glm45"),
        # GLM-4.7
        ("zai-org/GLM-4.7", "glm47"),
        ("zai-org/GLM-4.7-Flash", "glm47"),
        # MiniMax
        ("MiniMaxAI/MiniMax-M2", "minimax_m2"),
        ("MiniMaxAI/MiniMax-M2.1", "minimax_m2"),
        ("MiniMaxAI/MiniMax-M2.5", "minimax_m2"),
        # INTELLECT
        ("PrimeIntellect/INTELLECT-3", "hermes"),
        ("PrimeIntellect/INTELLECT-3-FP8", "hermes"),
        ("PrimeIntellect/INTELLECT-3.1", "hermes"),
        # Qwen3
        ("Qwen/Qwen3-0.6B", "hermes"),
        ("Qwen/Qwen3-32B", "hermes"),
        ("Qwen/Qwen3-235B-A22B", "hermes"),
        ("Qwen/Qwen3-4B-Instruct-2507", "hermes"),
        ("Qwen/Qwen3-Coder-480B-A35B-Instruct", "hermes"),
        ("Qwen/Qwen3-Next-80B-A3B-Instruct", "hermes"),
        ("Qwen/Qwen3.5-397B-A17B", "hermes"),
    ],
)
def test_auto_detect_tool_call_parser(model_name: str, expected_parser: str):
    config = ModelConfig(name=model_name)
    assert config.tool_call_parser == expected_parser


def test_explicit_parser_overrides_auto_detect():
    config = ModelConfig(name="Qwen/Qwen3-4B-Instruct-2507", tool_call_parser="qwen3_xml")
    assert config.tool_call_parser == "qwen3_xml"
