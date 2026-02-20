import pytest

from prime_rl.inference.config import ModelConfig


@pytest.mark.parametrize(
    "model_name,expected_parser",
    [
        # GLM
        ("zai-org/GLM-4.5", "glm45"),
        ("THUDM/GLM-4.5-Air", "glm45"),
        ("zai-org/GLM-4.7", "glm47"),
        ("zai-org/GLM-4.7-Flash", "glm47"),
        # MiniMax
        ("MiniMaxAI/MiniMax-M2", "minimax_m2"),
        ("MiniMaxAI/MiniMax-M2.1", "minimax_m2"),
        # INTELLECT
        ("PrimeIntellect/INTELLECT-3", "hermes"),
        ("PrimeIntellect/INTELLECT-3.1", "hermes"),
        # Qwen3
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", "hermes"),
        ("Qwen/Qwen3-4B-Instruct-2507", "hermes"),
        ("Qwen/Qwen3-Coder-8B-Instruct", "hermes"),
        ("Qwen/Qwen3-0.6B", "hermes"),
    ],
)
def test_auto_detect_tool_call_parser(model_name: str, expected_parser: str):
    config = ModelConfig(name=model_name)
    assert config.tool_call_parser == expected_parser


def test_explicit_parser_overrides_auto_detect():
    config = ModelConfig(name="Qwen/Qwen3-4B-Instruct-2507", tool_call_parser="qwen3_xml")
    assert config.tool_call_parser == "qwen3_xml"


def test_unknown_model_leaves_parser_none():
    config = ModelConfig(name="unknown-org/mystery-model-7B")
    assert config.tool_call_parser is None
