import pytest

from prime_rl.inference.vllm.server import resolve_tool_call_parser


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
        # Qwen3-VL (prefix-based)
        ("Qwen/Qwen3-VL-4B-Instruct", "hermes"),
        ("Qwen/Qwen3-VL-8B-Instruct", "hermes"),
        ("Qwen/Qwen3-VL-72B-Instruct", "hermes"),
    ],
)
def test_auto_detect_tool_call_parser(model_name: str, expected_parser: str):
    assert resolve_tool_call_parser(model_name, "auto") == expected_parser


def test_explicit_parser_not_overridden():
    assert resolve_tool_call_parser("Qwen/Qwen3-4B-Instruct-2507", "qwen3_xml") == "qwen3_xml"


def test_auto_unknown_model():
    assert resolve_tool_call_parser("some/unknown-model", "auto") is None


def test_none_skips_resolution():
    assert resolve_tool_call_parser("Qwen/Qwen3-0.6B", None) is None
