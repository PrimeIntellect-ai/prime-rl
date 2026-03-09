import pytest

from prime_rl.utils.parsers import resolve_tool_call_parser


# Every model that was explicitly listed in the old dict-based mapping.
# This ensures the regex patterns don't regress.
@pytest.mark.parametrize(
    "model_name,expected_parser",
    [
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
        ("PrimeIntellect/INTELLECT-3", "hermes"),
        ("PrimeIntellect/INTELLECT-3-FP8", "hermes"),
        ("PrimeIntellect/INTELLECT-3.1", "hermes"),
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
    assert resolve_tool_call_parser(model_name, None) == expected_parser


def test_explicit_parser_not_overridden():
    assert resolve_tool_call_parser("Qwen/Qwen3-4B-Instruct-2507", "qwen3_xml") == "qwen3_xml"


def test_unknown_model_returns_none():
    assert resolve_tool_call_parser("some/unknown-model", None) is None
