import pytest
from pydantic import ValidationError

from prime_rl.inference.config import ModelConfig


@pytest.mark.parametrize(
    "model_name,expected_parser",
    [
        # Qwen3
        ("Qwen/Qwen3-235B-A22B-Instruct-2507", "hermes"),
        ("Qwen/Qwen3-235B-A22B-Thinking-2507", "hermes"),
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "hermes"),
        ("Qwen/Qwen3-4B-Instruct-2507", "hermes"),
        ("Qwen/Qwen3-4B-Thinking-2507", "hermes"),
        ("Qwen/Qwen3-VL-4B-Instruct", "hermes"),
        ("PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT", "hermes"),
        # Qwen3 Coder
        ("Qwen/Qwen3-Coder-480B-A35B-Instruct", "qwen3_coder"),
        ("Qwen/Qwen3-Coder-8B-Instruct", "qwen3_coder"),
        # Qwen2.5
        ("Qwen/Qwen2.5-72B-Instruct", "hermes"),
        ("Qwen/Qwen2.5-Coder-32B-Instruct", "qwen3_coder"),
        # GLM
        ("zai-org/GLM-4.5", "glm45"),
        ("THUDM/GLM-4.5-Air", "glm45"),
        ("zai-org/GLM-4.7", "glm47"),
        # INTELLECT-3
        ("PrimeIntellect/INTELLECT-3", "qwen3_coder"),
        # SmolLM3
        ("HuggingFaceTB/SmolLM3-3B", "hermes"),
        # OLMo
        ("allenai/OLMo-3-7B-Instruct", "olmo3"),
        # Llama
        ("meta-llama/Llama-3.2-1B-Instruct", "llama3_json"),
        ("meta-llama/Llama-3.2-3B-Instruct", "llama3_json"),
        ("meta-llama/Llama-4-Scout-17B-16E-Instruct", "llama4_json"),
        # Arcee Trinity
        ("arcee-ai/Trinity-Mini", "hermes"),
        ("arcee-ai/Trinity-Nano-Preview", "hermes"),
        # Nvidia Nemotron
        ("nvidia/OpenReasoning-Nemotron-7B", "hermes"),
        # DeepSeek
        ("deepseek-ai/DeepSeek-V3", "deepseek_v3"),
        ("deepseek-ai/DeepSeek-R1", "deepseek_v3"),
        # Mistral
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
    ],
)
def test_auto_detect_tool_call_parser(model_name: str, expected_parser: str):
    config = ModelConfig(name=model_name)
    assert config.tool_call_parser == expected_parser


def test_explicit_parser_overrides_auto_detect():
    config = ModelConfig(name="Qwen/Qwen3-4B-Instruct-2507", tool_call_parser="qwen3_xml")
    assert config.tool_call_parser == "qwen3_xml"


def test_unknown_model_raises_error():
    with pytest.raises(ValidationError, match="Could not auto-detect tool_call_parser"):
        ModelConfig(name="unknown-org/mystery-model-7B")
