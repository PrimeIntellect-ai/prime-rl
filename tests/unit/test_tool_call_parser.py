import asyncio
import json
from types import SimpleNamespace

import pytest

from prime_rl.inference.vllm.server import liveness, resolve_tool_call_parser


class ResponsiveEngine:
    async def collective_rpc(self, method):
        assert method == "liveness_probe"
        return []


class HangingEngine:
    async def collective_rpc(self, method):
        assert method == "liveness_probe"
        await asyncio.sleep(60)


def request_with_engine(engine):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(engine_client=engine)),
    )


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
        # GLM-5.1
        ("zai-org/GLM-5.1", "glm47"),
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
        ("Qwen/Qwen3.5-397B-A17B", "qwen3_coder"),
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


def test_liveness_returns_ok_when_engine_responds():
    response = asyncio.run(liveness(request_with_engine(ResponsiveEngine())))

    assert response == {"status": "ok"}


def test_liveness_returns_503_when_engine_times_out(monkeypatch):
    monkeypatch.setattr("prime_rl.inference.vllm.server.LIVENESS_TIMEOUT_SECONDS", 0.001)

    response = asyncio.run(liveness(request_with_engine(HangingEngine())))

    assert response.status_code == 503
    assert json.loads(response.body) == {"status": "engine_unresponsive"}
