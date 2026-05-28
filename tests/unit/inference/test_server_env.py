import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.inference.server import setup_vllm_env


def test_setup_vllm_env_disables_deep_gemm_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_MOE_USE_DEEP_GEMM", raising=False)

    setup_vllm_env(InferenceConfig(use_deep_gemm=False))

    assert os.environ["VLLM_USE_DEEP_GEMM"] == "0"
    assert os.environ["VLLM_MOE_USE_DEEP_GEMM"] == "0"


def test_setup_vllm_env_enables_deep_gemm_when_requested(monkeypatch):
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_MOE_USE_DEEP_GEMM", raising=False)

    setup_vllm_env(InferenceConfig(use_deep_gemm=True))

    assert os.environ["VLLM_USE_DEEP_GEMM"] == "1"
    assert os.environ["VLLM_MOE_USE_DEEP_GEMM"] == "1"
