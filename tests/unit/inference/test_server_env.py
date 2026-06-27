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


def test_setup_vllm_env_configures_finite_topk_sampler_when_requested(monkeypatch):
    keys = [
        "PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_INTERVAL",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_HIT_LOG_LIMIT",
        "PRIME_RL_LOG_FINITE_TOPK_SAMPLED_LOGPROB_FALLBACK",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TAIL",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_K",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_P",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_VOCAB",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES",
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_BOUNDARY_TIE_GUARD",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    setup_vllm_env(
        InferenceConfig(
            finite_topk_sampled_logprob={
                "enabled": True,
                "tail": "triton",
                "dense_presence": True,
                "stats_interval": 1000,
                "hit_log_limit": 6,
                "log_fallback": True,
                "precompile_tail": True,
                "precompile_top_k": 20,
                "precompile_top_p": 0.95,
                "precompile_vocab": 248320,
                "precompile_batches": [1, 128, 256],
                "boundary_tie_guard": False,
            },
        )
    )

    assert os.environ["PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB"] == "1"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_TAIL"] == "triton"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_DENSE_PRESENCE"] == "1"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_STATS_INTERVAL"] == "1000"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_HIT_LOG_LIMIT"] == "6"
    assert os.environ["PRIME_RL_LOG_FINITE_TOPK_SAMPLED_LOGPROB_FALLBACK"] == "1"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TAIL"] == "1"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_K"] == "20"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_P"] == "0.95"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_VOCAB"] == "248320"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES"] == "1,128,256"
    assert os.environ["PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_BOUNDARY_TIE_GUARD"] == "0"


def test_setup_vllm_env_default_sampler_config_preserves_ambient_env(monkeypatch):
    monkeypatch.setenv("PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB", "1")

    setup_vllm_env(InferenceConfig())

    assert os.environ["PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB"] == "1"


def test_setup_vllm_env_can_force_native_sampler(monkeypatch):
    monkeypatch.setenv("PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB", "1")

    setup_vllm_env(InferenceConfig(finite_topk_sampled_logprob={"enabled": False}))

    assert os.environ["PRIME_RL_ENABLE_FINITE_TOPK_SAMPLED_LOGPROB"] == "0"
