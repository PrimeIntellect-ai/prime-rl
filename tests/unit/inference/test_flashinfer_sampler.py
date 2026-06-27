import inspect

from prime_rl.inference.vllm import flashinfer_sampler


def test_precompile_tail_batch_sizes_falls_back_to_single_batch(monkeypatch):
    monkeypatch.delenv(
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES",
        raising=False,
    )
    monkeypatch.setenv("PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCH", "128")

    assert flashinfer_sampler._precompile_tail_batch_sizes() == [128]


def test_precompile_tail_batch_sizes_parses_unique_list(monkeypatch):
    monkeypatch.setenv(
        "PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCHES",
        "1, 128,256,128",
    )
    monkeypatch.setenv("PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_BATCH", "64")

    assert flashinfer_sampler._precompile_tail_batch_sizes() == [1, 128, 256]


def test_precompile_tail_top_p_values_does_not_duplicate_runtime_float32(monkeypatch):
    monkeypatch.setenv("PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_P", "0.95")

    assert flashinfer_sampler._precompile_tail_top_p_values() == [0.95]


def test_precompile_tail_top_p_values_dedupe_exact_float32(monkeypatch):
    monkeypatch.setenv("PRIME_RL_FINITE_TOPK_SAMPLED_LOGPROB_PRECOMPILE_TOP_P", "1.0")

    assert flashinfer_sampler._precompile_tail_top_p_values() == [1.0]


def test_triton_tail_specializes_only_on_block_shape():
    signature = inspect.signature(flashinfer_sampler._k_tail_uniform_kernel.fn)

    assert "TOP_P" not in signature.parameters
    assert signature.parameters["K"].annotation is inspect.Signature.empty
    assert signature.parameters["top_p"].annotation is inspect.Signature.empty
    assert signature.parameters["K_BLOCK"].annotation == "tl.constexpr"
    assert flashinfer_sampler._k_tail_uniform_kernel.constexprs == [7]
