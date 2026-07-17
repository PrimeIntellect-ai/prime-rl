"""GPU correctness check that `layers/dsa.py`'s DSA machinery generalizes to a second,
materially different family (different MLA hyperparameters, different MoE shape) beyond
`glm_moe_dsa` — see `test_dsa_dense_mode.py` for the detailed dense/sparse equivalence
rationale, which this mirrors for Kimi K2.
"""

import pytest
import torch

from prime_rl.trainer.models.kimi_k2_dsa import KimiK2DsaConfig, KimiK2DsaForCausalLM
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DSA kernels (fp8_indexer, sparse_mla_fwd) require CUDA"
)


def _tiny_config(**overrides) -> KimiK2DsaConfig:
    kwargs = dict(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        n_group=1,
        topk_group=1,
        # kv_lora_rank/qk_rope_head_dim are NOT free -- see test_dsa_dense_mode.py's note;
        # the sparse tilelang kernel only runs at exactly these two values today.
        kv_lora_rank=512,
        q_lora_rank=64,
        qk_rope_head_dim=64,
        v_head_dim=32,
        qk_nope_head_dim=16,
        num_experts_per_tok=2,
        first_k_dense_replace=2,
        max_position_embeddings=256,
        index_n_heads=2,
        index_head_dim=128,
        index_topk=64,
        use_grouped_mm=False,
        pad_token_id=0,
    )
    kwargs.update(overrides)
    return KimiK2DsaConfig(**kwargs)


def _build_model(config: KimiK2DsaConfig, dtype: torch.dtype = torch.bfloat16) -> KimiK2DsaForCausalLM:
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = KimiK2DsaForCausalLM(config).to(dtype)
    inject_prime_lm_head(model)
    return model


@requires_cuda
def test_kimi_dense_mode_matches_sparse_mode_when_topk_covers_full_sequence():
    seq_len = 64  # == index_topk, so sparse top-k selects every causally-valid key
    config = _tiny_config(use_sparse_attn=True)
    model = _build_model(config)

    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    with torch.no_grad():
        sparse_out = model(input_ids=input_ids, position_ids=position_ids)

    for layer in model.model.layers:
        layer.self_attn.use_sparse_attn = False
    with torch.no_grad():
        dense_out = model(input_ids=input_ids, position_ids=position_ids)

    torch.testing.assert_close(sparse_out["logits"].float(), dense_out["logits"].float(), atol=0.1, rtol=0.05)
