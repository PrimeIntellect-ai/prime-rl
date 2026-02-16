import pytest
import torch

from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]


def test_glm_moe_dsa_sparse_attention() -> None:
    """Test sparse MLA attention with tilelang kernels (fwd + bwd).

    Requires tilelang + GPU. Uses production-scale MLA dims
    (kv_lora_rank=512, qk_rope_head_dim=64) since the tilelang
    kernels hardcode dim_plus_tail_dim=576 and D_V=512.
    """
    try:
        from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface  # noqa: F401
    except ImportError:
        pytest.skip("tilelang not available")

    from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention

    config = GlmMoeDsaConfig(
        vocab_size=1024,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=32,
        kv_lora_rank=512,
        q_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        max_position_embeddings=4096,
        rope_interleave=True,
        index_topk=64,
        use_grouped_mm=False,
    )

    seq_len = 128
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        attn = GlmMoeDsaAttention(config)
        hidden_states = torch.randn(1, seq_len, config.hidden_size, requires_grad=True)
        position_ids = torch.arange(seq_len).unsqueeze(0)

    from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig

    rotary_config = RotaryEmbeddingConfig(
        max_position_embeddings=config.max_position_embeddings,
        rope_type="default",
        model_config=config,
    )
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        rotary_emb = RotaryEmbedding(rotary_config)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    out, _ = attn(hidden_states, position_embeddings, cu_seqlens=cu_seqlens, max_seqlen=seq_len)

    assert out.shape == (1, seq_len, config.hidden_size), (
        f"Expected (1, {seq_len}, {config.hidden_size}), got {out.shape}"
    )

    out.sum().backward()

    assert hidden_states.grad is not None, "hidden_states should have gradients"
    assert hidden_states.grad.abs().sum() > 0, "hidden_states gradients should be non-zero"

    grad_params = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
    for name, param in attn.named_parameters():
        short = name.split(".")[-1] if "." in name else name
        if short in grad_params or name.rstrip(".weight").rstrip(".bias") in grad_params:
            if "weight" in name:
                assert param.grad is not None, f"{name} should have gradients"
                assert param.grad.abs().sum() > 0, f"{name} gradients should be non-zero"

    no_grad_params = ["indexer.wq_b", "indexer.wk", "indexer.k_norm", "indexer.weights_proj"]
    for name, param in attn.named_parameters():
        if any(name.startswith(param_prefix) for param_prefix in no_grad_params):
            assert param.grad is None or param.grad.abs().sum() == 0, (
                f"Indexer param {name} should have no gradient (topk is discrete)"
            )


def test_glm_moe_dsa_sparse_attention_varlen() -> None:
    """Test sparse MLA with packed sequences (varlen) — two sequences concatenated."""
    try:
        from prime_rl.trainer.models.kernels.sparse_mla_fwd import sparse_mla_fwd_interface  # noqa: F401
    except ImportError:
        pytest.skip("tilelang not available")

    from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention

    config = GlmMoeDsaConfig(
        vocab_size=1024,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=32,
        kv_lora_rank=512,
        q_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        max_position_embeddings=4096,
        rope_interleave=True,
        index_topk=64,
        use_grouped_mm=False,
    )

    seq1_len, seq2_len = 80, 96
    total = seq1_len + seq2_len
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        attn = GlmMoeDsaAttention(config)
        hidden_states = torch.randn(1, total, config.hidden_size, requires_grad=True)
        position_ids = torch.cat([torch.arange(seq1_len), torch.arange(seq2_len)]).unsqueeze(0)

    from prime_rl.trainer.models.layers.rotary_emb import RotaryEmbedding, RotaryEmbeddingConfig

    rotary_config = RotaryEmbeddingConfig(
        max_position_embeddings=config.max_position_embeddings,
        rope_type="default",
        model_config=config,
    )
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        rotary_emb = RotaryEmbedding(rotary_config)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    cu_seqlens = torch.tensor([0, seq1_len, total], dtype=torch.int32, device="cuda")

    out, _ = attn(hidden_states, position_embeddings, cu_seqlens=cu_seqlens, max_seqlen=max(seq1_len, seq2_len))

    assert out.shape == (1, total, config.hidden_size)

    out.sum().backward()
    assert hidden_states.grad is not None
    assert hidden_states.grad.abs().sum() > 0
