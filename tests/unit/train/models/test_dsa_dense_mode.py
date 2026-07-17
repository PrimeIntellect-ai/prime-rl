"""GPU correctness checks for the DSA dense-attention fallback (DSA conversion Phase 1).

`SparseMlaAttention`'s dense mode (`use_sparse_attn=False`) decompresses `kv_b_proj`
directly (real per-head K/V) instead of folding it into Q via the "weight absorption"
trick the sparse kernel path uses. These two computations should be mathematically
equivalent whenever the sparse path's top-k covers every causally-valid key — that's
the strongest correctness signal available without a second, independent reference
implementation, so we pin it down here.
"""

import pytest
import torch

from prime_rl.trainer.models.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from prime_rl.trainer.models.layers.dsa import compute_indexer_kl_loss
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="DSA kernels (fp8_indexer, sparse_mla_fwd) require CUDA")


def _tiny_config(**overrides) -> GlmMoeDsaConfig:
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
        # kv_lora_rank/qk_rope_head_dim are NOT free: sparse_mla_fwd_interface hardcodes
        # d_v=512 and asserts dim_plus_tail_dim==576, so the sparse kernel only runs at
        # exactly these two values today (a pre-existing constraint, not new here).
        kv_lora_rank=512,
        q_lora_rank=64,
        qk_rope_head_dim=64,
        v_head_dim=32,
        qk_nope_head_dim=16,
        num_experts_per_tok=2,
        first_k_dense_replace=2,  # all layers dense MLP: isolates the attention change
        max_position_embeddings=256,
        index_n_heads=2,
        # index_head_dim must exceed qk_rope_head_dim (Indexer slices off the first
        # qk_rope_head_dim entries as the RoPE'd part) and be >=32 (fp8_indexer's tl.dot
        # requires K=index_head_dim >= 32) -- 128 is GLM-5's own real default.
        index_head_dim=128,
        index_topk=64,
        use_grouped_mm=False,
        pad_token_id=0,
    )
    kwargs.update(overrides)
    return GlmMoeDsaConfig(**kwargs)


def _build_model(config: GlmMoeDsaConfig, dtype: torch.dtype = torch.float32) -> GlmMoeDsaForCausalLM:
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = GlmMoeDsaForCausalLM(config).to(dtype)
    inject_prime_lm_head(model)
    return model


@requires_cuda
def test_dense_mode_matches_sparse_mode_when_topk_covers_full_sequence():
    # The sparse tilelang kernel is compiled for bfloat16 inputs only, so this comparison
    # inherently carries bf16 rounding error across two layers -- tolerances are loose
    # accordingly; a tight match would only be meaningful in the same dtype as sparse mode.
    seq_len = 64  # == index_topk, so sparse top-k selects every causally-valid key
    sparse_config = _tiny_config(use_sparse_attn=True)
    model = _build_model(sparse_config, dtype=torch.bfloat16)

    input_ids = torch.randint(0, sparse_config.vocab_size, (1, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    with torch.no_grad():
        sparse_out = model(input_ids=input_ids, position_ids=position_ids)

    for layer in model.model.layers:
        layer.self_attn.use_sparse_attn = False
    with torch.no_grad():
        dense_out = model(input_ids=input_ids, position_ids=position_ids)

    torch.testing.assert_close(sparse_out["logits"].float(), dense_out["logits"].float(), atol=0.1, rtol=0.05)


@requires_cuda
def test_indexer_kl_loss_trains_indexer_toward_dense_attention():
    config = _tiny_config(use_sparse_attn=False, train_indexer=True)
    model = _build_model(config)

    for layer in model.model.layers:
        for param in layer.self_attn.indexer.parameters():
            param.requires_grad = True
        for name, param in model.named_parameters():
            if "indexer" not in name:
                param.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )

    seq_len = 64
    torch.manual_seed(1)
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        model(input_ids=input_ids, position_ids=position_ids)
        loss = compute_indexer_kl_loss(model, coeff=1.0)
        assert loss is not None
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"indexer KL loss did not decrease: {losses}"


@requires_cuda
def test_indexer_kl_loss_is_finite_in_sparse_mode():
    """Stage B (sparse-adaptation): the KL target comes from the actual sparse-selected
    keys instead of the dense O(S^2) distribution. Sanity-check it's finite.

    Forward-only: a full backward here would need to run sparse_mla_bwd (layer 1's
    indexer score depends on layer 0's hidden_states, which depend on layer 0's sparse
    attention output), and that tilelang kernel has a pre-existing minimum-tile-size
    requirement this repo's models never hit at unit-test scale -- confirmed by the same
    assertion firing on the *unmodified* sparse path's backward with no indexer-KL
    involved at all, so it's not something introduced here.
    """
    config = _tiny_config(use_sparse_attn=True, train_indexer=True)
    model = _build_model(config, dtype=torch.bfloat16)

    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)

    with torch.no_grad():
        model(input_ids=input_ids, position_ids=position_ids)
    loss = compute_indexer_kl_loss(model, coeff=1.0)
    assert loss is not None
    assert torch.isfinite(loss)
