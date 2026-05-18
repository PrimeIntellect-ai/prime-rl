import torch
from torch import nn

from prime_rl.trainer.models.glm_moe_dsa import sparse_mla_attention
from prime_rl.trainer.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from prime_rl.trainer.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaModel
from prime_rl.trainer.models.glm_moe_dsa.sparse_mla_attention import GlmMoeDsaAttention, SparseMlaAttentionArgs


def _attention_args(use_index_cache: bool, skip_topk: bool = False) -> SparseMlaAttentionArgs:
    return SparseMlaAttentionArgs(
        hidden_size=4,
        num_attention_heads=1,
        kv_lora_rank=2,
        q_lora_rank=2,
        qk_rope_head_dim=1,
        qk_nope_head_dim=1,
        qk_head_dim=2,
        v_head_dim=2,
        attention_bias=False,
        rms_norm_eps=1e-5,
        index_n_heads=1,
        index_head_dim=2,
        index_topk=64,
        use_index_cache=use_index_cache,
        skip_topk=skip_topk,
    )


def _stub_attention(monkeypatch, attention: GlmMoeDsaAttention, computed_indices: torch.Tensor) -> dict[str, torch.Tensor]:
    captured = {}

    class FakeSparseMLA:
        @staticmethod
        def apply(sparse_q, sparse_kv, indices, scaling):
            captured["indices"] = indices
            return sparse_q

    def compute_sparse_indices(**kwargs):
        return computed_indices

    monkeypatch.setattr(sparse_mla_attention, "_SparseMLA", FakeSparseMLA)
    monkeypatch.setattr(
        attention,
        "attn_projections",
        lambda hidden_states: (
            torch.zeros(1, hidden_states.shape[1], 2),
            torch.zeros(1, hidden_states.shape[1], 2),
            torch.zeros(1, hidden_states.shape[1], 1),
        ),
    )
    monkeypatch.setattr(
        attention,
        "mla_up_proj",
        lambda **kwargs: (
            torch.zeros(1, computed_indices.shape[1], 1, 2),
            torch.zeros(1, computed_indices.shape[1] + 1, 1, 2),
            torch.zeros(1, 2, 2),
        ),
    )
    monkeypatch.setattr(attention, "output_proj", lambda attn_output, w_v: attn_output)
    monkeypatch.setattr(attention.indexer, "compute_sparse_indices", compute_sparse_indices)

    return captured


def test_attention_does_not_return_indices_when_index_cache_disabled(monkeypatch):
    attention = GlmMoeDsaAttention(_attention_args(use_index_cache=False))
    computed_indices = torch.ones(1, 3, 1, 64, dtype=torch.int32)
    captured = _stub_attention(monkeypatch, attention, computed_indices)

    _, returned_indices = attention(
        hidden_states=torch.zeros(1, 3, 4),
        position_embeddings=(torch.zeros(1, 3, 1), torch.zeros(1, 3, 1)),
        ks=torch.arange(3, dtype=torch.int32),
        ke=torch.arange(1, 4, dtype=torch.int32),
    )

    assert captured["indices"] is computed_indices
    assert returned_indices is None


def test_attention_returns_indices_when_index_cache_enabled(monkeypatch):
    attention = GlmMoeDsaAttention(_attention_args(use_index_cache=True))
    computed_indices = torch.ones(1, 3, 1, 64, dtype=torch.int32)
    _stub_attention(monkeypatch, attention, computed_indices)

    _, returned_indices = attention(
        hidden_states=torch.zeros(1, 3, 4),
        position_embeddings=(torch.zeros(1, 3, 1), torch.zeros(1, 3, 1)),
        ks=torch.arange(3, dtype=torch.int32),
        ke=torch.arange(1, 4, dtype=torch.int32),
    )

    assert returned_indices is computed_indices


def _tiny_config(use_index_cache: bool) -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=2,
        kv_lora_rank=2,
        q_lora_rank=2,
        qk_rope_head_dim=1,
        qk_nope_head_dim=1,
        v_head_dim=2,
        first_k_dense_replace=2,
        index_n_heads=1,
        index_head_dim=2,
        index_topk=64,
        max_position_embeddings=16,
        pad_token_id=0,
        use_index_cache=use_index_cache,
    )


class RecordingLayer(nn.Module):
    def __init__(self, next_cached_indices: torch.Tensor):
        super().__init__()
        self.next_cached_indices = next_cached_indices
        self.seen_cached_indices = []

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        ks=None,
        ke=None,
        cached_indices=None,
        routed_experts=None,
    ):
        self.seen_cached_indices.append(cached_indices)
        return hidden_states, self.next_cached_indices


def _run_model_with_recording_layers(use_index_cache: bool) -> list[RecordingLayer]:
    model = GlmMoeDsaModel(_tiny_config(use_index_cache=use_index_cache))
    layers = [
        RecordingLayer(torch.ones(1, 3, 1, 64, dtype=torch.int32)),
        RecordingLayer(torch.full((1, 3, 1, 64), 2, dtype=torch.int32)),
    ]
    model.layers = nn.ModuleList(layers)

    model(
        inputs_embeds=torch.zeros(1, 3, 4),
        position_ids=torch.arange(3).unsqueeze(0),
    )

    return layers


def test_model_does_not_thread_indices_when_index_cache_disabled():
    layers = _run_model_with_recording_layers(use_index_cache=False)

    assert layers[0].seen_cached_indices == [None]
    assert layers[1].seen_cached_indices == [None]


def test_model_threads_indices_when_index_cache_enabled():
    layers = _run_model_with_recording_layers(use_index_cache=True)

    assert layers[0].seen_cached_indices == [None]
    assert layers[1].seen_cached_indices[0] is layers[0].next_cached_indices
