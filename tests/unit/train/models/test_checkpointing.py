import torch
import torch.nn as nn

from prime_rl.trainer.models.layers.checkpointing import (
    get_supported_targets,
    set_selective_activation_checkpointing,
)

_PATCHED_METHODS_ATTR = "_prime_rl_selective_ac_patched_methods"


class DummySelfAttention(nn.Module):
    def attn_projections(self, hidden_states, position_embeddings=None):
        return hidden_states

    def output_proj(self, attn_output):
        return attn_output

    def forward(self, hidden_states):
        return hidden_states


class DummySlidingAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummySelfAttention()
        self.attention_type = "sliding_attention"


class DummyMamba(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class DummyMambaLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mamba = DummyMamba()


class DummyMoEMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("tokens_per_expert", torch.zeros(1), persistent=False)

    def forward(self, hidden_states):
        return hidden_states

    def run_experts_for_torch_a2a(self, hidden_states, *args):
        return hidden_states

    def run_experts_for_deepep(self, hidden_states, **expert_kwargs):
        return hidden_states


class DummyMoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = DummyMoEMlp()


def test_get_supported_targets_treats_mamba_as_linear_attention():
    assert get_supported_targets(DummyMambaLayer()) == frozenset({"norm", "linear_attn"})


def test_sliding_attention_linear_attn_subsumes_attn_proj_hooks():
    layer = DummySlidingAttentionLayer()

    set_selective_activation_checkpointing(layer, ["attn_proj", "linear_attn"])

    assert getattr(layer.self_attn, _PATCHED_METHODS_ATTR) == frozenset({"forward"})


def test_routed_experts_checkpointing_patches_local_and_global_helpers():
    layer = DummyMoELayer()

    assert "routed_experts" in get_supported_targets(layer)

    set_selective_activation_checkpointing(layer, ["routed_experts"])

    assert getattr(layer.mlp, _PATCHED_METHODS_ATTR) == frozenset(
        {"run_experts_for_torch_a2a", "run_experts_for_deepep"}
    )
