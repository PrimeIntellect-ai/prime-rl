import copy

import torch
from torch import nn

from prime_rl.trainer.models.fusions import (
    apply_model_fusions,
    get_model_packed_parameters,
    write_back_loaded_packed_parameters,
)
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.moe import GroupedExperts
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

NUM_EXPERTS, DIM, HIDDEN = 4, 16, 24
NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 4, 2, 8


def build_experts(expert_type="gated"):
    experts = GroupedExperts(dim=DIM, hidden_dim=HIDDEN, num_experts=NUM_EXPERTS, expert_type=expert_type)
    for parameter in experts.parameters():
        nn.init.normal_(parameter)
    return experts


def build_attention(attention_bias=True):
    config = AttentionConfig(
        hidden_size=DIM,
        head_dim=HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        is_causal=True,
        attention_bias=attention_bias,
        use_qk_norm=False,
        rms_norm_eps=1e-6,
    )
    attention = FlashAttention(config)
    for parameter in attention.parameters():
        nn.init.normal_(parameter)
    return attention


def assert_state_dict_stays_canonical(unfused, fusion):
    """Packing a module must not change what its state dict contains, or how it loads."""
    fused = copy.deepcopy(unfused)
    apply_model_fusions(fused, [fusion])
    expected = unfused.state_dict()

    assert set(fused.state_dict()) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(fused.state_dict()[name], tensor), name

    reloaded = copy.deepcopy(unfused)
    apply_model_fusions(reloaded, [fusion])
    reloaded.load_state_dict(expected)
    for packed_info, reloaded_info in zip(get_model_packed_parameters(fused), get_model_packed_parameters(reloaded)):
        assert torch.equal(reloaded_info.parameter, packed_info.parameter), packed_info.fqn
    return fused


def test_gate_up_packs_the_output_dimension():
    fused = assert_state_dict_stays_canonical(build_experts(), "gate_up")

    (packed_info,) = get_model_packed_parameters(fused)
    assert packed_info.fqn == "gate_up_proj"
    assert packed_info.logical_fqns == ("gate_proj", "up_proj")
    assert packed_info.parameter.shape == (NUM_EXPERTS, 2 * HIDDEN, DIM)
    assert packed_info.spec.muon_matrix_partitions(packed_info.parameter) == (HIDDEN, HIDDEN)
    assert fused.gate_proj is None and fused.up_proj is None


def test_qkv_packs_weights_and_biases():
    fused = assert_state_dict_stays_canonical(build_attention(), "qkv")

    weight, bias = get_model_packed_parameters(fused)
    query_size, key_size = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
    assert weight.fqn == "qkv_proj.weight"
    assert weight.logical_fqns == ("q_proj.weight", "k_proj.weight", "v_proj.weight")
    assert weight.parameter.shape == (query_size + 2 * key_size, DIM)
    assert weight.spec.muon_matrix_partitions(weight.parameter) == (query_size, key_size, key_size)

    assert bias.fqn == "qkv_proj.bias"
    assert bias.logical_fqns == ("q_proj.bias", "k_proj.bias", "v_proj.bias")
    # A one-dimensional parameter has no output dimension to partition
    assert bias.spec.muon_matrix_partitions(bias.parameter) is None


def test_qkv_without_bias_packs_only_weights():
    fused = assert_state_dict_stays_canonical(build_attention(attention_bias=False), "qkv")
    assert [packed_info.fqn for packed_info in get_model_packed_parameters(fused)] == ["qkv_proj.weight"]


def test_qkv_projections_match_the_unpacked_ones():
    unfused = build_attention()
    fused = copy.deepcopy(unfused)
    apply_model_fusions(fused, ["qkv"])

    hidden_states = torch.randn(1, 8, DIM)
    for fused_projection, unfused_projection in zip(
        fused.project_qkv(hidden_states), unfused.project_qkv(hidden_states)
    ):
        assert torch.equal(fused_projection, unfused_projection)


def test_unsupported_fusions_are_skipped():
    assert apply_model_fusions(build_experts(), ["qkv"]) == {}
    assert apply_model_fusions(build_experts(expert_type="non_gated"), ["gate_up"]) == {}


def test_write_back_of_a_partial_state_dict_leaves_other_packed_parameters_unchanged():
    model = nn.ModuleDict({"experts": build_experts(), "attention": build_attention(attention_bias=False)})
    apply_model_fusions(model, ["gate_up", "qkv"])
    qkv_before = model.attention.qkv_proj.weight.clone()
    state_dict = {
        fqn: torch.zeros_like(tensor) for fqn, tensor in model.state_dict().items() if fqn.startswith("experts.")
    }

    write_back_loaded_packed_parameters(model, state_dict)

    assert torch.equal(model.experts.gate_up_proj, torch.zeros_like(model.experts.gate_up_proj))
    assert torch.equal(model.attention.qkv_proj.weight, qkv_before)


def test_frozen_weight_keys_include_fused_weights():
    config = Qwen3MoeConfig(
        hidden_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        vocab_size=128,
    )
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM._from_config(config)
    apply_model_fusions(model, ["gate_up", "qkv"])
    model.model.embed_tokens.requires_grad_(False)
    model.model.layers[0].requires_grad_(False)

    assert model.frozen_weight_keys() == {
        key for key in model.state_dict() if key.startswith(("model.embed_tokens.", "model.layers.0."))
    }
