import copy

import torch
from torch import nn

from prime_rl.trainer.models.fusions import apply_model_fusions, get_model_packed_parameters
from prime_rl.trainer.models.layers.attn import AttentionConfig, FlashAttention
from prime_rl.trainer.models.layers.moe import GroupedExperts

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


def test_kda_qkv_packs_qkv_and_fgb_canonically():
    from prime_rl.trainer.models.glm5_next.linear_attention import KimiDeltaAttention

    kda = KimiDeltaAttention(
        hidden_size=DIM,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        conv_kernel_size=4,
        gate_lower_bound=-5.0,
        norm_eps=1e-6,
    )
    for parameter in kda.parameters():
        nn.init.normal_(parameter)

    fused = assert_state_dict_stays_canonical(kda, "qkv")
    packed = {info.fqn: info for info in get_model_packed_parameters(fused)}
    assert set(packed) == {"qkv_proj.weight", "fgb_proj.weight"}
    assert packed["qkv_proj.weight"].logical_fqns == ("q_proj.weight", "k_proj.weight", "v_proj.weight")
    assert packed["fgb_proj.weight"].logical_fqns == ("f_a_proj.weight", "g_a_proj.weight", "b_proj.weight")
    assert fused.q_proj.weight is None and fused.f_a_proj.weight is None and fused.b_proj.weight is None


def test_sparse_mla_qkv_packs_q_a_and_kv_a_canonically():
    from prime_rl.trainer.models.glm5_next.sparse_attention import SparseMlaAttention

    mla = SparseMlaAttention(
        hidden_size=DIM,
        num_heads=NUM_HEADS,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        norm_eps=1e-6,
        index_num_heads=2,
        index_head_dim=128,
        index_token_budget=16,
        index_pool_size=4,
        dsa_backend="tilelang",
    )
    for parameter in mla.parameters():
        nn.init.normal_(parameter)

    fused = assert_state_dict_stays_canonical(mla, "qkv")
    (packed_info,) = get_model_packed_parameters(fused)
    assert packed_info.fqn == "qkv_a_proj.weight"
    assert packed_info.logical_fqns == ("q_a_proj.weight", "kv_a_proj_with_mqa.weight")
    assert fused.q_a_proj.weight is None and fused.kv_a_proj_with_mqa.weight is None
