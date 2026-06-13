"""Mock equivalence: declarative chain == legacy imperative converter for glm_moe_dsa.

GLM-MoE-DSA's hf<->tt MoE conversion matches GLM-4 MoE; the MLA (DSA) attention
weights pass through untouched, which the mock exercises explicitly.
"""

import torch

from prime_rl.trainer.models.conversion_ops import apply_hf_to_tt, apply_tt_to_hf
from prime_rl.trainer.models.glm_moe_dsa.conversion_chain import build_glm_moe_dsa_chain
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_hf_to_tt_moe, convert_tt_to_hf_moe

from .conftest import assert_state_dicts_equal, clone

NUM_LAYERS, E, MOE, DIM = 3, 4, 6, 8


def _mock_hf(per_expert: bool = True) -> dict:
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {}
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        # MLA / DSA attention weights — must pass through untouched.
        sd[f"{p}.self_attn.q_a_proj.weight"] = torch.randn(DIM, DIM)
        sd[f"{p}.self_attn.kv_a_proj_with_mqa.weight"] = torch.randn(DIM, DIM)
        sd[f"{p}.self_attn.q_b_proj.weight"] = torch.randn(DIM, DIM)
        sd[f"{p}.self_attn.indexer.wk.weight"] = torch.randn(DIM, DIM)
        sd[f"{p}.mlp.gate.weight"] = torch.randn(E, DIM)
        sd[f"{p}.mlp.gate.e_score_correction_bias"] = torch.randn(E)
        sd[f"{p}.mlp.shared_experts.gate_proj.weight"] = torch.randn(MOE, DIM)
        sd[f"{p}.mlp.shared_experts.down_proj.weight"] = torch.randn(DIM, MOE)
        sd[f"{p}.mlp.shared_experts.up_proj.weight"] = torch.randn(MOE, DIM)
        if per_expert:
            for e in range(E):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(MOE, DIM)
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(DIM, MOE)
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(MOE, DIM)
        else:
            sd[f"{p}.mlp.experts.gate_up_proj"] = torch.randn(E, 2 * MOE, DIM)
            sd[f"{p}.mlp.experts.down_proj"] = torch.randn(E, DIM, MOE)
    return sd


def test_glm_moe_dsa_forward_matches(per_expert=True):
    hf = _mock_hf(per_expert)
    chain = build_glm_moe_dsa_chain(NUM_LAYERS)
    imperative = clone(hf)
    convert_hf_to_tt_moe(imperative)
    declarative = apply_hf_to_tt(clone(hf), chain)
    assert_state_dicts_equal(imperative, declarative, "glm_moe_dsa hf->tt (per_expert)")


def test_glm_moe_dsa_forward_matches_fused():
    test_glm_moe_dsa_forward_matches(per_expert=False)


def test_glm_moe_dsa_backward_matches():
    hf = _mock_hf(per_expert=True)
    chain = build_glm_moe_dsa_chain(NUM_LAYERS)
    tt = clone(hf)
    convert_hf_to_tt_moe(tt)
    imperative = clone(tt)
    convert_tt_to_hf_moe(imperative)
    declarative = apply_tt_to_hf(clone(tt), chain)
    assert_state_dicts_equal(imperative, declarative, "glm_moe_dsa tt->hf")


def test_glm_moe_dsa_backward_matches_with_leading_singleton():
    tt = clone(_mock_hf(per_expert=True))
    convert_hf_to_tt_moe(tt)
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}.mlp.shared_expert"
        for wn in ("w1", "w2", "w3"):
            tt[f"{p}.{wn}"] = tt[f"{p}.{wn}"].unsqueeze(0)  # (1, ...)
        tt[f"model.layers.{i}.mlp.tokens_per_expert"] = torch.zeros(E)
    chain = build_glm_moe_dsa_chain(NUM_LAYERS)
    imperative = clone(tt)
    convert_tt_to_hf_moe(imperative)
    declarative = apply_tt_to_hf(clone(tt), chain)
    assert_state_dicts_equal(imperative, declarative, "glm_moe_dsa tt->hf (leading singleton)")


def test_glm_moe_dsa_roundtrip():
    hf = _mock_hf(per_expert=True)
    chain = build_glm_moe_dsa_chain(NUM_LAYERS)
    back = apply_tt_to_hf(apply_hf_to_tt(clone(hf), chain), chain)
    assert_state_dicts_equal(hf, back, "glm_moe_dsa roundtrip")
