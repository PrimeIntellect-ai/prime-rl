"""Mock equivalence: declarative chain == legacy imperative converter for qwen3_moe."""

import torch

from prime_rl.trainer.models.qwen3_moe.conversion_chain import build_qwen3_moe_chain
from prime_rl.trainer.models.conversion_ops import apply_hf_to_tt, apply_tt_to_hf
from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import convert_hf_to_tt_moe, convert_tt_to_hf_moe

from .conftest import assert_state_dicts_equal, clone

NUM_LAYERS, E, MOE, DIM = 3, 4, 6, 8


def _mock_hf(per_expert: bool = True) -> dict:
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {}
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(DIM, DIM)  # untouched passthrough
        sd[f"{p}.mlp.gate.weight"] = torch.randn(E, DIM)
        if per_expert:
            for e in range(E):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(MOE, DIM)
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(DIM, MOE)
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(MOE, DIM)
        else:
            sd[f"{p}.mlp.experts.gate_up_proj"] = torch.randn(E, 2 * MOE, DIM)
            sd[f"{p}.mlp.experts.down_proj"] = torch.randn(E, DIM, MOE)
    return sd


def test_qwen3_moe_forward_matches(per_expert=True):
    hf = _mock_hf(per_expert)
    chain = build_qwen3_moe_chain(NUM_LAYERS)
    imperative = clone(hf)
    convert_hf_to_tt_moe(imperative)
    declarative = apply_hf_to_tt(clone(hf), chain)
    assert_state_dicts_equal(imperative, declarative, "qwen3_moe hf->tt (per_expert)")


def test_qwen3_moe_forward_matches_fused():
    test_qwen3_moe_forward_matches(per_expert=False)


def test_qwen3_moe_backward_matches():
    hf = _mock_hf(per_expert=True)
    chain = build_qwen3_moe_chain(NUM_LAYERS)
    tt = clone(hf)
    convert_hf_to_tt_moe(tt)
    imperative = clone(tt)
    convert_tt_to_hf_moe(imperative)
    declarative = apply_tt_to_hf(clone(tt), chain)
    assert_state_dicts_equal(imperative, declarative, "qwen3_moe tt->hf")


def test_qwen3_moe_roundtrip():
    hf = _mock_hf(per_expert=True)
    chain = build_qwen3_moe_chain(NUM_LAYERS)
    back = apply_tt_to_hf(apply_hf_to_tt(clone(hf), chain), chain)
    assert_state_dicts_equal(hf, back, "qwen3_moe roundtrip")
