"""Mock equivalence: declarative chain == legacy imperative converter for minimax_m2."""

import torch

from prime_rl.trainer.models.conversion_ops import apply_hf_to_tt, apply_tt_to_hf
from prime_rl.trainer.models.minimax_m2.conversion_chain import build_minimax_m2_chain
from prime_rl.trainer.models.minimax_m2.converting_minimax_m2 import convert_hf_to_tt_moe, convert_tt_to_hf_moe

from .conftest import assert_state_dicts_equal, clone

NUM_LAYERS, E, MOE, DIM = 3, 4, 6, 8


def _mock_hf() -> dict:
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {}
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(DIM, DIM)  # untouched passthrough
        sd[f"{p}.block_sparse_moe.gate.weight"] = torch.randn(E, DIM)
        sd[f"{p}.block_sparse_moe.e_score_correction_bias"] = torch.randn(E)
        for e in range(E):
            sd[f"{p}.block_sparse_moe.experts.{e}.w1.weight"] = torch.randn(MOE, DIM)
            sd[f"{p}.block_sparse_moe.experts.{e}.w2.weight"] = torch.randn(DIM, MOE)
            sd[f"{p}.block_sparse_moe.experts.{e}.w3.weight"] = torch.randn(MOE, DIM)
    return sd


def test_minimax_m2_forward_matches():
    hf = _mock_hf()
    chain = build_minimax_m2_chain(NUM_LAYERS)
    imperative = clone(hf)
    convert_hf_to_tt_moe(imperative)
    declarative = apply_hf_to_tt(clone(hf), chain)
    assert_state_dicts_equal(imperative, declarative, "minimax_m2 hf->tt")


def test_minimax_m2_backward_matches():
    hf = _mock_hf()
    chain = build_minimax_m2_chain(NUM_LAYERS)
    tt = clone(hf)
    convert_hf_to_tt_moe(tt)
    imperative = clone(tt)
    convert_tt_to_hf_moe(imperative)
    declarative = apply_tt_to_hf(clone(tt), chain)
    assert_state_dicts_equal(imperative, declarative, "minimax_m2 tt->hf")


def test_minimax_m2_roundtrip():
    hf = _mock_hf()
    chain = build_minimax_m2_chain(NUM_LAYERS)
    back = apply_tt_to_hf(apply_hf_to_tt(clone(hf), chain), chain)
    assert_state_dicts_equal(hf, back, "minimax_m2 roundtrip")
