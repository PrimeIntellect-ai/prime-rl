"""Mock equivalence: declarative chain == legacy imperative converter for afmoe."""

import torch

from prime_rl.trainer.models.afmoe.conversion_chain import build_afmoe_chain
from prime_rl.trainer.models.afmoe.converting_afmoe import convert_hf_to_tt_moe, convert_tt_to_hf_moe
from prime_rl.trainer.models.conversion_ops import apply_hf_to_tt, apply_tt_to_hf

from .conftest import assert_state_dicts_equal, clone

NUM_LAYERS, E, MOE, DIM, SHARED = 3, 4, 6, 8, 5
DENSE_LAYER = 1  # exercise the present-guards: this layer has no experts


def _mock_hf() -> dict:
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {}
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(DIM, DIM)  # untouched passthrough
        sd[f"{p}.mlp.router.weight"] = torch.randn(E, DIM)  # router: shared name, not renamed
        if i == DENSE_LAYER:
            # Dense layer: a plain MLP, no experts -> ops must no-op.
            sd[f"{p}.mlp.gate_proj.weight"] = torch.randn(MOE, DIM)
            sd[f"{p}.mlp.down_proj.weight"] = torch.randn(DIM, MOE)
            sd[f"{p}.mlp.up_proj.weight"] = torch.randn(MOE, DIM)
            continue
        sd[f"{p}.mlp.shared_experts.gate_proj.weight"] = torch.randn(SHARED, DIM)
        sd[f"{p}.mlp.shared_experts.down_proj.weight"] = torch.randn(DIM, SHARED)
        sd[f"{p}.mlp.shared_experts.up_proj.weight"] = torch.randn(SHARED, DIM)
        for e in range(E):
            sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(MOE, DIM)
            sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(DIM, MOE)
            sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(MOE, DIM)
    return sd


def _add_prime_buffers(tt: dict) -> dict:
    """Prime-only runtime buffers that must be dropped on tt->hf."""
    for i in range(NUM_LAYERS):
        if i == DENSE_LAYER:
            continue
        p = f"model.layers.{i}.mlp"
        tt[f"{p}.tokens_per_expert"] = torch.zeros(E, dtype=torch.long)
        tt[f"{p}.reorderer.indices"] = torch.zeros(E, dtype=torch.long)
    return tt


def test_afmoe_forward_matches():
    hf = _mock_hf()
    chain = build_afmoe_chain(NUM_LAYERS)
    imperative = clone(hf)
    convert_hf_to_tt_moe(imperative)
    declarative = apply_hf_to_tt(clone(hf), chain)
    assert_state_dicts_equal(imperative, declarative, "afmoe hf->tt")


def test_afmoe_backward_matches():
    hf = _mock_hf()
    chain = build_afmoe_chain(NUM_LAYERS)
    tt = clone(hf)
    convert_hf_to_tt_moe(tt)
    tt = _add_prime_buffers(tt)
    imperative = clone(tt)
    convert_tt_to_hf_moe(imperative)
    declarative = apply_tt_to_hf(clone(tt), chain)
    assert_state_dicts_equal(imperative, declarative, "afmoe tt->hf")


def test_afmoe_roundtrip():
    hf = _mock_hf()
    chain = build_afmoe_chain(NUM_LAYERS)
    back = apply_tt_to_hf(apply_hf_to_tt(clone(hf), chain), chain)
    assert_state_dicts_equal(hf, back, "afmoe roundtrip")
