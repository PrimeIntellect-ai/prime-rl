"""Mock equivalence: declarative chain == legacy imperative converter for laguna."""

import torch

from prime_rl.trainer.models.conversion_ops import apply_hf_to_tt, apply_tt_to_hf
from prime_rl.trainer.models.laguna.conversion_chain import build_laguna_chain
from prime_rl.trainer.models.laguna.converting_laguna import convert_hf_to_prime, convert_prime_to_hf

from .conftest import assert_state_dicts_equal, clone

NUM_LAYERS, E, MOE, DIM, SHARED = 3, 4, 6, 8, 5


def _mock_hf(per_expert: bool = True, shared_plural: bool = False, bias_on_gate: bool = False) -> dict:
    """Build a mock HF state dict exercising the laguna input variants.

    * ``per_expert``: per-expert routed weights vs. the fused gate_up layout.
    * ``shared_plural``: ``mlp.shared_experts.*`` vs. the singular form.
    * ``bias_on_gate``: expert bias on ``mlp.gate.*`` vs. ``mlp.experts.*``.
    """
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {}
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_proj.weight"] = torch.randn(DIM, DIM)  # untouched passthrough
        sd[f"{p}.mlp.gate.weight"] = torch.randn(E, DIM)

        bias_key = "gate" if bias_on_gate else "experts"
        sd[f"{p}.mlp.{bias_key}.e_score_correction_bias"] = torch.randn(E)

        if per_expert:
            for e in range(E):
                sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(MOE, DIM)
                sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = torch.randn(DIM, MOE)
                sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = torch.randn(MOE, DIM)
        else:
            sd[f"{p}.mlp.experts.gate_up_proj"] = torch.randn(E, 2 * MOE, DIM)
            sd[f"{p}.mlp.experts.down_proj"] = torch.randn(E, DIM, MOE)

        shared = "shared_experts" if shared_plural else "shared_expert"
        sd[f"{p}.mlp.{shared}.gate_proj.weight"] = torch.randn(SHARED, DIM)
        sd[f"{p}.mlp.{shared}.down_proj.weight"] = torch.randn(DIM, SHARED)
        sd[f"{p}.mlp.{shared}.up_proj.weight"] = torch.randn(SHARED, DIM)
    return sd


def _add_prime_buffers(tt: dict) -> dict:
    """Prime-only runtime buffer that must be dropped on tt->hf."""
    for i in range(NUM_LAYERS):
        tt[f"model.layers.{i}.mlp.tokens_per_expert"] = torch.zeros(E, dtype=torch.long)
    return tt


def test_laguna_forward_matches(per_expert=True, shared_plural=False, bias_on_gate=False):
    hf = _mock_hf(per_expert, shared_plural, bias_on_gate)
    chain = build_laguna_chain(NUM_LAYERS)
    imperative = clone(hf)
    convert_hf_to_prime(imperative)
    declarative = apply_hf_to_tt(clone(hf), chain)
    assert_state_dicts_equal(imperative, declarative, "laguna hf->tt")


def test_laguna_forward_matches_fused():
    test_laguna_forward_matches(per_expert=False)


def test_laguna_forward_matches_shared_plural():
    test_laguna_forward_matches(shared_plural=True)


def test_laguna_forward_matches_bias_on_gate():
    test_laguna_forward_matches(bias_on_gate=True)


def test_laguna_backward_matches():
    hf = _mock_hf(per_expert=True)
    chain = build_laguna_chain(NUM_LAYERS)
    tt = clone(hf)
    convert_hf_to_prime(tt)
    tt = _add_prime_buffers(tt)
    imperative = clone(tt)
    convert_prime_to_hf(imperative)
    declarative = apply_tt_to_hf(clone(tt), chain)
    assert_state_dicts_equal(imperative, declarative, "laguna tt->hf")


def test_laguna_roundtrip():
    # Canonical inputs (per-expert, singular shared, bias on experts) roundtrip
    # losslessly; the fused / plural / gate-bias inputs are normalized to the
    # canonical HF form on the way back, so they are not byte-for-byte invertible.
    hf = _mock_hf(per_expert=True, shared_plural=False, bias_on_gate=False)
    chain = build_laguna_chain(NUM_LAYERS)
    back = apply_tt_to_hf(apply_hf_to_tt(clone(hf), chain), chain)
    assert_state_dicts_equal(hf, back, "laguna roundtrip")
