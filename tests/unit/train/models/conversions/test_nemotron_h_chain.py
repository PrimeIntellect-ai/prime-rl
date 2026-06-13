"""Mock equivalence: declarative chain == legacy imperative converter for NemotronH.

The router-bias shift is intentionally lossy, so only forward and backward are
checked against the imperative converter (no strict roundtrip)."""

import torch

from prime_rl.trainer.models.conversion_ops import apply_hf_to_tt, apply_tt_to_hf
from prime_rl.trainer.models.nemotron_h.conversion_chain import build_nemotron_h_chain
from prime_rl.trainer.models.nemotron_h.converting_nemotron_h import convert_hf_to_prime, convert_prime_to_hf

from .conftest import assert_state_dicts_equal, clone

LAYERS = ["mamba", "attention", "moe"]
E, MOE, DIM = 4, 6, 8


def _mock_hf() -> dict:
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {}
    sd["backbone.embeddings.weight"] = torch.randn(16, DIM)
    sd["backbone.norm_f.weight"] = torch.randn(DIM)
    sd["mtp.layers.0.weight"] = torch.randn(DIM, DIM)  # dropped on the way to prime
    for i, lt in enumerate(LAYERS):
        p = f"backbone.layers.{i}"
        sd[f"{p}.input_layernorm.weight"] = torch.randn(DIM)  # passthrough (not under mixer.)
        if lt == "mamba":
            sd[f"{p}.mixer.in_proj.weight"] = torch.randn(2 * DIM, DIM)
            sd[f"{p}.mixer.A_log"] = torch.randn(DIM)
        elif lt == "attention":
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                sd[f"{p}.mixer.{proj}.weight"] = torch.randn(DIM, DIM)
        elif lt == "moe":
            sd[f"{p}.mixer.gate.weight"] = torch.randn(E, DIM)
            sd[f"{p}.mixer.gate.e_score_correction_bias"] = torch.randn(E) + 50.0
            for e in range(E):
                sd[f"{p}.mixer.experts.{e}.up_proj.weight"] = torch.randn(MOE, DIM)
                sd[f"{p}.mixer.experts.{e}.down_proj.weight"] = torch.randn(DIM, MOE)
            sd[f"{p}.mixer.shared_experts.up_proj.weight"] = torch.randn(MOE, DIM)
            sd[f"{p}.mixer.shared_experts.down_proj.weight"] = torch.randn(DIM, MOE)
            sd[f"{p}.mixer.fc1_latent_proj.weight"] = torch.randn(DIM, DIM)
            sd[f"{p}.mixer.fc2_latent_proj.weight"] = torch.randn(DIM, DIM)
    return sd


def test_nemotron_h_forward_matches():
    hf = _mock_hf()
    chain = build_nemotron_h_chain(LAYERS)
    imperative = clone(hf)
    convert_hf_to_prime(imperative, LAYERS)
    declarative = apply_hf_to_tt(clone(hf), chain)
    assert_state_dicts_equal(imperative, declarative, "nemotron_h hf->prime")


def test_nemotron_h_backward_matches():
    hf = _mock_hf()
    chain = build_nemotron_h_chain(LAYERS)
    tt = clone(hf)
    convert_hf_to_prime(tt, LAYERS)  # produce a valid prime state dict
    imperative = clone(tt)
    convert_prime_to_hf(imperative, LAYERS)
    declarative = apply_tt_to_hf(clone(tt), chain)
    assert_state_dicts_equal(imperative, declarative, "nemotron_h prime->hf")
