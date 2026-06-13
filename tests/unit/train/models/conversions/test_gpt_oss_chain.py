"""GPT-OSS conversion is the identity — the chain is empty and a no-op."""

import torch

from prime_rl.trainer.models.conversion_ops import apply_hf_to_tt, apply_tt_to_hf
from prime_rl.trainer.models.gpt_oss.conversion_chain import build_gpt_oss_chain

from .conftest import assert_state_dicts_equal, clone


def test_gpt_oss_identity():
    torch.manual_seed(0)
    sd = {
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 16, 8),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
    }
    chain = build_gpt_oss_chain()
    assert chain == []
    assert_state_dicts_equal(apply_hf_to_tt(clone(sd), chain), sd, "gpt_oss hf->tt")
    assert_state_dicts_equal(apply_tt_to_hf(clone(sd), chain), sd, "gpt_oss tt->hf")
