from types import SimpleNamespace

import torch

from prime_rl.inference.vllm.models.inkling2.moe import Inkling2MoE


def _moe(bias):
    live = torch.zeros(4)
    return SimpleNamespace(
        gate=SimpleNamespace(expert_bias=bias),
        experts=SimpleNamespace(
            routed_experts=SimpleNamespace(e_score_correction_bias=live),
            router=SimpleNamespace(e_score_correction_bias=live),
        ),
    ), live


def test_refresh_router_bias_repoints_at_a_loaded_bias():
    bias = torch.ones(4)
    moe, _ = _moe(bias)
    Inkling2MoE.refresh_router_bias(moe)
    assert moe.experts.routed_experts.e_score_correction_bias is bias
    assert moe.experts.router.e_score_correction_bias is bias


def test_refresh_router_bias_keeps_the_live_bias_during_a_meta_capture():
    moe, live = _moe(torch.empty(4, device="meta"))
    Inkling2MoE.refresh_router_bias(moe)
    assert moe.experts.routed_experts.e_score_correction_bias is live
    assert moe.experts.router.e_score_correction_bias is live
