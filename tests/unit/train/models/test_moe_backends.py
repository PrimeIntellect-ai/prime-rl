import pytest
import torch

from prime_rl.configs.trainer import MoEBackendConfig, ModelConfig
from prime_rl.trainer.models.layers.moe import GroupedExperts, MoE, MoEArgs
from prime_rl.trainer.models.layers.moe_backends import MoEBackendSelection


def _build_moe(backends: MoEBackendSelection, score_before_experts: bool) -> MoE:
    moe = MoE(
        MoEArgs(
            num_experts=4,
            num_shared_experts=0,
            score_func="softmax",
            route_norm=False,
            route_scale=1.0,
            score_before_experts=score_before_experts,
            top_k=2,
            use_grouped_mm=False,
            load_balance_coeff=None,
            backends=backends,
        ),
        dim=8,
        hidden_dim=16,
    )
    moe.init_weights(init_std=0.02, buffer_device=torch.device("cpu"))
    return moe


def test_model_config_exposes_moe_backends() -> None:
    config = ModelConfig(
        name="PrimeIntellect/GLM-0.5B",
        moe_backends=MoEBackendConfig(
            routing="triton",
            scatter="triton",
            gather="triton",
        ),
    )

    assert config.moe_backends.routing == "triton"
    assert config.moe_backends.scatter == "triton"
    assert config.moe_backends.gather == "triton"
    assert config.moe_backends.grouped_ffn == "torch"


@pytest.mark.parametrize("score_before_experts", [False, True])
def test_moe_cpu_fallback_matches_default_path(score_before_experts: bool) -> None:
    torch.manual_seed(0)
    reference = _build_moe(MoEBackendSelection(), score_before_experts=score_before_experts)
    candidate = _build_moe(
        MoEBackendSelection(
            routing="triton",
            scatter="triton",
            gather="triton",
            grouped_ffn="triton",
        ),
        score_before_experts=score_before_experts,
    )
    candidate.load_state_dict(reference.state_dict())

    x = torch.randn(2, 3, 8)
    out_reference = reference(x)
    out_candidate = candidate(x)

    torch.testing.assert_close(out_candidate, out_reference)


def test_grouped_experts_grouped_ffn_matches_reference() -> None:
    torch.manual_seed(0)
    experts = GroupedExperts(dim=8, hidden_dim=16, num_experts=4, use_grouped_mm=False)
    experts.init_weights(init_std=0.02)

    x = torch.randn(6, 8)
    num_tokens_per_expert = torch.tensor([2, 1, 0, 3], dtype=torch.int32)

    out_reference = experts(x, num_tokens_per_expert)
    out_grouped_ffn = experts.forward_grouped_ffn(x, num_tokens_per_expert)

    torch.testing.assert_close(out_grouped_ffn, out_reference)
