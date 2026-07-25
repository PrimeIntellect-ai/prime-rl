import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import _get_fqns

from prime_rl.trainer.models.layers.moe import GroupedExperts, TokenChoiceTopKRouter


def test_grouped_experts_only_fuses_gate_up_at_runtime():
    experts = GroupedExperts(dim=16, hidden_dim=8, num_experts=4)

    assert set(dict(experts.named_parameters())) == {"input_weight", "w2"}
    assert experts.input_weight.shape == (4, 16, 16)

    state_dict = experts.state_dict()
    assert set(state_dict) == {"w1", "w2", "w3"}
    assert state_dict["w1"].shape == (4, 8, 16)
    assert state_dict["w3"].shape == (4, 8, 16)

    state_dict["w1"].fill_(1)
    state_dict["w3"].fill_(3)
    torch.testing.assert_close(experts.input_weight, torch.cat((state_dict["w1"], state_dict["w3"]), dim=1))

    checkpoint = {name: torch.randn_like(weight) for name, weight in state_dict.items()}
    experts.load_state_dict(checkpoint)
    torch.testing.assert_close(experts.input_weight, torch.cat((checkpoint["w1"], checkpoint["w3"]), dim=1))
    torch.testing.assert_close(experts.w2, checkpoint["w2"])

    model = nn.Sequential(experts)
    assert _get_fqns(model, "0.w1") == {"0.w1"}
    assert _get_fqns(model, "0.w3") == {"0.w3"}

    interleaved_experts = GroupedExperts(
        dim=12,
        hidden_dim=8,
        num_experts=4,
        input_weight_names=("gate_up_proj",),
        input_weight_sizes=(16,),
        output_weight_name="down_proj",
        input_bias_name="gate_up_proj_bias",
        output_bias_name="down_proj_bias",
        transpose_weights_for_state_dict=True,
    )

    assert set(dict(interleaved_experts.named_parameters())) == {
        "input_weight",
        "w2",
        "input_bias",
        "output_bias",
    }
    assert interleaved_experts.input_weight.shape == (4, 16, 12)

    state_dict = interleaved_experts.state_dict()
    assert set(state_dict) == {"gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias"}
    assert state_dict["gate_up_proj"].shape == (4, 12, 16)
    assert state_dict["down_proj"].shape == (4, 8, 12)

    checkpoint = {name: torch.randn_like(weight) for name, weight in state_dict.items()}
    interleaved_experts.load_state_dict(checkpoint)
    torch.testing.assert_close(interleaved_experts.input_weight, checkpoint["gate_up_proj"].transpose(-2, -1))
    torch.testing.assert_close(interleaved_experts.w2, checkpoint["down_proj"].transpose(-2, -1))
    torch.testing.assert_close(interleaved_experts.input_bias, checkpoint["gate_up_proj_bias"])
    torch.testing.assert_close(interleaved_experts.output_bias, checkpoint["down_proj_bias"])


def test_router_runtime_is_independent_of_checkpoint_names():
    nemotron_router = TokenChoiceTopKRouter(
        dim=8,
        num_experts=4,
        top_k=2,
        score_func="sigmoid",
        route_norm=True,
        route_scale=1.0,
        weight_state_dict_name="gate",
        selection_bias_state_dict_name="e_score_correction_bias",
    )
    assert set(dict(nemotron_router.named_parameters())) == {"gate.weight"}
    assert set(nemotron_router.state_dict()) == {"gate", "e_score_correction_bias"}

    nemotron_checkpoint = {
        "gate": torch.randn(4, 8),
        "e_score_correction_bias": torch.randn(4),
    }
    nemotron_router.load_state_dict(nemotron_checkpoint)
    torch.testing.assert_close(nemotron_router.gate.weight, nemotron_checkpoint["gate"])
    torch.testing.assert_close(
        nemotron_router.e_score_correction_bias,
        nemotron_checkpoint["e_score_correction_bias"],
    )

    nemotron_model = nn.Sequential(nemotron_router)
    assert _get_fqns(nemotron_model, "0.gate") == {"0.gate"}
    assert _get_fqns(nemotron_model, "0.e_score_correction_bias") == {"0.e_score_correction_bias"}

    gpt_oss_router = TokenChoiceTopKRouter(
        dim=8,
        num_experts=4,
        top_k=2,
        score_func="topk_softmax",
        route_norm=False,
        route_scale=1.0,
        gate_bias=True,
        weight_state_dict_name="weight",
        bias_state_dict_name="bias",
    )
    assert set(dict(gpt_oss_router.named_parameters())) == {"gate.weight", "gate.bias"}
    assert set(gpt_oss_router.state_dict()) == {"weight", "bias"}

    gpt_oss_checkpoint = {
        "weight": torch.randn(4, 8),
        "bias": torch.randn(4),
    }
    gpt_oss_router.load_state_dict(gpt_oss_checkpoint)
    torch.testing.assert_close(gpt_oss_router.gate.weight, gpt_oss_checkpoint["weight"])
    torch.testing.assert_close(gpt_oss_router.gate.bias, gpt_oss_checkpoint["bias"])

    gpt_oss_model = nn.Sequential(gpt_oss_router)
    assert _get_fqns(gpt_oss_model, "0.weight") == {"0.weight"}
    assert _get_fqns(gpt_oss_model, "0.bias") == {"0.bias"}
