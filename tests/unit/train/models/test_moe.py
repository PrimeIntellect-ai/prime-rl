import pytest
import torch
import torch.nn.functional as F

from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer.distributed.token_dispatcher import LocalTokenDispatcher
from prime_rl.trainer.model import compute_expert_load_stats, is_tt_moe_model
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4Experts
from prime_rl.trainer.models.fusions import fuse_gate_up_projections
from prime_rl.trainer.models.layers.activations import ActivationDispatch
from prime_rl.trainer.models.layers.expert_compute import BF16ExpertCompute, GroupedGemmExpertCompute
from prime_rl.trainer.models.layers.mlp import FeedForward
from prime_rl.trainer.models.layers.moe import (
    GroupedExperts,
    MoE,
    MoEArgs,
    TokenChoiceTopKRouter,
    filter_replayed_experts,
)
from prime_rl.trainer.models.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)
from prime_rl.trainer.moe_runtime import configure_moe_runtime
from prime_rl.trainer.parallel_dims import ParallelDims


@pytest.mark.parametrize(
    ("config_cls", "expected"),
    [
        (Qwen3_5TextConfig, False),
        (Qwen3_5MoeTextConfig, True),
        (Qwen3_5Config, False),
        (Qwen3_5MoeConfig, True),
    ],
    ids=["dense-text", "moe-text", "dense-vlm", "moe-vlm"],
)
def test_moe_detection_for_text_and_vlm(config_cls, expected):
    model = torch.nn.Module()
    model.config = config_cls()
    assert is_tt_moe_model(model) is expected


@pytest.mark.parametrize("selection", [[], [0], "0%", "50%"])
@pytest.mark.parametrize("compute", [{"type": "mxfp8"}, {"type": "bf16", "backend": "sonicmoe"}])
def test_unselected_moe_uses_bf16_without_loading_compute_backend(selection, compute):
    moe = MoE.from_args(MoEArgs(num_experts=2), dim=4, hidden_dim=8, shared_expert=None)
    parameters = dict(moe.named_parameters())
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Identity(), moe])
    config = ModelConfig.model_validate({"moe": {"compute": {**compute, "apply_to": selection}}})
    dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, pp=1, ep=1, world_size=1)

    configure_moe_runtime(model, config, dims)

    assert isinstance(moe.experts.compute, BF16ExpertCompute)
    assert isinstance(moe.token_dispatcher, LocalTokenDispatcher)
    assert moe.token_dispatcher.token_group_alignment == moe.experts.compute.token_group_alignment
    assert all(moe.get_parameter(name) is parameter for name, parameter in parameters.items())


@pytest.mark.parametrize(
    ("expert_cls", "kwargs", "fused", "error"),
    [
        pytest.param(GroupedExperts, {}, True, None, id="swiglu"),
        pytest.param(GroupedExperts, {}, False, "fused gate/up", id="unfused"),
        pytest.param(GroupedExperts, {"expert_type": "non_gated"}, False, "gated experts", id="non-gated"),
        pytest.param(GroupedExperts, {"activation": "relu2"}, True, "standard SwiGLU", id="relu2"),
        pytest.param(GroupedExperts, {"activation": "clamped_swiglu"}, True, "standard SwiGLU", id="clamped-swiglu"),
        pytest.param(GroupedExperts, {"bias": True}, True, "bias-free", id="bias"),
        pytest.param(DeepseekV4Experts, {"swiglu_limit": 10.0}, True, "standard SwiGLU", id="deepseek-v4"),
    ],
)
def test_set_compute_validates_sonic_expert_structure(expert_cls, kwargs, fused, error):
    pytest.importorskip("sonicmoe")
    from prime_rl.trainer.models.layers.sonic_moe import SonicMoEExpertCompute

    experts = expert_cls(dim=4, hidden_dim=8, num_experts=2, **kwargs)
    if fused:
        fuse_gate_up_projections(experts)
    original_compute = experts.compute
    compute = SonicMoEExpertCompute()

    if error is not None:
        with pytest.raises(ValueError, match=error):
            experts.set_compute(compute)
        assert experts.compute is original_compute
    else:
        experts.set_compute(compute)
        assert experts.compute is compute


def _grouped_mm_reference(x: torch.Tensor, weights: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    outputs = []
    start = 0
    for expert, end in enumerate(offs.tolist()):
        outputs.append(x[start:end] @ weights[expert])
        start = end
    return torch.cat(outputs)


def _scaled_square_experts(x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(x)
    scales = torch.repeat_interleave(torch.arange(1, len(counts) + 1, dtype=x.dtype), counts)
    output[: len(scales)] = x[: len(scales)].square() * scales.unsqueeze(1)
    return output


@pytest.mark.parametrize("score_before_experts", [True, False])
def test_local_token_dispatcher(score_before_experts):
    dispatcher = LocalTokenDispatcher(num_experts=3, top_k=2, token_group_alignment=1)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_()
    scores = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.9, 0.1]], requires_grad=True)
    selected = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 0]])

    actual = dispatcher.run(
        x,
        scores,
        selected,
        _scaled_square_experts,
        score_before_experts=score_before_experts,
    )

    expected = torch.zeros_like(x)
    for token in range(len(x)):
        for route in range(selected.shape[1]):
            score = scores[token, route]
            expert_scale = selected[token, route] + 1
            expert_input = x[token] * score if score_before_experts else x[token]
            contribution = expert_input.square() * expert_scale
            if not score_before_experts:
                contribution = contribution * score
            expected[token] = expected[token] + contribution

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None
    assert scores.grad is not None


def test_local_token_dispatcher_handles_empty_input_and_unused_experts():
    dispatcher = LocalTokenDispatcher(num_experts=3, top_k=1, token_group_alignment=1)
    x = torch.randn(2, 4)
    scores = torch.ones(2, 1)
    selected = torch.tensor([[0], [2]])

    output = dispatcher.run(x, scores, selected, _scaled_square_experts, score_before_experts=False)
    torch.testing.assert_close(output[0], x[0].square())
    torch.testing.assert_close(output[1], x[1].square() * 3)

    empty = dispatcher.run(
        x.new_empty((0, 4)),
        scores.new_empty((0, 1)),
        selected.new_empty((0, 1)),
        _scaled_square_experts,
        score_before_experts=False,
    )
    assert empty.shape == (0, 4)


@pytest.mark.parametrize(
    ("expert_type", "activation"),
    [
        ("gated", "silu"),
        ("gated", "relu2"),
        ("non_gated", "silu"),
        ("non_gated", "relu2"),
    ],
)
def test_expert_type_and_activation_are_independent(expert_type, activation):
    experts = GroupedExperts(
        dim=4,
        hidden_dim=8,
        num_experts=2,
        expert_type=expert_type,
        activation=activation,
        bias=True,
    )
    experts.set_compute(GroupedGemmExpertCompute(_grouped_mm_reference, token_group_alignment=1))
    assert experts.token_group_alignment == 1
    experts.init_weights(0.02)

    has_gate = expert_type == "gated"
    assert (experts.gate_proj is not None) == has_gate
    assert (experts.gate_proj_bias is not None) == has_gate
    assert ("gate_proj" in experts.state_dict()) == has_gate

    x = torch.randn(3, 4)
    counts = torch.tensor([2, 1])
    actual = experts(x, counts)

    expected = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        expert_input = x[start : start + count].bfloat16()
        up = F.linear(expert_input, experts.up_proj[expert].bfloat16(), experts.up_proj_bias[expert].bfloat16())
        activation_input = up
        if has_gate:
            activation_input = F.linear(
                expert_input,
                experts.gate_proj[expert].bfloat16(),
                experts.gate_proj_bias[expert].bfloat16(),
            )
        hidden = F.silu(activation_input) if activation == "silu" else F.relu(activation_input).square()
        if has_gate:
            hidden = hidden * up
        expected.append(
            F.linear(hidden, experts.down_proj[expert].bfloat16(), experts.down_proj_bias[expert].bfloat16())
        )
        start += count

    torch.testing.assert_close(actual, torch.cat(expected).float())

    with torch.device("meta"):
        shared_expert = FeedForward(
            dim=4,
            hidden_dim=8,
            expert_type=expert_type,
            activation=activation,
        )
        moe = MoE.from_args(
            MoEArgs(
                num_experts=2,
                expert_type=expert_type,
                activation=activation,
                load_balance_coeff=None,
            ),
            dim=4,
            hidden_dim=8,
            shared_expert=shared_expert,
        )
    assert moe.shared_expert is shared_expert
    assert (moe.experts.gate_proj is not None) == has_gate
    assert moe.experts.activation is ActivationDispatch[activation]
    assert (moe.shared_expert.gate_proj is not None) == has_gate
    assert moe.shared_expert.activation is ActivationDispatch[activation]


def test_expert_load_stats():
    # Layer 0 is balanced; layer 1 sends 8 of 16 tokens to one expert and leaves one expert cold (< 0.1x mean).
    tokens_per_expert = torch.tensor([[4.0, 4.0, 4.0, 4.0], [8.0, 4.0, 4.0, 0.0]])
    stats = compute_expert_load_stats(tokens_per_expert)
    expected = {
        "expert_load/cv/mean": 2**0.5 / 4,
        "expert_load/cv/max": 2**0.5 / 2,
        "expert_load/max_mean/mean": 1.5,
        "expert_load/max_mean/max": 2.0,
        "expert_load/cold_frac/mean": 0.125,
        "expert_load/cold_frac/max": 0.25,
    }
    assert stats.keys() == expected.keys()
    for name, value in expected.items():
        assert stats[name].item() == pytest.approx(value)


def test_filter_replayed_experts():
    scores = torch.tensor([[0.30, 0.25, 0.20, 0.15, 0.06, 0.04], [0.01, 0.02, 0.40, 0.30, 0.15, 0.12]])
    routed = torch.tensor([[3, 4, 1], [0, 2, 1]], dtype=torch.int32)
    # Thresholds are 0.5 * 0.20 and 0.5 * 0.15: dropped slots take the router's best experts not already kept.
    filtered = filter_replayed_experts(scores, routed, min_score_ratio=0.5)
    assert filtered.dtype == routed.dtype
    assert filtered.tolist() == [[3, 0, 1], [3, 2, 4]]

    router = TokenChoiceTopKRouter(
        dim=8, num_experts=6, top_k=3, score_func="softmax", route_norm=True, route_scale=1.0
    )
    router.init_weights(0.5)
    x = torch.randn(16, 8)
    routed = torch.stack([torch.randperm(6)[:3] for _ in range(16)])
    router.replay_min_score_ratio = 1.0
    top_scores, selected, _, _ = router(x, routed_experts=routed)
    # With ratio 1 only the router's own top-k survive, as distinct experts.
    own = torch.topk(router.gate(x), k=3, dim=1).indices
    assert selected.sort(dim=1).values.tolist() == own.sort(dim=1).values.tolist()
    top_scores.pow(2).sum().backward()
    assert router.gate.weight.grad.abs().sum() > 0
