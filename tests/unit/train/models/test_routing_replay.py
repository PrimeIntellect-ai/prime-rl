"""CPU tests for immutable routing inputs and the native direct-replay router.

MoE tests inject a small CPU reference dispatcher/expert implementation through
its supported interfaces. They do not test CUDA grouped GEMM or permutation.
"""

import copy
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from prime_rl.configs.trainer import ActivationCheckpointConfig
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.trainer.models.layers.moe import MoE, TokenChoiceTopKRouter
from prime_rl.trainer.routing_replay import RoutingReplay, select_routing_layer


def make_router(**kwargs):
    return TokenChoiceTopKRouter(
        dim=4,
        num_experts=3,
        top_k=2,
        score_func=kwargs.pop("score_func", "softmax"),
        route_norm=kwargs.pop("route_norm", True),
        route_scale=kwargs.pop("route_scale", 1.0),
        **kwargs,
    )


def test_routing_replay_is_an_immutable_paired_tensor_input():
    ids = torch.arange(24).reshape(1, 3, 4, 2)
    weights = torch.arange(24, dtype=torch.float32).reshape(1, 3, 4, 2) / 31
    replay = RoutingReplay(ids, weights)
    with pytest.raises(AttributeError):
        replay.ids = ids.clone()
    leaves, structure = tree_flatten(replay)
    assert leaves[0] is ids and leaves[1] is weights
    rebuilt = tree_unflatten(leaves, structure)
    assert isinstance(rebuilt, RoutingReplay)
    assert rebuilt.ids is ids and rebuilt.weights is weights
    moved = replay.to("cpu", non_blocking=True)
    assert isinstance(moved, RoutingReplay)
    assert moved.ids.dtype == ids.dtype
    assert moved.weights.dtype == torch.float32
    torch.testing.assert_close(moved.ids, ids, rtol=0, atol=0)
    torch.testing.assert_close(moved.weights, weights, rtol=0, atol=0)


def test_select_layer_preserves_noncontiguous_pair_and_legacy_inputs():
    ids = torch.arange(48).reshape(2, 3, 4, 2)
    weights = torch.arange(48, dtype=torch.float32).reshape(2, 3, 4, 2) / 53
    replay = RoutingReplay(ids, weights)
    selected = select_routing_layer(replay, 2)
    assert isinstance(selected, RoutingReplay)
    assert not selected.ids.is_contiguous() and not selected.weights.is_contiguous()
    torch.testing.assert_close(selected.ids, ids[:, :, 2, :], rtol=0, atol=0)
    torch.testing.assert_close(selected.weights, weights[:, :, 2, :], rtol=0, atol=0)
    torch.testing.assert_close(select_routing_layer(ids, 2), selected.ids, rtol=0, atol=0)
    assert select_routing_layer(None, 2) is None
    with pytest.raises(IndexError):
        select_routing_layer(replay, 4)
    with pytest.raises(TypeError, match="Tensor, RoutingReplay, or None"):
        select_routing_layer((ids, weights), 0)
    with pytest.raises(ValueError, match="same shape"):
        select_routing_layer(RoutingReplay(ids, weights[:, :-1]), 0)
    with pytest.raises(ValueError, match="Model RoutingReplay"):
        select_routing_layer(RoutingReplay(ids[0], weights[0]), 0)


@pytest.mark.parametrize("score_func", ["softmax", "sigmoid", "topk_softmax"])
@pytest.mark.parametrize("fp32_gate", [False, True])
def test_direct_router_skips_gate_and_uses_exact_detached_final_coefficients(monkeypatch, score_func, fp32_gate):
    router = make_router(score_func=score_func, route_norm=True, route_scale=7.0, selection_bias=True)
    router.fp32_gate = fp32_gate
    router.force_balanced = True  # Recorded routes take priority over selection controls.
    x = torch.randn(3, 4, requires_grad=True)
    ids = torch.tensor([[2, 0], [1, 2], [0, 1]], dtype=torch.int32)
    # Intentionally not all normalized: detect accidental normalization or scaling.
    weights = torch.tensor([[0.1, 0.7], [1.125, 0.3], [0.0, 0.0]], requires_grad=True)
    gate = Mock(side_effect=AssertionError("gate must not execute in direct replay"))
    monkeypatch.setattr(router.gate, "forward", gate)
    monkeypatch.setattr(F, "linear", Mock(side_effect=AssertionError("FP32 gate must not execute")))
    monkeypatch.setattr(F, "softmax", Mock(side_effect=AssertionError("scores must not be recomputed")))
    monkeypatch.setattr(torch, "sigmoid", Mock(side_effect=AssertionError("scores must not be recomputed")))

    scores, selected, counts, confidence = router(x, RoutingReplay(ids, weights))

    assert selected is ids
    assert scores.data_ptr() == weights.data_ptr()
    assert not scores.requires_grad and scores.grad_fn is None
    assert scores.dtype == torch.float32
    assert torch.equal(scores.view(torch.int32), weights.detach().view(torch.int32))
    torch.testing.assert_close(counts, torch.tensor([2, 2, 2]), rtol=0, atol=0)
    assert torch.isnan(confidence)  # Captured coefficient mass is NOT trainer confidence.
    gate.assert_not_called()
    (x.unsqueeze(1) * scores.unsqueeze(-1)).sum().backward()
    assert weights.grad is None
    assert router.gate.weight.grad is None
    assert x.grad is not None and x.grad[:2].abs().sum() > 0


@pytest.mark.parametrize("score_func", ["softmax", "sigmoid", "topk_softmax"])
@pytest.mark.parametrize("route_norm", [False, True])
def test_legacy_ids_only_retains_recomputed_coefficients_and_gradients(score_func, route_norm):
    torch.manual_seed(3)
    router = make_router(score_func=score_func, route_norm=route_norm, route_scale=1.7)
    x = torch.randn(4, 4, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_gate = router.gate.weight.detach().clone().requires_grad_()
    ids = torch.tensor([[2, 0], [1, 0], [0, 1], [2, 1]])
    call_count = []
    hook = router.gate.register_forward_hook(lambda *args: call_count.append(1))
    actual, selected, counts, confidence = router(x, ids)
    hook.remove()
    assert call_count == [1]

    logits = F.linear(reference_x, reference_gate).float()
    if score_func == "topk_softmax":
        expected = F.softmax(logits.gather(1, ids), dim=-1)
        expected_confidence = expected.sum()
    else:
        probabilities = F.softmax(logits, dim=1) if score_func == "softmax" else torch.sigmoid(logits)
        expected = probabilities.gather(1, ids)
        mass = expected if score_func == "softmax" else expected / (probabilities.sum(-1, keepdim=True) + 1e-20)
        expected_confidence = mass.sum()
    if route_norm:
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-20)
    expected = expected * 1.7

    assert selected is ids
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(confidence, expected_confidence.detach(), rtol=0, atol=0)
    torch.testing.assert_close(counts, torch.tensor([3, 3, 2]), rtol=0, atol=0)
    factor = torch.arange(8).reshape(4, 2)
    (actual * factor).sum().backward()
    (expected * factor).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(router.gate.weight.grad, reference_gate.grad)
    assert router.gate.weight.grad.abs().sum() > 0


def test_no_replay_and_legacy_replay_of_selected_ids_match():
    torch.manual_seed(11)
    router = make_router()
    x = torch.randn(7, 4)
    scores, ids, counts, confidence = router(x)
    replayed = router(x, ids)
    for actual, expected in zip(replayed, (scores, ids, counts, confidence)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("ids", "weights", "error", "message"),
    [
        (torch.zeros(3, 2), torch.ones(3, 2), TypeError, "IDs must have dtype"),
        (torch.zeros(3, 2, dtype=torch.bool), torch.ones(3, 2), TypeError, "IDs must have dtype"),
        (torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 2).half(), TypeError, "weights must have dtype"),
        (torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 1), ValueError, "same shape"),
        (torch.zeros(3, 1, dtype=torch.long), torch.ones(3, 1), ValueError, "does not match expected"),
        (torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2), ValueError, "does not match expected"),
        (torch.zeros(1, 3, 2, dtype=torch.long), torch.ones(1, 3, 2), ValueError, "does not match expected"),
        (torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 2, device="meta"), ValueError, "same device"),
        (torch.zeros(3, 2, dtype=torch.long), None, TypeError, "must be tensors"),
    ],
)
def test_direct_router_rejects_malformed_pair_metadata(ids, weights, error, message):
    router = make_router()
    with pytest.raises(error, match=message):
        router(torch.randn(3, 4), RoutingReplay(ids, weights))


def test_direct_router_rejects_hidden_state_device_mismatch():
    replay = RoutingReplay(torch.zeros(3, 2, dtype=torch.long), torch.ones(3, 2))
    with pytest.raises(ValueError, match="same device as the hidden states"):
        make_router()(torch.empty(3, 4, device="meta"), replay)


def test_direct_router_accepts_empty_token_batch():
    scores, ids, counts, confidence = make_router()(
        torch.empty(0, 4), RoutingReplay(torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2))
    )
    assert scores.shape == ids.shape == (0, 2)
    assert counts.sum() == 0
    assert torch.isnan(confidence)


class CPUScaleExperts(nn.Module):
    num_experts = 3
    token_group_alignment = 1

    def __init__(self):
        super().__init__()
        self.scales = nn.Parameter(torch.arange(1, 13, dtype=torch.float32).reshape(3, 4) / 5)

    def forward(self, x, counts):
        scales = torch.repeat_interleave(self.scales, counts, dim=0, output_size=x.shape[0])
        return x.square() * scales


class CPUReferenceDispatcher:
    """Small CPU backend, injected instead of the CUDA-oriented permutation path."""

    def run(self, x, scores, ids, experts, *, score_before_experts):
        # These are the production local dispatch ordering and coefficient-application
        # functions. Only the padding/permutation backend is omitted in this CPU test.
        from prime_rl.trainer.distributed.token_dispatcher import _local_reorder, _scatter_routed_output

        routed, indices, sorted_scores, counts = _local_reorder(
            x, scores, ids, num_experts=experts.num_experts, top_k=ids.shape[-1]
        )
        if score_before_experts:
            routed = (routed.float() * sorted_scores[:, None]).to(x.dtype)
        output = experts(routed, counts)
        return _scatter_routed_output(
            output,
            num_tokens=x.shape[0],
            token_indices_experts_sorted=indices,
            scores_after_experts=None if score_before_experts else sorted_scores,
        )

    def synchronize(self):
        pass


def make_cpu_moe(score_before_experts=False):
    moe = MoE(
        router=make_router(),
        experts=CPUScaleExperts(),
        shared_expert=None,
        score_before_experts=score_before_experts,
        load_balance_coeff=None,
    )
    moe.set_token_dispatcher(CPUReferenceDispatcher())
    return moe


@pytest.mark.parametrize("score_before_experts", [False, True])
def test_moe_flattens_paired_layer_and_preserves_expert_and_input_gradients(score_before_experts):
    torch.manual_seed(7)
    moe = make_cpu_moe(score_before_experts)
    x = torch.randn(2, 3, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_scales = moe.experts.scales.detach().clone().requires_grad_()
    ids = torch.arange(24).reshape(2, 3, 2, 2) % 3
    weights = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 29).requires_grad_()
    pair = select_routing_layer(RoutingReplay(ids, weights), 1)
    output = moe(x, pair)

    chosen_scales = ref_scales[pair.ids]
    expert_inputs = ref_x.unsqueeze(-2)
    if score_before_experts:
        expert_inputs = expert_inputs * pair.weights.detach().unsqueeze(-1)
    contributions = expert_inputs.square() * chosen_scales
    if not score_before_experts:
        contributions = contributions * pair.weights.detach().unsqueeze(-1)
    expected = contributions.sum(-2)
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(moe.experts.scales.grad, ref_scales.grad)
    assert x.grad.abs().sum() > 0 and moe.experts.scales.grad.abs().sum() > 0
    assert moe.router.gate.weight.grad is None and weights.grad is None
    torch.testing.assert_close(moe.tokens_per_expert, torch.bincount(pair.ids.reshape(-1), minlength=3).float())
    assert torch.isnan(moe.routing_confidence_sum)


def test_moe_rejects_pair_with_wrong_batch_axes_even_when_token_count_matches():
    pair = RoutingReplay(torch.zeros(3, 2, 2, dtype=torch.long), torch.ones(3, 2, 2))
    with pytest.raises(ValueError, match="does not match expected"):
        make_cpu_moe()(torch.randn(2, 3, 4), pair)


@pytest.mark.parametrize("mode", ["full", "selective"])
def test_checkpoint_reuses_immutable_pairs_across_queued_microbatches_and_records_counts_once(mode):
    torch.manual_seed(19)
    moe = make_cpu_moe()
    reference = copy.deepcopy(moe)
    checkpointed = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode=mode))(moe)
    inputs = [torch.randn(1, 3, 4, requires_grad=True) for _ in range(2)]
    ref_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    pairs = [
        RoutingReplay(
            (torch.arange(6).reshape(1, 3, 2) + i) % 3,
            (torch.arange(1, 7, dtype=torch.float32).reshape(1, 3, 2) / (7 + i)).requires_grad_(),
        )
        for i in range(2)
    ]
    originals = [RoutingReplay(p.ids.clone(), p.weights.detach().clone()) for p in pairs]
    outputs = [checkpointed(x, routed_experts=p) for x, p in zip(inputs, pairs)]
    expected_outputs = [reference(x, routed_experts=p) for x, p in zip(ref_inputs, pairs)]
    counts_after_forward = moe.tokens_per_expert.clone()
    for output, expected in reversed(list(zip(outputs, expected_outputs))):
        torch.testing.assert_close(output, expected)
        output.square().sum().backward()
        expected.square().sum().backward()
    torch.testing.assert_close(moe.tokens_per_expert, counts_after_forward, rtol=0, atol=0)
    assert counts_after_forward.sum() == 12
    assert torch.isnan(moe.routing_confidence_sum)
    for x, ref_x, p, original in zip(inputs, ref_inputs, pairs, originals):
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(p.ids, original.ids, rtol=0, atol=0)
        torch.testing.assert_close(p.weights, original.weights, rtol=0, atol=0)
        assert p.weights.grad is None
    torch.testing.assert_close(moe.experts.scales.grad, reference.experts.scales.grad)
    assert moe.router.gate.weight.grad is None


def test_direct_router_and_paired_layer_slice_compile_with_eager_backend():
    router = make_router()

    def forward(x, pair):
        layer = select_routing_layer(pair, 1)
        flat = RoutingReplay(layer.ids.reshape(-1, 2), layer.weights.reshape(-1, 2))
        return router(x, flat)

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    x = torch.randn(3, 4)
    pair = RoutingReplay(torch.arange(12).reshape(1, 3, 2, 2) % 3, torch.rand(1, 3, 2, 2))
    expected = forward(x, pair)
    actual = compiled(x, pair)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)


def test_full_moe_bypasses_router_module_hooks_but_legacy_replay_keeps_them():
    moe = make_cpu_moe()
    moe.router.requires_grad_(False)
    calls = []
    handle = moe.router.register_forward_pre_hook(lambda *args: calls.append("router"))
    x = torch.randn(1, 3, 4, requires_grad=True)
    pair = RoutingReplay(torch.arange(6).reshape(1, 3, 2) % 3, torch.full((1, 3, 2), 0.5))
    try:
        moe(x, pair).sum().backward()
        assert not calls
        assert x.grad is not None and moe.experts.scales.grad is not None
        # Registered/frozen router parameters remain available for checkpointing.
        assert "router.gate.weight" in moe.state_dict()
        moe(x.detach(), pair.ids)
        assert calls == ["router"]
    finally:
        handle.remove()


def test_full_moe_helper_compiles_with_eager_backend():
    model = make_cpu_moe()
    reference = copy.deepcopy(model)
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    x = torch.randn(1, 3, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    pair = RoutingReplay(torch.arange(6).reshape(1, 3, 2) % 3, torch.full((1, 3, 2), 0.5))
    actual = compiled(x, pair)
    expected = reference(ref_x, pair)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch.testing.assert_close(model.experts.scales.grad, reference.experts.scales.grad)
