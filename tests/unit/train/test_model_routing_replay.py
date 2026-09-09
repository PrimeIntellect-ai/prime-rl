"""CPU admission/forward-plumbing checks; not full Qwen3 numerical tests."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from prime_rl.trainer.model import forward, get_load_balance_stats, validate_full_router_replay_model
from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from prime_rl.trainer.routing_replay import RoutingReplay


def tiny_model():
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        max_position_embeddings=32,
        norm_topk_prob=True,
        load_balance_coeff=None,
    )
    config._attn_implementation = "flash_attention_2"
    model = Qwen3MoeForCausalLM._from_config(config)
    for layer in model.model.layers:
        layer.mlp.router.requires_grad_(False)
    return model


def test_full_replay_guard_accepts_only_frozen_custom_qwen3_convention():
    model = tiny_model()
    validate_full_router_replay_model(model)
    model.model.layers[0].mlp.router.gate.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen MoE router"):
        validate_full_router_replay_model(model)
    model.model.layers[0].mlp.router.requires_grad_(False)
    model.model.layers[0].mlp.router.route_scale = 2.0
    with pytest.raises(ValueError, match="route scale 1"):
        validate_full_router_replay_model(model)
    model.model.layers[0].mlp.router.route_scale = 1.0
    model.config.norm_topk_prob = False
    with pytest.raises(ValueError, match="normalized top-k coefficients"):
        validate_full_router_replay_model(model)
    with pytest.raises(ValueError, match="custom Qwen3"):
        validate_full_router_replay_model(nn.Linear(2, 2))


@pytest.mark.parametrize("kwargs", [{"cp_size": 2}, {"ep_size": 2}, {"has_vlm": True}])
def test_full_replay_guard_rejects_unsupported_parallel_and_multimodal_modes(kwargs):
    with pytest.raises(ValueError, match="CP>1, EP>1, or VLM"):
        validate_full_router_replay_model(tiny_model(), **kwargs)


def test_unified_forward_passes_exact_pair_and_rejects_unsupported_call_shapes(monkeypatch):
    model = tiny_model()
    capture = Mock(return_value={"logits": torch.zeros(1, 3, 16)})
    monkeypatch.setattr(model, "forward", capture)
    ids = torch.tensor([[1, 2, 3]])
    positions = torch.arange(3).unsqueeze(0)
    pair = RoutingReplay(torch.zeros(1, 3, 2, 2, dtype=torch.int32), torch.ones(1, 3, 2, 2) / 2)
    forward(model, ids, positions, seq_lens=torch.tensor([3]), routed_experts=pair)
    assert capture.call_args.kwargs["routed_experts"] is pair
    assert "seq_lens" in capture.call_args.kwargs
    capture.reset_mock()
    with pytest.raises(ValueError, match="CP>1, EP>1, or VLM"):
        forward(model, ids, positions, seq_lens=torch.tensor([3]), routed_experts=pair, seq_lens_are_pre_shard=True)
    capture.assert_not_called()
    with pytest.raises(ValueError, match="does not match expected"):
        forward(model, ids[:, :-1], positions[:, :-1], seq_lens=torch.tensor([2]), routed_experts=pair)
    capture.assert_not_called()


def test_full_replay_metrics_omit_unavailable_confidence_and_reset_counts():
    model = tiny_model()
    for layer in model.model.layers:
        layer.mlp.tokens_per_expert.fill_(2)
        layer.mlp.routing_confidence_sum.fill_(float("nan"))
    stats = get_load_balance_stats(model, try_to_avoid_padding_experts=False, include_routing_confidence=False)
    assert stats["routing_confidence"] is None
    torch.testing.assert_close(stats["max_vio"], torch.zeros(2), rtol=0, atol=0)
    for layer in model.model.layers:
        assert layer.mlp.tokens_per_expert.sum() == 0
        assert layer.mlp.routing_confidence_sum == 0
    for layer in model.model.layers:
        layer.mlp.tokens_per_expert.fill_(2)
        layer.mlp.routing_confidence_sum.fill_(3)
    legacy_stats = get_load_balance_stats(model, try_to_avoid_padding_experts=False)
    torch.testing.assert_close(legacy_stats["routing_confidence"], torch.full((2,), 0.75), rtol=0, atol=0)


@pytest.mark.parametrize("preserve", [False, True])
def test_fsdp_policy_preserves_full_pair_dtypes_without_changing_legacy_policy(monkeypatch, preserve):
    from types import SimpleNamespace

    import prime_rl.trainer.model as model_module

    model = tiny_model()
    policies = []
    monkeypatch.setattr(model_module, "fully_shard", lambda *args, **kwargs: policies.append(kwargs["mp_policy"]))
    from prime_rl.configs.trainer import ModelConfig

    config = ModelConfig(
        reduce_dtype="float32",
        fsdp_cpu_offload=False,
        reshard_after_forward=True,
        vlm=None,
        moe_router_dtype="float32",
        fusions={"shard_fused_on_dim1": False},
    )
    parallel_dims = SimpleNamespace(ep_enabled=False, get_mesh=lambda name: None)
    model_module.setup_fsdp(model, config, parallel_dims, preserve_routing_weights=preserve)
    assert policies
    assert all(policy.cast_forward_inputs is (not preserve) for policy in policies)
    assert any(policy.param_dtype == torch.float32 for policy in policies)
    assert any(policy.param_dtype == torch.bfloat16 for policy in policies)


def test_full_replay_guard_checks_layer_list_without_recursive_module_walk(monkeypatch):
    model = tiny_model()
    monkeypatch.setattr(model, "modules", Mock(side_effect=AssertionError("no recursive model traversal")))
    validate_full_router_replay_model(model)
    model.model.layers[0].mlp = nn.Identity()
    with pytest.raises(ValueError, match="MoE block in every"):
        validate_full_router_replay_model(model)


def test_full_replay_guard_rejects_incomplete_layer_layout():
    model = tiny_model()
    del model.model.layers[-1]
    with pytest.raises(ValueError, match="complete Qwen3 MoE layer layout"):
        validate_full_router_replay_model(model)
