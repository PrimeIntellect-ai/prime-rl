import torch
from torch import nn

from prime_rl.trainer.models.layers.moe import (
    TokenChoiceTopKRouter,
    RoutingReplay,
    enable_routing_replay,
    register_routing_replay_keys,
    set_routing_replay_stage,
)


def test_token_choice_router_routing_replay_reuses_indices():
    torch.manual_seed(0)
    enable_routing_replay(True)

    router_a = TokenChoiceTopKRouter(dim=4, num_experts=6, top_k=2, score_func="sigmoid", route_norm=False, route_scale=1.0)
    router_b = TokenChoiceTopKRouter(dim=4, num_experts=6, top_k=2, score_func="sigmoid", route_norm=False, route_scale=1.0)

    module_a = nn.Module()
    module_a.router = router_a
    module_b = nn.Module()
    module_b.router = router_b
    register_routing_replay_keys(module_a)
    register_routing_replay_keys(module_b)

    assert router_a.routing_replay is router_b.routing_replay

    tokens = torch.randn(8, 4)

    set_routing_replay_stage("record")
    _, recorded_indices, _ = router_a(tokens)

    with torch.no_grad():
        torch.manual_seed(1)
        router_b.gate.weight.copy_(torch.randn_like(router_b.gate.weight))

    alt_scores = router_b.gate(tokens)
    alt_scores = torch.sigmoid(alt_scores.to(torch.float32))
    expected_indices = torch.topk(alt_scores, k=router_b.top_k, dim=1).indices

    set_routing_replay_stage("replay_forward")
    _, replay_indices, _ = router_b(tokens)

    assert torch.equal(replay_indices, recorded_indices)
    assert not torch.equal(replay_indices, expected_indices)

    RoutingReplay.clear_all()
    set_routing_replay_stage(None)
    enable_routing_replay(False)
