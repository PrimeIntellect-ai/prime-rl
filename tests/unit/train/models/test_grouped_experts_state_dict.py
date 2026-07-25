import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import _get_fqns

from prime_rl.trainer.models.layers.moe import GroupedExperts


def test_grouped_experts_only_fuses_gate_up_at_runtime():
    experts = GroupedExperts(dim=16, hidden_dim=8, num_experts=4)

    assert set(dict(experts.named_parameters())) == {"w13", "w2"}
    assert experts.w13.shape == (4, 16, 16)

    state_dict = experts.state_dict()
    assert set(state_dict) == {"w1", "w2", "w3"}
    assert state_dict["w1"].shape == (4, 8, 16)
    assert state_dict["w3"].shape == (4, 8, 16)

    state_dict["w1"].fill_(1)
    state_dict["w3"].fill_(3)
    torch.testing.assert_close(experts.w13, torch.cat((state_dict["w1"], state_dict["w3"]), dim=1))

    checkpoint = {name: torch.randn_like(weight) for name, weight in state_dict.items()}
    experts.load_state_dict(checkpoint)
    torch.testing.assert_close(experts.w13, torch.cat((checkpoint["w1"], checkpoint["w3"]), dim=1))
    torch.testing.assert_close(experts.w2, checkpoint["w2"])

    model = nn.Sequential(experts)
    assert _get_fqns(model, "0.w1") == {"0.w1"}
    assert _get_fqns(model, "0.w3") == {"0.w3"}
