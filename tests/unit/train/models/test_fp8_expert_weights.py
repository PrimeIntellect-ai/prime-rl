import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from prime_rl.trainer.models.layers.fp8_expert_weights import Fp8ExpertMaster, all_gather_expert_weights_in_fp8


class Experts(nn.Module):
    def __init__(self, num_experts: int = 4):
        super().__init__()
        self.down_proj = nn.Parameter(torch.randn(num_experts, 8, 16))
        self.down_proj_bias = nn.Parameter(torch.randn(num_experts, 8))


def fp8_all_gathered(experts: Experts) -> Experts:
    all_gather_expert_weights_in_fp8(experts, shard_size=1)
    assert type(experts.down_proj.data) is Fp8ExpertMaster
    return experts


def test_checkpoints_hold_plain_tensors_with_the_fp8_all_gather_on_or_off(tmp_path):
    on, off = fp8_all_gathered(Experts()), Experts()
    dcp.save(on.state_dict(), checkpoint_id=tmp_path / "on", no_dist=True)
    dcp.save(off.state_dict(), checkpoint_id=tmp_path / "off", no_dist=True)

    resumed_off = Experts()
    dcp.load(resumed_off.state_dict(), checkpoint_id=tmp_path / "on", no_dist=True)
    assert torch.equal(resumed_off.down_proj, on.down_proj._tensor)

    resumed_on = fp8_all_gathered(Experts())
    dcp.load(resumed_on.state_dict(), checkpoint_id=tmp_path / "off", no_dist=True)
    assert type(resumed_on.down_proj.data) is Fp8ExpertMaster
    assert torch.equal(resumed_on.down_proj._tensor, off.down_proj)

    torch.save(on.state_dict(), tmp_path / "weights.pt")
    assert type(torch.load(tmp_path / "weights.pt", weights_only=True)["down_proj"]) is torch.Tensor
