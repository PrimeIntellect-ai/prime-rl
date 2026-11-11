from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from prime_rl.trainer.models.layers.nvfp4_expert_weights import (
    NVFP4ExpertMaster,
    NVFP4ExpertWeight,
    all_gather_expert_weights_in_nvfp4,
)


def fake_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """A per-expert quantizer with NVFP4's mix of dtypes and shapes: packed bytes, per-block FP8 scales, one FP32
    scale per expert, and an MXFP8-like second form."""
    experts = weight.shape[0]
    amax = weight.float().abs().flatten(1).amax(dim=1)
    packed = (weight.float() / amax[:, None, None] * 7).round().to(torch.int8).view(torch.uint8)[..., ::2]
    blocks = weight.float().abs().reshape(experts, -1, 16).amax(-1).to(torch.float8_e4m3fn)
    dgrad = weight.transpose(1, 2).contiguous().to(torch.float8_e4m3fn)
    return packed.contiguous(), blocks, amax, dgrad, blocks[:, :3].contiguous().view(torch.uint8)


POLICY = SimpleNamespace(param_dtype=torch.bfloat16)


def gather(weight: torch.Tensor, ranks: int) -> NVFP4ExpertWeight:
    """What FSDP does: each rank's pre-all-gather on its shard, the inputs concatenated over ranks, the post."""
    mesh = SimpleNamespace(size=lambda: ranks)
    shards = [NVFP4ExpertMaster(shard, fake_quantize) for shard in weight.chunk(ranks)]
    inputs = [shard.fsdp_pre_all_gather(mesh, None, None, None, POLICY) for shard in shards]
    metadata = inputs[0][1]
    outputs = tuple(torch.cat(parts) for parts in zip(*(i[0] for i in inputs)))
    master = NVFP4ExpertMaster(weight, fake_quantize)
    gathered, _ = master.fsdp_post_all_gather(outputs, metadata, torch.bfloat16)
    return gathered


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_gathered_quantized_shards_equal_the_whole_weight_quantized(ranks):
    torch.manual_seed(0)
    weight = torch.randn(8, 32, 64).bfloat16()
    gathered = gather(weight, ranks)
    assert isinstance(gathered, NVFP4ExpertWeight)
    assert gathered.shape == weight.shape and gathered.dtype == torch.bfloat16
    for actual, expected in zip(gathered.quantized, fake_quantize(weight), strict=True):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape and actual.is_contiguous()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    # The GEMM takes the [experts, in, out] view; views keep the quantized tensors
    transposed = gathered.transpose(-2, -1)
    assert transposed._transposed and transposed.shape == (8, 64, 32)
    assert transposed.detach().quantized is gathered.quantized
    # The expert compute casts the GEMM weight to BF16: a same-dtype cast returns it as is, without dispatch
    assert transposed.bfloat16() is transposed


class Experts(nn.Module):
    def __init__(self, num_experts: int = 4):
        super().__init__()
        self.down_proj = nn.Parameter(torch.randn(num_experts, 32, 16))
        self.down_proj_bias = nn.Parameter(torch.randn(num_experts, 32))


def test_checkpoints_hold_plain_tensors(tmp_path):
    on, off = Experts(), Experts()
    all_gather_expert_weights_in_nvfp4(on, shard_size=1, quantize=fake_quantize)
    assert type(on.down_proj.data) is NVFP4ExpertMaster
    # Ops FSDP runs on the shard keep the master and its quantizer
    assert type(on.down_proj.data.clone()) is NVFP4ExpertMaster and on.down_proj.data.clone()._quantize is fake_quantize
    dcp.save(on.state_dict(), checkpoint_id=tmp_path / "on", no_dist=True)
    dcp.load(off.state_dict(), checkpoint_id=tmp_path / "on", no_dist=True)
    assert torch.equal(off.down_proj, on.down_proj._tensor)
    torch.save(on.state_dict(), tmp_path / "weights.pt")
    assert type(torch.load(tmp_path / "weights.pt", weights_only=True)["down_proj"]) is torch.Tensor
    with pytest.raises(ValueError, match="quantized_all_gather needs .* split evenly"):
        all_gather_expert_weights_in_nvfp4(Experts(num_experts=4), shard_size=3, quantize=fake_quantize)
