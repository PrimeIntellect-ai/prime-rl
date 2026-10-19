import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_common import ShardPlacementResult
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info
from torch.distributed.tensor import Shard
from torch.utils.checkpoint import checkpoint

from prime_rl.trainer.bf16_rounding import (
    BF16SRShard,
    bf16_rounding_disabled,
    install_bf16_stochastic_rounding,
    pop_cast_elements,
    set_bf16_rounding_step,
    sr_to_bf16,
)

pytestmark = [pytest.mark.gpu]


@pytest.fixture(scope="module")
def two_ranks():
    if os.environ.get("WORLD_SIZE") != "2":
        pytest.skip("run with torchrun --nproc-per-node=2")
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())
    yield dist.get_rank()
    dist.destroy_process_group()


class Recorded(torch.autograd.Function):
    """y = x @ w.T that records the weight seen in forward and in backward."""

    @staticmethod
    def forward(ctx, x, w, log):
        ctx.save_for_backward(x, w)
        ctx.log = log
        log.append(("fwd", w.detach().clone()))
        return x @ w.t()

    @staticmethod
    def backward(ctx, grad):
        x, w = ctx.saved_tensors
        ctx.log.append(("bwd", w.detach().clone()))
        return grad @ w, grad.t() @ x, None


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, 64, 64) * 0.1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Parameter(torch.randn(37, 64) * 0.1)
        self.out = nn.Parameter(torch.randn(64, 37) * 0.1)
        self.experts = Experts()
        self.router = nn.Linear(64, 2, bias=False)
        self.log: list = []

    def forward(self, x):
        def body(t):
            h = torch.tanh(Recorded.apply(t, self.dense, self.log))
            h = Recorded.apply(h, self.out, self.log)
            self.log.append(("expert", self.experts.weight.detach().clone()))
            return h + t @ self.experts.weight[0].t() + self.router(t).sum(-1, keepdim=True)

        return checkpoint(body, x, use_reentrant=False)


def build(seed: int = 0) -> tuple[Block, set]:
    torch.manual_seed(seed)
    block = Block().cuda()
    with torch.no_grad():
        block.experts.weight.copy_(torch.randn(2, 64, 64, generator=torch.Generator().manual_seed(1)).cuda() * 0.1)
    world = init_device_mesh("cuda", (1, 2), mesh_dim_names=("dp_shard_mod_ep", "ep"))
    dense_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
    expert_info = _get_mesh_info(world["dp_shard_mod_ep"])
    expert_params = set(block.experts.parameters())

    def placement(parameter):
        if parameter in expert_params:
            return ShardPlacementResult(placement=Shard(0), mesh_info=expert_info)
        return None

    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(block.router, mesh=dense_mesh, mp_policy=fp32)
    fully_shard(block, mesh=dense_mesh, mp_policy=bf16, shard_placement_fn=placement)
    return block, expert_params


def step_weights(block: Block, x: torch.Tensor) -> list:
    block.log.clear()
    block(x).sum().backward()
    return list(block.log)


def master(param: nn.Parameter) -> torch.Tensor:
    return param.detach().full_tensor().as_subclass(torch.Tensor)


def test_sr_cast_covers_dense_and_one_rank_groups_and_is_consistent_within_a_step(two_ranks):
    block, _ = build()
    counts = install_bf16_stochastic_rounding(block, base_seed=7)
    assert counts["dense_patched"] > 0 and counts["expert_patched"] > 0
    assert isinstance(block.dense._local_tensor, BF16SRShard)
    assert isinstance(block.experts.weight._local_tensor, BF16SRShard)
    assert not isinstance(block.router.weight._local_tensor, BF16SRShard)

    x = torch.randn(8, 64, device="cuda")
    set_bf16_rounding_step(1)
    pop_cast_elements()
    log = step_weights(block, x)
    cast = pop_cast_elements()
    assert cast["dense"] > 0 and cast["expert"] > 0

    dense_seen = [w for kind, w in log if kind != "expert" and w.shape == (37, 64)]
    assert len(dense_seen) == 3  # forward, recompute, backward
    assert all(torch.equal(dense_seen[0], w) for w in dense_seen[1:])
    assert dense_seen[0].dtype == torch.bfloat16

    rtn = master(block.dense).bfloat16()
    differs = (dense_seen[0] != rtn).float().mean().item()
    assert 0.15 < differs < 0.35  # E[min(f, 1 - f)] = 0.25 for uniform bin positions

    set_bf16_rounding_step(2)
    assert not torch.equal(step_weights(block, x)[0][1], dense_seen[0])
    set_bf16_rounding_step(1)
    assert torch.equal(step_weights(block, x)[0][1], dense_seen[0])

    with bf16_rounding_disabled():
        assert torch.equal(step_weights(block, x)[0][1], rtn)


def expert_weight(block: Block, x: torch.Tensor) -> torch.Tensor:
    block.log.clear()
    with torch.no_grad():
        block(x)
    block.reshard()
    return next(w for kind, w in block.log if kind == "expert")


def test_sr_cast_is_unbiased_and_independent_across_ranks(two_ranks):
    block, _ = build()
    install_bf16_stochastic_rounding(block, base_seed=3)
    x = torch.randn(4, 64, device="cuda")
    draws = 128
    total = torch.zeros(2, 64, 64, dtype=torch.float64, device="cuda")
    for step in range(draws):
        set_bf16_rounding_step(step)
        total += expert_weight(block, x).double()
    block.reshard()
    m = block.experts.weight.detach()._local_tensor.as_subclass(torch.Tensor).double()
    ulp = (m.float().abs().view(torch.int32) & -65536).view(torch.float32).double()
    ulp = ulp * 2.0**-7
    sr_error = ((total / draws - m).abs() / ulp).mean().item()
    rtn_error = ((m.float().bfloat16().double() - m).abs() / ulp).mean().item()
    assert sr_error < 0.1 < rtn_error

    # Both ranks hold identical expert values (same init), so different realizations mean independent noise.
    set_bf16_rounding_step(0)
    realized = expert_weight(block, x).contiguous()
    gathered = [torch.empty_like(realized) for _ in range(2)]
    dist.all_gather(gathered, realized)
    assert not torch.equal(gathered[0], gathered[1])


def test_gradients_optimizer_and_checkpoint_round_trip(two_ranks, tmp_path_factory):
    block, _ = build()
    install_bf16_stochastic_rounding(block, base_seed=11)
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-2)
    before = master(block.dense).clone()
    set_bf16_rounding_step(1)
    step_weights(block, torch.randn(8, 64, device="cuda"))
    assert block.dense.grad is not None and block.dense.grad.dtype == torch.float32
    assert block.experts.weight.grad is not None
    optimizer.step()
    optimizer.zero_grad()
    assert isinstance(block.dense._local_tensor, BF16SRShard)
    after = master(block.dense)
    assert not torch.equal(before, after)

    path = tmp_path_factory.getbasetemp() / "dcp" if dist.get_rank() == 0 else None
    shared = [str(path)]
    dist.broadcast_object_list(shared, src=0)
    dcp.save(block.state_dict(), checkpoint_id=shared[0])
    restored, _ = build(seed=5)
    install_bf16_stochastic_rounding(restored, base_seed=11)
    state = restored.state_dict()
    dcp.load(state, checkpoint_id=shared[0])
    assert torch.equal(master(restored.dense), after)
    assert isinstance(restored.dense._local_tensor, BF16SRShard)
    x = torch.randn(8, 64, device="cuda")
    set_bf16_rounding_step(3)
    assert torch.equal(step_weights(block, x)[0][1], step_weights(restored, x)[0][1])


def test_sr_keeps_nan_and_infinity():
    x = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0], device="cuda")
    x = torch.cat([x, torch.tensor([0x7FFFFFFF], dtype=torch.int32, device="cuda").view(torch.float32)])
    for seed in range(32):
        y = sr_to_bf16(x, seed).float()
        assert y[0].isnan() and y[4].isnan()
        assert y[1] == float("inf") and y[2] == -float("inf") and y[3] == 1.0
