"""Run DistMoEExpertCompute with real kernels under AC {none, full, selective} x compile {off, fullgraph}.

Checks the forward op runs once per step, no graph breaks, and outputs/grads match eager.
Usage: torchrun --nproc_per_node=8 check_compile_ac.py
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

import prime_rl.trainer.models.layers.dist_moe_experts as dme
from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
from prime_rl.trainer.distributed.token_dispatcher import FusedTokenDispatcher
from prime_rl.trainer.models.fusions import fuse_gate_up_projections
from prime_rl.trainer.models.layers.dist_moe_experts import DistMoEExpertCompute
from prime_rl.trainer.models.layers.moe import GroupedExperts

NUM_EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, TOKENS = 64, 8, 2048, 768, 4096

forward_calls = 0
_orig_args = dme._bf16_forward_args


def _counting_args(*args, **kwargs):
    global forward_calls
    if not torch._guards.TracingContext.try_get() and not isinstance(args[0], torch._subclasses.FakeTensor):
        forward_calls += 1
    return _orig_args(*args, **kwargs)


dme._bf16_forward_args = _counting_args


class Block(nn.Module):
    def __init__(self, experts, dispatcher):
        super().__init__()
        self.experts = experts
        self.dispatcher = dispatcher

    def forward(self, x, top_scores, idx):
        y = self.dispatcher.run(x, top_scores, idx, self.experts, score_before_experts=False)
        return y * 2 + x


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(rank)

    experts = GroupedExperts(dim=HIDDEN, hidden_dim=INTERMEDIATE, num_experts=NUM_EXPERTS // world).to(
        device=device, dtype=torch.bfloat16
    )
    experts.init_weights(init_std=0.02)
    fuse_gate_up_projections(experts)
    experts.compute = DistMoEExpertCompute(
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        hidden=HIDDEN,
        intermediate_hidden=INTERMEDIATE,
        group=dist.group.WORLD,
        max_tokens_per_rank=TOKENS,
        num_moe_layers=1,
        scratch_capacity_factor=2.0,
        vmm_capacity_factor=None,
        activation_slot_capacity_factor=None,
        num_reserved_sms=None,
    )
    params = [experts.gate_up_proj, experts.down_proj]
    block = Block(experts, FusedTokenDispatcher())

    x0 = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
    dy = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
    scores0 = torch.softmax(torch.randn(TOKENS, NUM_EXPERTS, device=device), dim=-1)
    s0, idx = torch.topk(scores0, TOP_K, dim=-1)

    def step(module):
        global forward_calls
        for p in params:
            p.grad = None
        x = x0.clone().requires_grad_(True)
        s = s0.clone().requires_grad_(True)
        forward_calls = 0
        y = module(x, s, idx.contiguous())
        y.backward(dy)
        torch.cuda.synchronize()
        return forward_calls, [y.detach(), x.grad, s.grad, *[p.grad for p in params]]

    _, ref = step(block)
    names = ["y", "dx", "dscores", "dw13", "dw2"]
    for ac in ("none", "full", "selective"):
        for compile_mode in ("off", "fullgraph"):
            torch._dynamo.reset()
            module = (
                block
                if ac == "none"
                else get_activation_checkpoint_wrapper(SimpleNamespace(mode=ac, targets=None))(block)
            )
            if compile_mode == "fullgraph":
                module = torch.compile(module, fullgraph=True)
            try:
                step(module)
                calls, out = step(module)
                diffs = [
                    ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()
                    for a, b in zip(out, ref)
                ]
                bitwise = all(torch.equal(a, b) for a, b in zip(out, ref))
                calls_t = torch.tensor([calls], device=device)
                dist.all_reduce(calls_t, op=dist.ReduceOp.MAX)
                if rank == 0:
                    print(
                        f"ac={ac:9} compile={compile_mode:9} fwd_op_calls_per_step={calls_t.item()} bitwise={bitwise} "
                        + " ".join(f"{n}={d:.2e}" for n, d in zip(names, diffs)),
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"rank{rank} ac={ac} compile={compile_mode} FAILED {type(e).__name__}: {str(e)[:500]}", flush=True
                )
                raise
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
