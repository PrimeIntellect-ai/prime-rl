"""DeepSeek-V4-Flash MoE layers with Mega MoE vs LLEP-on-Mega dispatch, eager and under torch.compile.

Routing imbalance comes from the router's selection bias (the aux-loss-free load-balancing buffer): a bias on
the first experts pulls tokens to them without touching the routing weights. Checks that LLEP's outputs and
input gradients equal Mega MoE's bit for bit, compares expert gradients, times both, and counts recompiles
as the imbalance changes from step to step.

    torchrun --nproc_per_node=8 benchmarks/llep/check_moe_layers.py --compile
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

from prime_rl.configs.trainer import LLEPConfig
from prime_rl.trainer.distributed.token_dispatcher import FusedTokenDispatcher
from prime_rl.trainer.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from prime_rl.trainer.models.deepseek_v4.moe import DeepseekV4MoE
from prime_rl.trainer.models.fusions import fuse_gate_up_projections
from prime_rl.trainer.models.layers.llep_mega_moe import LLEPMegaMoEExpertCompute
from prime_rl.trainer.models.layers.mega_moe import MegaMoEExpertCompute, reserve_sms_for_comm


def build_layers(config, num_layers, rank, world, compute):
    torch.manual_seed(0)
    layers = torch.nn.ModuleList()
    for i in range(num_layers):
        moe = DeepseekV4MoE(config, layer_idx=i).cuda()
        moe.init_weights(0.02, torch.device("cuda"))
        fuse_gate_up_projections(moe.experts)
        per_rank = moe.experts.num_experts // world
        for name in ("gate_up_proj", "down_proj"):
            full = getattr(moe.experts, name)
            setattr(
                moe.experts, name, torch.nn.Parameter(full[rank * per_rank : (rank + 1) * per_rank].detach().clone())
            )
        moe.experts.set_compute(compute)
        moe.set_token_dispatcher(FusedTokenDispatcher())
        moe.to(torch.bfloat16)
        moe.router.selection_bias = moe.router.selection_bias.float()
        layers.append(moe)
    return layers


def run(layers, x, dy):
    h = x
    for moe in layers:
        h = h + moe(h)
    h.backward(dy)
    return h.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-rank", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--biases", default="0,2,4,8", help="selection bias on the hot experts, one step each")
    parser.add_argument("--hot", type=int, default=8, help="number of hot experts (the first ones, on rank 0)")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("NCCL_MAX_CTAS", "8")
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    config = DeepseekV4Config(num_hash_layers=0, n_shared_experts=0)
    num_sms = reserve_sms_for_comm(16, 2)
    shape = (config.n_routed_experts, config.num_experts_per_tok, config.hidden_size, config.moe_intermediate_size)
    computes = {
        "mega": MegaMoEExpertCompute(*shape, group, args.tokens_per_rank, num_sms),
        "llep": LLEPMegaMoEExpertCompute(*shape, group, args.tokens_per_rank, num_sms, LLEPConfig()),
    }
    stacks = {name: build_layers(config, args.layers, rank, world, compute) for name, compute in computes.items()}
    gen = torch.Generator(device="cuda").manual_seed(rank)
    x0 = torch.randn(1, args.tokens_per_rank, config.hidden_size, device="cuda", generator=gen).bfloat16()
    dy = torch.randn_like(x0)

    import torch._dynamo.utils as dynamo_utils

    fns = {name: (torch.compile(run) if args.compile else run) for name in stacks}
    for bias in [float(b) for b in args.biases.split(",")]:
        for layers in stacks.values():
            for moe in layers:
                moe.router.selection_bias.zero_()
                moe.router.selection_bias[: args.hot] = bias
        results = {}
        for name, layers in stacks.items():
            for moe in layers:
                moe.zero_grad(set_to_none=True)
            x = x0.clone().requires_grad_()
            y = fns[name](layers, x, dy)
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(3):
                for moe in layers:
                    moe.zero_grad(set_to_none=True)
                xt = x0.clone().requires_grad_()
                fns[name](layers, xt, dy)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) / 3 * 1e3
            grads = [p.grad for moe in layers for p in (moe.experts.gate_up_proj, moe.experts.down_proj)]
            results[name] = (y, x.grad, grads, ms)
        loads = torch.stack([moe.tokens_per_expert for moe in stacks["mega"]]).sum(0)
        dist.all_reduce(loads)
        rank_load = loads.view(world, -1).sum(1)
        mega, llep = results["mega"], results["llep"]
        # Mega MoE against itself: how deterministic the reference is
        for moe in stacks["mega"]:
            moe.zero_grad(set_to_none=True)
        x_again = x0.clone().requires_grad_()
        y_again = fns["mega"](stacks["mega"], x_again, dy)
        flags = torch.tensor(
            [
                float(torch.equal(mega[0], llep[0])),
                float(torch.equal(mega[1], llep[1])),
                float(torch.equal(mega[0], y_again)),
                float(torch.equal(mega[1], x_again.grad)),
            ],
            device="cuda",
        )
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        dw_err = max(((a.float() - b.float()).norm() / b.float().norm()).item() for a, b in zip(llep[2], mega[2]))
        dw_err = torch.tensor(dw_err, device="cuda")
        dist.all_reduce(dw_err, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"bias={bias:4.1f} rank_imbalance={(rank_load.max() / rank_load.mean()).item():.2f} "
                f"mega={mega[3]:.1f}ms llep={llep[3]:.1f}ms llep==mega y={bool(flags[0])} dx={bool(flags[1])} "
                f"mega==mega y={bool(flags[2])} dx={bool(flags[3])} dW rel={dw_err.item():.2e}",
                flush=True,
            )
    if rank == 0 and args.compile:
        print("dynamo counters:", dict(dynamo_utils.counters["stats"]), dict(dynamo_utils.counters["recompiles"]))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
