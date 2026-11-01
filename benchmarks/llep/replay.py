"""Replay recorded DeepSeek-V4-Flash routing (per-expert loads of every layer of an EP group's micro step, from the
EP-imbalance run's dumps, HF dataset PrimeIntellect/llep-dsv4f-routing-counts) through the B300 recipe's MoE paths,
forward and backward timed separately:

- `ep_qat`: the recipe, torch all-to-all dispatch with MXFP8 QAT expert compute (stochastic weight rounding);
- `mega`: Mega MoE `mxfp8_qat` (MXFP8 forward with the same weight rounding, bf16 backward);
- `llep` / `llep_chunk`: LLEP on Mega MoE `mxfp8_qat`, in one call or in `--chunk-tokens` chunks.

Each rank samples its tokens' 6 distinct experts so that every expert's global load matches the recording. LLEP runs
with the backward re-push (`--reuse` keeps the slots, as when the forward was just recomputed by full AC).

    torchrun --nproc_per_node=8 benchmarks/llep/replay.py --dumps <dir> --tokens-per-rank 32768 --cases 30:0:0
"""

import argparse
import json
import os
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist

from prime_rl.configs.trainer import LLEPConfig
from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher
from prime_rl.trainer.models.fp8_rounding import FP8RoundingState, _stream_id
from prime_rl.trainer.models.layers.activations import ClampedSilu
from prime_rl.trainer.models.layers.llep_mega_moe import _STATES, LLEPMegaMoEExpertCompute
from prime_rl.trainer.models.layers.mega_moe import MegaMoEExpertCompute, reserve_sms_for_comm
from prime_rl.trainer.models.layers.mxfp8_qat import MXFP8QATExpertCompute

E, K, H, I, LIMIT = 256, 6, 4096, 2048, 10.0


def exact_routing(counts: torch.Tensor, tokens_per_rank: int, world: int, rank: int, seed: int) -> torch.Tensor:
    """This rank's [tokens, K] experts such that every expert's global load is the recording's, scaled to
    `world * tokens_per_rank * K` routes, with distinct experts per token."""
    num_tokens = world * tokens_per_rank
    total = num_tokens * K
    scaled = counts.double() / counts.sum() * total
    loads = scaled.floor().long()
    remainder = total - int(loads.sum())
    loads[torch.argsort(scaled - loads, descending=True)[:remainder]] += 1
    assert int(loads.max()) <= num_tokens, "an expert would need two routes of one token"
    routes = torch.repeat_interleave(torch.arange(E), loads)
    topk = routes.view(K, num_tokens).T
    order = torch.randperm(num_tokens, generator=torch.Generator().manual_seed(seed))
    return topk[order][rank * tokens_per_rank : (rank + 1) * tokens_per_rank].contiguous().cuda()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", required=True, help="directory of step_*_group_*.pt routing dumps")
    parser.add_argument("--tokens-per-rank", type=int, default=32768)
    parser.add_argument("--cases", default="30:0:0", help="step:group_first_rank:micro, comma separated")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--methods", default="ep_qat,mega,llep")
    parser.add_argument("--chunk-tokens", type=int, default=None, help="for the llep_chunk method")
    parser.add_argument("--reuse", action="store_true", help="LLEP backward reuses the forward's slots")
    parser.add_argument("--num-reserved-sms", type=int, default=16)
    parser.add_argument("--nolb-lam", type=float, default=2.5, help="lam = max_recv of the nolb methods")
    parser.add_argument("--alpha", type=float, default=1.05, help="LLEP methods")
    parser.add_argument("--max-recv", type=float, default=2.0, help="LLEP methods")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--balanced", action="store_true", help="every expert gets the same load (the ideal)")
    parser.add_argument("--check", action="store_true", help="compare every method's outputs with the first method's")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    methods = args.methods.split(",")
    mega = any(name != "ep_qat" for name in methods)
    # Mega MoE caps NCCL to fit its reserved SMs; torch EP alone runs at NCCL's default, its best setting
    if mega:
        os.environ.setdefault("NCCL_MAX_CTAS", str(args.num_reserved_sms // 2))
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    per_rank = E // world
    T = args.tokens_per_rank
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)
    gate_up = (torch.randn(per_rank, 2 * I, H, device="cuda", generator=gen) * 0.02).bfloat16()
    down = (torch.randn(per_rank, H, I, device="cuda", generator=gen) * 0.02).bfloat16()
    x0 = torch.randn(T, H, device="cuda", generator=gen).bfloat16()
    dy = torch.randn_like(x0)
    # The recipe's stochastic weight rounding: one stream per step, salted per weight, at the global expert index
    rounding = FP8RoundingState(0, torch.empty((), device="cuda", dtype=torch.int64))
    rounding.set_step(1)
    salts = {
        name: torch.tensor(_stream_id(f"layer.experts.{name}"), device="cuda") for name in ("gate_up_proj", "down_proj")
    }
    # One persistent tensor: the Mega MoE weight cache (MEGA_MOE_WEIGHT_CACHE_LAYERS) keys on it
    mega_salts = torch.stack([salts["gate_up_proj"], salts["down_proj"]])
    num_sms = reserve_sms_for_comm(args.num_reserved_sms, 2) if mega else None
    computes = {}
    for name in methods:
        if name == "mega":
            computes[name] = MegaMoEExpertCompute(E, K, H, I, group, T, num_sms, "mxfp8_qat", stochastic_rounding=True)
        elif name in ("llep", "llep_chunk", "nolb", "nolb_chunk"):
            # nolb: LLEP's kernels with a plan that never copies at this routing (standard EP)
            chunked = name.endswith("_chunk")
            options = LLEPConfig(
                chunk_tokens=args.chunk_tokens if chunked else None, alpha=args.alpha, max_recv=args.max_recv
            )
            if name.startswith("nolb"):
                options = LLEPConfig(chunk_tokens=options.chunk_tokens, lam=args.nolb_lam, max_recv=args.nolb_lam)
            computes[name] = LLEPMegaMoEExpertCompute(
                E, K, H, I, group, T, num_sms, options, "mxfp8_qat", stochastic_rounding=True
            )
        elif name == "rail":
            # Two nodes: rail swaps over InfiniBand, LLEP on each node's experts (EP group = WORLD of 16)
            from prime_rl.trainer.models.layers.rail_llep_mega_moe import RailLLEPMegaMoEExpertCompute

            options = LLEPConfig(chunk_tokens=args.chunk_tokens, alpha=args.alpha, max_recv=args.max_recv)
            computes[name] = RailLLEPMegaMoEExpertCompute(
                E, K, H, I, group, T, num_sms, options, "mxfp8_qat", stochastic_rounding=True, ranks_per_node=8
            )
        elif name == "ep_qat":
            computes[name] = (TorchTokenDispatcher(E, K, 128, group), MXFP8QATExpertCompute(stochastic_rounding=True))

    def step(name, scores, topk, split=None):
        gu, dn = gate_up.clone().requires_grad_(), down.clone().requires_grad_()
        experts = types.SimpleNamespace(
            gate_up_proj=gu,
            gate_proj=None,
            up_proj=None,
            down_proj=dn,
            activation=ClampedSilu(LIMIT),
            gate_proj_bias=None,
            up_proj_bias=None,
            down_proj_bias=None,
            fp8_rounding=rounding,
            rounding_salts=salts,
            fp8_expert_offset=rank * per_rank,
            mega_rounding_salts=mega_salts,
        )
        x = x0.clone().requires_grad_()
        s = scores.clone().requires_grad_()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        if name == "ep_qat":
            dispatcher, compute = computes[name]
            y = dispatcher.run(x, s, topk, lambda xr, counts: compute(experts, xr, counts), score_before_experts=False)
        else:
            y = computes[name](experts, x, s, topk)
            if name.startswith(("llep", "nolb", "rail")) and not args.reuse:
                # Other layers' forwards run between this forward and its backward
                state = _STATES[computes[name].state_key]
                state.slots.plan_id = state.fp8_slots.plan_id = None
        torch.cuda.synchronize()
        middle = time.perf_counter()
        y.backward(dy)
        torch.cuda.synchronize()
        if split is not None:
            split.append((middle - start, time.perf_counter() - middle))
        return y.detach(), x.grad, s.grad

    rows = []
    for case in args.cases.split(","):
        step_idx, first_rank, micro = (int(v) for v in case.split(":"))
        dump = torch.load(Path(args.dumps) / f"step_{step_idx:05d}_group_{first_rank:04d}.pt", weights_only=False)
        counts = dump["ep_counts"][micro].float()  # [layers, experts]
        if args.balanced:
            counts = torch.ones_like(counts)
        layers = range(counts.shape[0]) if args.layers == "all" else [int(v) for v in args.layers.split(",")]
        for layer in layers:
            topk = exact_routing(
                counts[layer], T, world, rank, seed=((step_idx * 64 + first_rank) * 8 + micro) * 64 + layer
            )
            g = torch.Generator(device="cuda").manual_seed(rank)
            scores = torch.softmax(torch.randn(T, K, device="cuda", generator=g), dim=-1) * 1.5
            loads = torch.bincount(topk.reshape(-1), minlength=E).float()
            dist.all_reduce(loads)
            rank_loads = loads.view(world, -1).sum(1)
            recorded = counts[layer].view(world, -1).sum(1)
            reference = None
            for name in computes:
                torch.cuda.reset_peak_memory_stats()
                outputs = step(name, scores, topk)
                if args.check:
                    if reference is None:
                        reference = outputs
                    else:
                        errs = torch.stack([((a.float() - b.float()).norm() / b.float().norm()) for a, b in zip(outputs, reference)])
                        dist.all_reduce(errs, op=dist.ReduceOp.MAX)
                        if rank == 0:
                            print(f"check layer {layer} {name} vs {next(iter(computes))}: relative error y {errs[0]:.2e} "
                                  f"dx {errs[1]:.2e} dscores {errs[2]:.2e}", flush=True)
                split = []
                for _ in range(args.iters):
                    step(name, scores, topk, split)
                fwd = sorted(f for f, _ in split)[len(split) // 2] * 1e3
                bwd = sorted(b for _, b in split)[len(split) // 2] * 1e3
                stats = torch.tensor([fwd, bwd, torch.cuda.max_memory_allocated() / 2**30], device="cuda")
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
                fwd, bwd, peak = stats.tolist()
                row = {
                    "step": step_idx,
                    "group": first_rank,
                    "micro": micro,
                    "layer": layer,
                    "method": name,
                    "tokens_per_rank": T,
                    "chunk_tokens": args.chunk_tokens if name == "llep_chunk" else None,
                    "reuse": args.reuse,
                    "balanced": args.balanced,
                    "ms": fwd + bwd,
                    "fwd_ms": fwd,
                    "bwd_ms": bwd,
                    "peak_gib": peak,
                    "rank_imb_recorded": (recorded.max() / recorded.mean()).item(),
                    "rank_imb_sampled": (rank_loads.max() / rank_loads.mean()).item(),
                }
                rows.append(row)
                if rank == 0:
                    print(json.dumps(row), flush=True)
    if rank == 0:
        with open(args.out, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
