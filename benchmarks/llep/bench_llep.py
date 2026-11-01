"""Speed, peak memory and exactness of expert-parallel MoE forward paths under routing imbalance.

Shapes default to the DeepSeek-V4-Flash routed experts (256 experts, top-6, hidden 4096, intermediate 2048,
SwiGLU clamped at 10, router weights applied after the down projection).

    torchrun --nproc_per_node=8 benchmarks/llep/bench_llep.py --tokens-per-rank 8192 \
        --methods ep,llep_torch --scenarios balanced,hot:50:4 --check
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    expert_ffn,
    gib,
    global_loads,
    imbalance_metrics,
    make_routing,
    reference_moe,
    rel_err,
)

from prime_rl.configs.trainer import LLEPConfig
from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--limit", type=float, default=10.0)
    parser.add_argument("--tokens-per-rank", type=int, default=8192)
    parser.add_argument("--methods", default="ep,llep_torch")
    parser.add_argument("--scenarios", default="balanced,hot:30:16,hot:50:16,hot:80:16,hot:95:16,hot:50:4,hot:95:1")
    parser.add_argument("--alpha", type=float, default=1.05)
    parser.add_argument("--min-chunk", type=int, default=1024)
    parser.add_argument("--lam", type=float, default=1.1)
    parser.add_argument("--max-foreign", type=int, default=8)
    parser.add_argument("--chunk-tokens", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--check", action="store_true", help="compare against an fp32 reference (needs full weights)")
    parser.add_argument("--backward", action="store_true", help="time forward + backward")
    parser.add_argument("--out", default=None, help="append JSON lines here")
    return parser.parse_args()


class Ctx(types.SimpleNamespace):
    pass


def method_ep(ctx, x, scores, topk):
    dispatcher = ctx.cache.setdefault(
        "ep", TorchTokenDispatcher(ctx.num_experts, ctx.top_k, token_group_alignment=8, group=ctx.group)
    )
    return dispatcher.run(
        x,
        scores,
        topk,
        lambda xr, n: expert_ffn(xr, n, ctx.gate_up, ctx.down, ctx.limit),
        score_before_experts=False,
    )


def method_ep_mxfp8(ctx, x, scores, topk):
    """Standard EP with MXFP8 dispatch transport (bf16 combine and expert compute)."""
    from prime_rl.trainer.distributed.token_dispatcher import MXFP8TorchTokenDispatcher

    dispatcher = ctx.cache.setdefault(
        "ep_mxfp8", MXFP8TorchTokenDispatcher(ctx.num_experts, ctx.top_k, token_group_alignment=8, group=ctx.group)
    )
    return dispatcher.run(
        x, scores, topk, lambda xr, n: expert_ffn(xr, n, ctx.gate_up, ctx.down, ctx.limit), score_before_experts=False
    )


def method_deepep(ctx, x, scores, topk):
    from prime_rl.trainer.distributed.deepep import DeepEPTokenDispatcher

    if "deepep" not in ctx.cache:
        ctx.cache["deepep"] = DeepEPTokenDispatcher(
            num_experts=ctx.num_experts, token_group_alignment=8, group=ctx.group, num_sms=20, token_chunk_size=None
        )
    dispatcher = ctx.cache["deepep"]
    out = dispatcher.run(
        x, scores, topk, lambda xr, n: expert_ffn(xr, n, ctx.gate_up, ctx.down, ctx.limit), score_before_experts=False
    )
    dispatcher.synchronize()
    return out


def method_llep_torch(ctx, x, scores, topk):
    """LLEP over torch EP: the routes are remapped to virtual experts and run through standard torch EP."""
    options = ctx.options
    slots = _ensure_slots(ctx)
    if "llep_dispatcher" not in ctx.cache:
        num_virtual = ctx.world * (ctx.experts_per_rank + options.max_foreign)
        ctx.cache["llep_dispatcher"] = TorchTokenDispatcher(num_virtual, ctx.top_k, 8, ctx.group)
    from deep_gemm.llep import plan_routes

    plan = plan_routes(topk, ctx.num_experts, ctx.group, options)
    foreign_gate_up, foreign_down = slots.push(plan.foreign, [ctx.gate_up, ctx.down])
    gate_up = torch.cat([ctx.gate_up, foreign_gate_up])
    down = torch.cat([ctx.down, foreign_down])
    return ctx.cache["llep_dispatcher"].run(
        x,
        scores,
        plan.vids,
        lambda xr, n: expert_ffn(xr, n, gate_up, down, ctx.limit),
        score_before_experts=False,
    )


def method_mega(ctx, x, scores, topk):
    from prime_rl.trainer.models.layers.activations import ClampedSilu
    from prime_rl.trainer.models.layers.mega_moe import MegaMoEExpertCompute, reserve_sms_for_comm

    if "mega" not in ctx.cache:
        ctx.cache["mega"] = MegaMoEExpertCompute(
            num_experts=ctx.num_experts,
            top_k=ctx.top_k,
            hidden=ctx.hidden,
            intermediate_hidden=ctx.intermediate,
            group=ctx.group,
            max_tokens_per_rank=ctx.tokens_per_rank,
            num_sms=reserve_sms_for_comm(16, 2),
        )
        ctx.cache["mega_experts"] = types.SimpleNamespace(
            gate_up_proj=ctx.gate_up,
            gate_proj=None,
            up_proj=None,
            down_proj=ctx.down,
            activation=ClampedSilu(ctx.limit),
        )
    return ctx.cache["mega"](ctx.cache["mega_experts"], x, scores, topk)


def _ensure_slots(ctx):
    from deep_gemm.llep import ForeignSlots

    if "slots" not in ctx.cache:
        ctx.cache["slots"] = ForeignSlots(
            ctx.group,
            ctx.options.max_foreign,
            [tuple(ctx.gate_up.shape[1:]), tuple(ctx.down.shape[1:])],
            ctx.gate_up.dtype,
        )
    return ctx.cache["slots"]


def _experts(ctx):
    from prime_rl.trainer.models.layers.activations import ClampedSilu

    return types.SimpleNamespace(
        gate_up_proj=ctx.gate_up, gate_proj=None, up_proj=None, down_proj=ctx.down, activation=ClampedSilu(ctx.limit)
    )


def method_mega_ag(ctx, x, scores, topk):
    """Mega MoE through prime-rl's autograd op (forward + backward kernels)."""
    from prime_rl.trainer.models.layers.mega_moe import MegaMoEExpertCompute, reserve_sms_for_comm

    if "mega_ag" not in ctx.cache:
        ctx.cache["mega_ag"] = MegaMoEExpertCompute(
            ctx.num_experts,
            ctx.top_k,
            ctx.hidden,
            ctx.intermediate,
            ctx.group,
            ctx.tokens_per_rank,
            reserve_sms_for_comm(16, 2),
        )
    return ctx.cache["mega_ag"](_experts(ctx), x, scores, topk)


def method_llep_mega_ag(ctx, x, scores, topk):
    """LLEP on Mega MoE through its autograd op."""
    from prime_rl.trainer.models.layers.llep_mega_moe import LLEPMegaMoEExpertCompute
    from prime_rl.trainer.models.layers.mega_moe import reserve_sms_for_comm

    if "llep_mega_ag" not in ctx.cache:
        ctx.cache["llep_mega_ag"] = LLEPMegaMoEExpertCompute(
            ctx.num_experts,
            ctx.top_k,
            ctx.hidden,
            ctx.intermediate,
            ctx.group,
            ctx.tokens_per_rank,
            reserve_sms_for_comm(16, 2),
            ctx.options,
        )
    return ctx.cache["llep_mega_ag"](_experts(ctx), x, scores, topk)


def method_llep_mega_ag_repush(ctx, x, scores, topk):
    """As `llep_mega_ag`, with the slots marked stale after the forward, as when other layers run in between."""
    from prime_rl.trainer.models.layers.llep_mega_moe import _STATES

    y = method_llep_mega_ag(ctx, x, scores, topk)
    _STATES[ctx.cache["llep_mega_ag"].state_key].slots.plan_id = None
    return y


def method_llep_mega(ctx, x, scores, topk):
    """LLEP on Mega MoE through the prime-rl op, forward only."""
    with torch.no_grad():
        return method_llep_mega_ag(ctx, x, scores, topk)


METHODS = {
    "ep": method_ep,
    "ep_mxfp8": method_ep_mxfp8,
    "deepep": method_deepep,
    "llep_torch": method_llep_torch,
    "mega": method_mega,
    "llep_mega": method_llep_mega,
    "mega_ag": method_mega_ag,
    "llep_mega_ag": method_llep_mega_ag,
    "llep_mega_ag_repush": method_llep_mega_ag_repush,
}


def static_bytes(ctx, name: str) -> int:
    """Buffers outside the caching allocator a method keeps across calls (symmetric memory, DeepEP)."""
    from prime_rl.trainer.models.layers.llep_mega_moe import _STATES
    from prime_rl.trainer.models.layers.mega_moe import _BUFFER_REGISTRY

    cache = ctx.cache
    slots_bytes = 0
    if "slots" in cache:
        slots = cache["slots"]
        slots_bytes = sum(part.buffer.nbytes for part in slots.weights + slots.grads)
    if name in ("mega", "mega_ag"):
        key = "mega" if name == "mega" else "mega_ag"
        _, buffers, _ = _BUFFER_REGISTRY[cache[key].buffer_key]
        return buffers.backward.buffer.nbytes
    if name in ("llep_mega", "llep_mega_ag", "llep_mega_ag_repush"):
        state = _STATES[cache["llep_mega_ag"].state_key]
        return state.buffer.buffer.nbytes + sum(p.buffer.nbytes for p in state.slots.weights + state.slots.grads)
    if name == "llep_torch":
        return slots_bytes
    if name == "deepep":
        from prime_rl.trainer.distributed import deepep

        return deepep._buffer.num_nvl_bytes + deepep._buffer.num_rdma_bytes
    return 0


def timed(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("NCCL_MAX_CTAS", "8")
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()
    experts_per_rank = args.num_experts // world
    ctx = Ctx(
        group=group,
        rank=rank,
        world=world,
        num_experts=args.num_experts,
        experts_per_rank=experts_per_rank,
        top_k=args.top_k,
        hidden=args.hidden,
        intermediate=args.intermediate,
        limit=args.limit,
        tokens_per_rank=args.tokens_per_rank,
        options=LLEPConfig(
            alpha=args.alpha,
            min_chunk=args.min_chunk,
            lam=args.lam,
            max_foreign=args.max_foreign,
            chunk_tokens=args.chunk_tokens,
        ),
        cache={},
    )

    gen = torch.Generator(device="cuda")
    if args.check:
        gen.manual_seed(1234)
        full_gate_up = (
            torch.randn(
                args.num_experts, 2 * args.intermediate, args.hidden, device="cuda", generator=gen, dtype=torch.bfloat16
            )
            * 0.02
        )
        full_down = (
            torch.randn(
                args.num_experts, args.hidden, args.intermediate, device="cuda", generator=gen, dtype=torch.bfloat16
            )
            * 0.02
        )
        ctx.gate_up = full_gate_up[rank * experts_per_rank : (rank + 1) * experts_per_rank].contiguous()
        ctx.down = full_down[rank * experts_per_rank : (rank + 1) * experts_per_rank].contiguous()
    else:
        gen.manual_seed(1234 + rank)
        ctx.gate_up = (
            torch.randn(
                experts_per_rank, 2 * args.intermediate, args.hidden, device="cuda", generator=gen, dtype=torch.bfloat16
            )
            * 0.02
        )
        ctx.down = (
            torch.randn(
                experts_per_rank, args.hidden, args.intermediate, device="cuda", generator=gen, dtype=torch.bfloat16
            )
            * 0.02
        )
    gen.manual_seed(99 + rank)
    x = torch.randn(args.tokens_per_rank, args.hidden, device="cuda", generator=gen, dtype=torch.bfloat16)
    dy = torch.randn(args.tokens_per_rank, args.hidden, device="cuda", generator=gen, dtype=torch.bfloat16)
    if args.backward:
        ctx.gate_up.requires_grad_()
        ctx.down.requires_grad_()

    if rank == 0:
        print(
            f"world={world} experts={args.num_experts} top_k={args.top_k} hidden={args.hidden} "
            f"intermediate={args.intermediate} tokens/rank={args.tokens_per_rank} options={ctx.options}",
            flush=True,
        )
    results = []
    for scenario in args.scenarios.split(","):
        scores, topk = make_routing(scenario, args.tokens_per_rank, args.num_experts, args.top_k, rank, world)
        metrics = imbalance_metrics(global_loads(topk, args.num_experts, group), args.top_k, world)
        reference = None
        if args.check:
            reference = reference_moe(x, scores, topk, full_gate_up, full_down, args.limit)
        outputs = {}
        for name in args.methods.split(","):
            fn = METHODS[name]
            torch.cuda.synchronize()
            dist.barrier()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            try:
                if args.backward:
                    xg = x.detach().requires_grad_()
                    sg = scores.detach().requires_grad_()

                    def step():
                        out = fn(ctx, xg, sg, topk)
                        out.reshape(x.shape).backward(dy)
                        return out

                    out = step().detach()
                    times = timed(step, args.warmup, args.iters)
                else:
                    with torch.no_grad():
                        out = fn(ctx, x, scores, topk)
                        times = timed(lambda: fn(ctx, x, scores, topk), args.warmup, args.iters)
                error = None
            except torch.OutOfMemoryError:
                times, out, error = [float("nan")], None, "OOM"
            peak = torch.tensor([torch.cuda.max_memory_allocated() - base], dtype=torch.float64, device="cuda")
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            t = torch.tensor(sorted(times)[len(times) // 2], dtype=torch.float64, device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            row = {
                "scenario": scenario,
                "method": name,
                "ms": t.item(),
                "peak_gib": gib(peak.item()),
                "error": error,
                **metrics,
            }
            if error is None:
                row["static_gib"] = gib(static_bytes(ctx, name))
            if out is not None:
                outputs[name] = out.reshape(x.shape)
                if reference is not None:
                    err = torch.tensor(rel_err(out.reshape(x.shape), reference), device="cuda", dtype=torch.float64)
                    dist.all_reduce(err, op=dist.ReduceOp.MAX)
                    row["rel_err"] = err.item()
            if name.startswith("llep"):
                from deep_gemm.llep import plan_routes

                plan = plan_routes(topk, ctx.num_experts, ctx.group, ctx.options)
                load = plan.rank_load.float()
                row["llep_rank_imbalance"] = (load.max() / load.mean()).item()
                row["num_foreign"] = int((plan.foreign >= 0).sum())
            results.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
        for reference_name, name in (("ep", "llep_torch"), ("mega", "llep_mega")):
            if reference_name in outputs and name in outputs:
                same = torch.tensor(float(torch.equal(outputs[name], outputs[reference_name])), device="cuda")
                dist.all_reduce(same, op=dist.ReduceOp.MIN)
                if rank == 0:
                    print(f"  {scenario}: {name} bitwise equal to {reference_name}: {bool(same.item())}", flush=True)
    if rank == 0 and args.out:
        with open(args.out, "a") as f:
            for row in results:
                f.write(
                    json.dumps({**row, "tokens_per_rank": args.tokens_per_rank, "world": world, "time": time.time()})
                    + "\n"
                )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
