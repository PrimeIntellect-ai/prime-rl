"""Benchmark meta-pytorch dist_moe (bf16 forward + backward) against prime-rl's bf16 EP path
(``TorchTokenDispatcher`` + ``BF16ExpertCompute``) for the same routed-expert workload, weights, and
token counts, and check the outputs and gradients agree.

Runs a force-balanced routing case (round-robin experts) and a random routing case (top-k of random
logits), each at the fixed ``--tokens-per-rank``. Requires 2-8 SM100+ GPUs in one node and the
``dist-moe`` extra. ``--hidden`` must be a multiple of 64.

Usage:
    torchrun --nproc_per_node=8 benchmarks/dist_moe/bench_dist_moe.py \
        --num-experts 128 --top-k 8 --hidden 2048 --intermediate 1024 --tokens-per-rank 2048
"""

import argparse
import os

import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.token_dispatcher import FusedTokenDispatcher, TorchTokenDispatcher
from prime_rl.trainer.models.fusions import fuse_gate_up_projections
from prime_rl.trainer.models.layers.dist_moe_experts import DistMoEExpertCompute
from prime_rl.trainer.models.layers.expert_compute import BF16ExpertCompute
from prime_rl.trainer.models.layers.moe import GroupedExperts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--tokens-per-rank", type=int, default=2048)
    parser.add_argument("--scratch-capacity-factor", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--host-lead-ms", type=float, default=20.0)
    return parser.parse_args()


def sleep_cycles(ms: float) -> int:
    """GPU clock cycles that ``torch.cuda._sleep`` needs to spin for about ``ms`` milliseconds."""
    probe_cycles = 10_000_000
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda._sleep(probe_cycles)
    start.record()
    torch.cuda._sleep(probe_cycles)
    end.record()
    end.synchronize()
    return int(probe_cycles * ms / start.elapsed_time(end))


def bench(fwd, bwd, warmup: int, iters: int, lead_cycles: int) -> torch.Tensor:
    """Per-iteration (fwd, bwd, fwd+bwd, back-to-back fwd+bwd) ms, each the max over ranks.

    A sleep kernel ahead of each GPU-timed iteration lets the host queue it all, so host launch time stays out of the
    first three; the back-to-back loop keeps it in. dist_moe pops its planner state only in backward, so forward and
    backward are timed within one iteration."""
    for _ in range(warmup):
        bwd(fwd())
    gpu_ms = []
    for _ in range(iters):
        start, mid, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda._sleep(lead_cycles)
        start.record()
        y = fwd()
        mid.record()
        bwd(y)
        end.record()
        end.synchronize()
        gpu_ms.append((start.elapsed_time(mid), mid.elapsed_time(end), start.elapsed_time(end)))
    torch.cuda.synchronize()
    dist.barrier()
    events = [[torch.cuda.Event(enable_timing=True) for _ in range(2)] for _ in range(iters)]
    for start, end in events:
        start.record()
        bwd(fwd())
        end.record()
    torch.cuda.synchronize()
    ms = torch.tensor(
        [(*row, start.elapsed_time(end)) for row, (start, end) in zip(gpu_ms, events, strict=True)], device="cuda"
    )
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return ms


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def route(scores: torch.Tensor, top_k: int, balanced: bool) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = scores.shape
    if balanced:
        token_offsets = torch.arange(num_tokens, device=scores.device).unsqueeze(1) * top_k
        selected_experts_indices = (token_offsets + torch.arange(top_k, device=scores.device)) % num_experts
        top_scores = scores.gather(1, selected_experts_indices)
    else:
        top_scores, selected_experts_indices = torch.topk(scores, top_k, dim=-1)
    return top_scores.contiguous(), selected_experts_indices.contiguous()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD

    if args.num_experts % world_size:
        raise ValueError(f"num_experts ({args.num_experts}) must be divisible by world_size ({world_size}).")

    torch.manual_seed(rank)
    experts_per_rank = args.num_experts // world_size
    device = torch.device("cuda", local_rank)

    experts = GroupedExperts(
        dim=args.hidden,
        hidden_dim=args.intermediate,
        num_experts=experts_per_rank,
    ).to(device=device, dtype=torch.bfloat16)
    experts.init_weights(init_std=0.02)
    fuse_gate_up_projections(experts)
    params = [experts.gate_up_proj, experts.down_proj]

    x = torch.randn(args.tokens_per_rank, args.hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
    dy = torch.randn(args.tokens_per_rank, args.hidden, dtype=torch.bfloat16, device=device)
    scores = torch.softmax(
        torch.randn(args.tokens_per_rank, args.num_experts, device=device, dtype=torch.float32), dim=-1
    )

    bf16_compute = BF16ExpertCompute()
    baseline = (
        TorchTokenDispatcher(
            num_experts=args.num_experts,
            top_k=args.top_k,
            token_group_alignment=bf16_compute.token_group_alignment,
            group=group,
        ),
        bf16_compute,
    )
    dist_moe_compute = DistMoEExpertCompute(
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden=args.hidden,
        intermediate_hidden=args.intermediate,
        group=group,
        max_tokens_per_rank=args.tokens_per_rank,
        num_moe_layers=1,
        scratch_capacity_factor=args.scratch_capacity_factor,
        vmm_capacity_factor=None,
        activation_slot_capacity_factor=None,
        num_reserved_sms=None,
    )
    fused = (FusedTokenDispatcher(), dist_moe_compute)
    lead_cycles = sleep_cycles(args.host_lead_ms)

    if rank == 0:
        print(
            f"world_size={world_size} num_experts={args.num_experts} (per rank={experts_per_rank}) "
            f"top_k={args.top_k} hidden={args.hidden} intermediate={args.intermediate} "
            f"tokens_per_rank={args.tokens_per_rank} scratch_capacity_factor={args.scratch_capacity_factor}"
        )

    for routing in ("balanced", "random"):
        top_scores, selected_experts_indices = route(scores, args.top_k, balanced=routing == "balanced")
        top_scores.requires_grad_(True)

        def forward(path):
            dispatcher, compute = path

            def fn():
                experts.compute = compute
                return dispatcher.run(x, top_scores, selected_experts_indices, experts, score_before_experts=False)

            return fn

        def backward(y: torch.Tensor) -> None:
            y.backward(dy)

        def grads_after(path):
            for p in params:
                p.grad = None
            x.grad = None
            top_scores.grad = None
            y = forward(path)()
            backward(y)
            torch.cuda.synchronize()
            return y.detach().clone(), x.grad.clone(), top_scores.grad.clone(), [p.grad.clone() for p in params]

        y_ref, dx_ref, ds_ref, dp_ref = grads_after(baseline)
        y_fused, dx_fused, ds_fused, dp_fused = grads_after(fused)
        diffs = {
            "y": rel_diff(y_fused, y_ref),
            "dx": rel_diff(dx_fused, dx_ref),
            "dtop_scores": rel_diff(ds_fused, ds_ref),
            "dw13": rel_diff(dp_fused[0], dp_ref[0]),
            "dw2": rel_diff(dp_fused[1], dp_ref[1]),
        }

        base_ms = bench(forward(baseline), backward, args.warmup, args.iters, lead_cycles)
        fused_ms = bench(forward(fused), backward, args.warmup, args.iters, lead_cycles)

        if rank == 0:
            base_med, fused_med = base_ms.median(dim=0).values.tolist(), fused_ms.median(dim=0).values.tolist()
            print(f"\nrouting={routing}")
            print(
                "GPU time (fwd, bwd, fwd+bwd) and back-to-back loop time per fwd+bwd: median, then fwd+bwd min/max, "
                f"over {args.iters} iterations of the max over ranks"
            )
            print(f"{'':48} {'fwd':>10} {'bwd':>10} {'fwd+bwd':>10} {'min':>10} {'max':>10} {'loop':>10}")
            for name, ms, med in (
                ("bf16 EP (TorchTokenDispatcher+BF16ExpertCompute)", base_ms, base_med),
                ("dist_moe (fused bf16 fwd + fused bf16 bwd)", fused_ms, fused_med),
            ):
                print(
                    f"{name:48} {med[0]:8.3f}ms {med[1]:8.3f}ms {med[2]:8.3f}ms "
                    f"{ms[:, 2].min().item():8.3f}ms {ms[:, 2].max().item():8.3f}ms {med[3]:8.3f}ms"
                )
            speedup = [b / f for b, f in zip(base_med, fused_med, strict=True)]
            print(
                f"{'speedup (bf16 EP time / dist_moe time)':48} {speedup[0]:9.2f}x {speedup[1]:9.2f}x "
                f"{speedup[2]:9.2f}x {'':10} {'':10} {speedup[3]:9.2f}x"
            )
            print("relative diff vs bf16 EP path: " + ", ".join(f"{k}={v:.4f}" for k, v in diffs.items()))

    dist_moe_compute.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
