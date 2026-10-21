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
    return parser.parse_args()


def bench(fwd, bwd, warmup: int, iters: int) -> tuple[float, float]:
    """Mean forward and backward ms, timed inside each fwd+bwd iteration since dist_moe pops its planner state
    only in backward."""
    for _ in range(warmup):
        bwd(fwd())
    torch.cuda.synchronize()
    dist.barrier()
    events = [[torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(iters)]
    for start, mid, end in events:
        start.record()
        y = fwd()
        mid.record()
        bwd(y)
        end.record()
    torch.cuda.synchronize()
    fwd_ms = sum(start.elapsed_time(mid) for start, mid, _ in events) / iters
    bwd_ms = sum(mid.elapsed_time(end) for _, mid, end in events) / iters
    return fwd_ms, bwd_ms


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

        base_fwd, base_bwd = bench(forward(baseline), backward, args.warmup, args.iters)
        fused_fwd, fused_bwd = bench(forward(fused), backward, args.warmup, args.iters)

        if rank == 0:
            print(f"\nrouting={routing}")
            print(f"{'':48} {'fwd':>10} {'bwd':>10} {'fwd+bwd':>10}")
            print(
                f"{'bf16 EP (TorchTokenDispatcher+BF16ExpertCompute)':48} "
                f"{base_fwd:9.3f}ms {base_bwd:9.3f}ms {base_fwd + base_bwd:9.3f}ms"
            )
            print(
                f"{'dist_moe (fused bf16 fwd + fused bf16 bwd)':48} "
                f"{fused_fwd:9.3f}ms {fused_bwd:9.3f}ms {fused_fwd + fused_bwd:9.3f}ms"
            )
            print(
                f"{'speedup (bf16 EP time / dist_moe time)':48} {base_fwd / fused_fwd:9.2f}x "
                f"{base_bwd / fused_bwd:9.2f}x {(base_fwd + base_bwd) / (fused_fwd + fused_bwd):9.2f}x"
            )
            print("relative diff vs bf16 EP path: " + ", ".join(f"{k}={v:.4f}" for k, v in diffs.items()))

    dist_moe_compute.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
