"""Benchmark DeepGEMM's fused Mega MoE kernels (bf16 forward + backward) against prime-rl's
bf16 EP path (``TorchTokenDispatcher`` + ``BF16GroupedGemm``) for the same routed-expert
workload, weights, and token counts, and check the gradients agree.

Requires >=2 GPUs with symmetric-memory support (SM100/Blackwell) and PyTorch >= 2.9.
``--hidden`` must be a multiple of 256 and ``--intermediate`` a multiple of 128.

Usage:
    torchrun --nproc_per_node=8 benchmarks/mega_moe/bench_mega_moe.py \
        --num-experts 128 --top-k 8 --hidden 2048 --intermediate 1024 --tokens-per-rank 2048
"""

import argparse
import os

import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.mega_moe_dispatcher import MegaMoeTokenDispatcher
from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher
from prime_rl.trainer.models.layers.grouped_gemm import BF16GroupedGemm
from prime_rl.trainer.models.layers.mega_moe import mega_moe_available
from prime_rl.trainer.models.layers.moe import GroupedExperts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--tokens-per-rank", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD

    if not mega_moe_available():
        raise RuntimeError("Mega MoE requires a deep_gemm build with bf16 Mega MoE forward+backward on an SM100+ GPU.")
    if args.num_experts % world_size:
        raise ValueError(f"num_experts ({args.num_experts}) must be divisible by world_size ({world_size}).")

    torch.manual_seed(rank)
    experts_per_rank = args.num_experts // world_size
    device = torch.device("cuda", local_rank)

    experts = GroupedExperts(
        dim=args.hidden,
        hidden_dim=args.intermediate,
        num_experts=experts_per_rank,
        grouped_gemm=BF16GroupedGemm(),
    ).to(device=device, dtype=torch.bfloat16)
    experts.init_weights(init_std=0.02)
    params = [p for p in experts.parameters() if p is not None]

    x = torch.randn(args.tokens_per_rank, args.hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
    dy = torch.randn(args.tokens_per_rank, args.hidden, dtype=torch.bfloat16, device=device)
    scores = torch.softmax(
        torch.randn(args.tokens_per_rank, args.num_experts, device=device, dtype=torch.float32), dim=-1
    )
    top_scores, selected_experts_indices = torch.topk(scores, args.top_k, dim=-1)
    top_scores = top_scores.reshape(-1).contiguous().requires_grad_(True)
    selected_experts_indices = selected_experts_indices.reshape(-1).contiguous()

    baseline = TorchTokenDispatcher(
        num_experts=args.num_experts,
        top_k=args.top_k,
        token_group_alignment=experts.token_group_alignment,
        group=group,
    )
    mega = MegaMoeTokenDispatcher(
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden=args.hidden,
        intermediate_hidden=args.intermediate,
        group=group,
        max_tokens_per_rank=args.tokens_per_rank,
    )

    def run(dispatcher, backward: bool):
        def fn():
            y = dispatcher.run(x, top_scores, selected_experts_indices, experts, score_before_experts=False)
            if backward:
                y.backward(dy)
            return y

        return fn

    def grads_after(fn):
        for p in params:
            p.grad = None
        x.grad = None
        top_scores.grad = None
        y = fn()
        torch.cuda.synchronize()
        return y.detach().clone(), x.grad.clone(), top_scores.grad.clone(), [p.grad.clone() for p in params]

    y_ref, dx_ref, ds_ref, dp_ref = grads_after(run(baseline, backward=True))
    y_mega, dx_mega, ds_mega, dp_mega = grads_after(run(mega, backward=True))
    diffs = {
        "y": rel_diff(y_mega, y_ref),
        "dx": rel_diff(dx_mega, dx_ref),
        "dtop_scores": rel_diff(ds_mega, ds_ref),
        "dexperts": max(rel_diff(a, b) for a, b in zip(dp_mega, dp_ref)),
    }

    def bench_fwd_bwd(dispatcher):
        fwd = bench(run(dispatcher, backward=False), args.warmup, args.iters)
        total = bench(run(dispatcher, backward=True), args.warmup, args.iters)
        return fwd, total

    base_fwd, base_total = bench_fwd_bwd(baseline)
    mega_fwd, mega_total = bench_fwd_bwd(mega)

    if rank == 0:
        print(
            f"world_size={world_size} num_experts={args.num_experts} (per rank={experts_per_rank}) "
            f"top_k={args.top_k} hidden={args.hidden} intermediate={args.intermediate} "
            f"tokens_per_rank={args.tokens_per_rank}"
        )
        print(f"{'':46} {'fwd':>10} {'bwd':>10} {'fwd+bwd':>10}")
        print(
            f"{'bf16 EP (TorchTokenDispatcher+BF16GroupedGemm)':46} "
            f"{base_fwd:9.3f}ms {base_total - base_fwd:9.3f}ms {base_total:9.3f}ms"
        )
        print(
            f"{'Mega MoE (fused bf16 fwd + fused bf16 bwd)':46} "
            f"{mega_fwd:9.3f}ms {mega_total - mega_fwd:9.3f}ms {mega_total:9.3f}ms"
        )
        print(
            f"{'speedup':46} {base_fwd / mega_fwd:9.2f}x "
            f"{(base_total - base_fwd) / (mega_total - mega_fwd):9.2f}x {base_total / mega_total:9.2f}x"
        )
        print("relative diff vs bf16 EP path: " + ", ".join(f"{k}={v:.4f}" for k, v in diffs.items()))

    mega.buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
