"""End-to-end correctness check + wall-clock benchmark for the fused overlapped_moe prototype
(dispatch + grouped-GEMM + combine, CTA-specialized, in one/two Triton kernel launches).

Compares against `prime_rl.trainer.distributed.token_dispatcher.TorchTokenDispatcher` -- this
codebase's existing, already-validated plain-NCCL dispatcher -- as ground truth, using the same
routing, weights, and a real `torch._grouped_mm` expert function.

Not a pytest unit test: needs a real multi-GPU distributed environment (symmetric memory,
NCCL). Run with:

    uv run torchrun --nproc_per_node=<N> tools/overlapped_moe_bench.py
"""

import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.overlapped_moe.api import run_overlapped_moe_layer_with_buffers
from prime_rl.trainer.distributed.overlapped_moe.buffers import init_overlapped_moe_buffers
from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    group = dist.group.WORLD

    num_local_tokens = 512
    num_experts = 4 * world_size
    num_local_experts = num_experts // world_size
    top_k = 2
    hidden_dim = 128
    out_dim = 128
    block_m = 32
    n_comm_ctas = 8
    n_compute_ctas = 24

    torch.manual_seed(7)  # same weight on every rank
    weight = (torch.randn(num_local_experts, hidden_dim, out_dim, device=device) * 0.02).to(torch.bfloat16)

    def reference_experts(rx: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        offs = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        return torch._grouped_mm(rx.bfloat16(), weight, offs=offs).type_as(rx)

    ref_dispatcher = TorchTokenDispatcher(
        num_experts=num_experts,
        top_k=top_k,
        token_group_alignment=1,
        group=group,
    )
    # Symmetric memory requires identical buffer sizes on every rank, so these can't be sized
    # from this call's own (per-rank-different) routing outcome -- generous fixed upper bound.
    # Allocated *once* and reused every iteration: repeatedly allocating fresh symmetric memory
    # is not just wasteful, it hung after the first call (rendezvous is itself a collective; see
    # `run_overlapped_moe_layer_with_buffers`'s docstring).
    max_capacity = 4 * num_local_tokens * top_k
    bufs = init_overlapped_moe_buffers(
        group,
        hidden_dim=hidden_dim,
        dispatch_capacity=max_capacity,
        combine_capacity=max_capacity,
        block_m=block_m,
        dtype=torch.bfloat16,
        device=device,
    )

    n_iters = 20
    all_ok = True
    for it in range(n_iters):
        torch.manual_seed(1000 + rank + it * 7919)
        x = torch.randn(num_local_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
        logits = torch.randn(num_local_tokens, num_experts, device=device)
        top_scores, selected = torch.topk(logits, k=top_k, dim=1)
        top_scores = torch.softmax(top_scores, dim=-1)

        ref_out = ref_dispatcher.run(x, top_scores, selected, reference_experts, score_before_experts=False)
        overlapped_out = run_overlapped_moe_layer_with_buffers(
            x,
            top_scores,
            selected,
            weight,
            bufs,
            num_experts=num_experts,
            top_k=top_k,
            group=group,
            block_m=block_m,
            n_comm_ctas=n_comm_ctas,
            n_compute_ctas=n_compute_ctas,
        )

        ok = torch.allclose(overlapped_out.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
        max_diff = (overlapped_out.float() - ref_out.float()).abs().max().item()
        if not ok:
            print(f"[rank {rank}] iter {it}: MISMATCH (max diff {max_diff:.4f})", flush=True)
        all_ok = all_ok and ok

    print(f"[rank {rank}] {n_iters} iterations, all correct: {all_ok}", flush=True)

    dist.barrier()
    ok_tensor = torch.tensor([1 if all_ok else 0], device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        assert bool(ok_tensor.item()), "overlapped_moe output mismatch vs reference on some iteration"
        print(
            f"PASS: overlapped_moe end-to-end matches the existing plain-NCCL dispatcher over {n_iters} iterations",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
