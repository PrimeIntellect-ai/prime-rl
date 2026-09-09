import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.overlapped_moe.api import run_overlapped_moe_layer_with_buffers
from prime_rl.trainer.distributed.overlapped_moe.buffers import init_overlapped_moe_buffers
from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher


def time_cuda(fn, n_warmup=5, n_iters=30):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    group = dist.group.WORLD

    num_local_tokens = 4096
    num_experts = 8 * world_size
    num_local_experts = num_experts // world_size
    top_k = 2
    hidden_dim = 256
    out_dim = 256
    block_m = 64
    n_comm_ctas = 16
    n_compute_ctas = 128

    torch.manual_seed(7)
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

    torch.manual_seed(1000 + rank)
    x = torch.randn(num_local_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
    logits = torch.randn(num_local_tokens, num_experts, device=device)
    top_scores, selected = torch.topk(logits, k=top_k, dim=1)
    top_scores = torch.softmax(top_scores, dim=-1)

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

    # Correctness at this size/CTA config before trusting any timing.
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
    print(f"[rank {rank}] correctness at bench size/CTA config: {ok}", flush=True)
    dist.barrier()
    assert ok, "correctness check failed at benchmark configuration"

    def ref_step():
        ref_dispatcher.run(x, top_scores, selected, reference_experts, score_before_experts=False)

    def overlapped_step():
        run_overlapped_moe_layer_with_buffers(
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

    ref_ms = time_cuda(ref_step)
    overlapped_ms = time_cuda(overlapped_step)

    if rank == 0:
        print(f"reference (plain NCCL dispatch + separate grouped GEMM + combine): {ref_ms:.4f} ms/iter")
        print(f"overlapped_moe (fused dispatch+GEMM, fused combine+reduce):             {overlapped_ms:.4f} ms/iter")
        print(f"speedup: {ref_ms / overlapped_ms:.2f}x")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
