import torch
import torch.distributed as dist
import torch.nn.functional as F

from prime_rl.trainer.distributed.overlapped_moe.autograd import (
    OverlappedMoELayerFunction,
    init_overlapped_moe_grad_buffers,
)
from prime_rl.trainer.distributed.overlapped_moe.buffers import init_overlapped_moe_buffers
from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher


def time_cuda(fn, n_warmup=5, n_iters=20):
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

    num_local_tokens = 1024
    num_experts = 128
    num_local_experts = num_experts // world_size
    top_k = 8
    hidden_dim = 2048
    intermediate = 768
    block_m = 128
    n_blocks = 132

    torch.manual_seed(7)
    gate_proj_data = (torch.randn(num_local_experts, intermediate, hidden_dim, device=device) * 0.02).to(torch.bfloat16)
    up_proj_data = (torch.randn(num_local_experts, intermediate, hidden_dim, device=device) * 0.02).to(torch.bfloat16)
    down_proj_data = (torch.randn(num_local_experts, hidden_dim, intermediate, device=device) * 0.02).to(torch.bfloat16)

    def reference_experts(rx, num_tokens_per_expert):
        offs = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        up = torch._grouped_mm(rx.bfloat16(), up_proj_ref.transpose(-2, -1), offs=offs)
        gate = torch._grouped_mm(rx.bfloat16(), gate_proj_ref.transpose(-2, -1), offs=offs)
        act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        y = torch._grouped_mm(act, down_proj_ref.transpose(-2, -1), offs=offs)
        return y.type_as(rx)

    ref_dispatcher = TorchTokenDispatcher(num_experts=num_experts, top_k=top_k, token_group_alignment=1, group=group)

    max_capacity = 4 * num_local_tokens * top_k + num_experts * block_m
    max_capacity = ((max_capacity + block_m - 1) // block_m) * block_m
    bufs = init_overlapped_moe_buffers(
        group,
        hidden_dim=hidden_dim,
        dispatch_capacity=max_capacity,
        combine_capacity=max_capacity,
        block_m=block_m,
        dtype=torch.bfloat16,
        device=device,
    )
    grad_recv, grad_combine = init_overlapped_moe_grad_buffers(
        group,
        hidden_dim=hidden_dim,
        dispatch_capacity=max_capacity,
        combine_capacity=max_capacity,
        block_m=block_m,
        dtype=torch.bfloat16,
        device=device,
    )
    from prime_rl.trainer.models.layers.activations import Silu

    torch.manual_seed(1000 + rank)
    x_data = torch.randn(num_local_tokens, hidden_dim, device=device, dtype=torch.bfloat16) * 0.05
    logits = torch.randn(num_local_tokens, num_experts, device=device)
    top_scores_data, selected = torch.topk(logits, k=top_k, dim=1)
    top_scores_data = torch.softmax(top_scores_data, dim=-1)

    # --- correctness first ---
    x_ref = x_data.clone().requires_grad_(True)
    top_scores_ref = top_scores_data.clone().float().requires_grad_(True)
    gate_proj_ref = gate_proj_data.clone().requires_grad_(True)
    up_proj_ref = up_proj_data.clone().requires_grad_(True)
    down_proj_ref = down_proj_data.clone().requires_grad_(True)
    ref_out = ref_dispatcher.run(x_ref, top_scores_ref, selected, reference_experts, score_before_experts=False)
    grad_seed = torch.randn_like(ref_out)
    (ref_out * grad_seed).sum().backward()

    x_cm = x_data.clone().requires_grad_(True)
    top_scores_cm = top_scores_data.clone().float().requires_grad_(True)
    gate_proj_cm = gate_proj_data.clone().requires_grad_(True)
    up_proj_cm = up_proj_data.clone().requires_grad_(True)
    down_proj_cm = down_proj_data.clone().requires_grad_(True)
    cm_out = OverlappedMoELayerFunction.apply(
        x_cm,
        top_scores_cm,
        selected,
        up_proj_cm,
        down_proj_cm,
        gate_proj_cm,
        Silu,
        bufs,
        grad_recv,
        grad_combine,
        group,
        num_experts,
        top_k,
        block_m,
        n_blocks,
    )
    (cm_out * grad_seed).sum().backward()

    def cmp(name, a, b, atol=2e-2, rtol=2e-2):
        d = (a.float() - b.float()).abs().max().item()
        is_ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
        print(f"[rank {rank}] {name}: ok={is_ok} max_diff={d:.5f}", flush=True)
        return is_ok

    ok = cmp("forward output", cm_out, ref_out)
    ok = cmp("grad_x", x_cm.grad, x_ref.grad) and ok
    ok = cmp("grad_top_scores", top_scores_cm.grad, top_scores_ref.grad) and ok
    ok = cmp("grad_gate_proj", gate_proj_cm.grad, gate_proj_ref.grad) and ok
    ok = cmp("grad_up_proj", up_proj_cm.grad, up_proj_ref.grad) and ok
    ok = cmp("grad_down_proj", down_proj_cm.grad, down_proj_ref.grad) and ok
    dist.barrier()
    assert ok, "correctness check failed at benchmark scale"

    # --- wallclock: one fwd+bwd step, both paths ---
    def ref_step():
        x_r = x_data.clone().requires_grad_(True)
        s_r = top_scores_data.clone().float().requires_grad_(True)
        nonlocal gate_proj_ref, up_proj_ref, down_proj_ref
        gate_proj_ref = gate_proj_data.clone().requires_grad_(True)
        up_proj_ref = up_proj_data.clone().requires_grad_(True)
        down_proj_ref = down_proj_data.clone().requires_grad_(True)
        out = ref_dispatcher.run(x_r, s_r, selected, reference_experts, score_before_experts=False)
        (out * grad_seed).sum().backward()

    def overlapped_step():
        x_c = x_data.clone().requires_grad_(True)
        s_c = top_scores_data.clone().float().requires_grad_(True)
        gp_c = gate_proj_data.clone().requires_grad_(True)
        up_c = up_proj_data.clone().requires_grad_(True)
        dp_c = down_proj_data.clone().requires_grad_(True)
        out = OverlappedMoELayerFunction.apply(
            x_c,
            s_c,
            selected,
            up_c,
            dp_c,
            gp_c,
            Silu,
            bufs,
            grad_recv,
            grad_combine,
            group,
            num_experts,
            top_k,
            block_m,
            n_blocks,
        )
        (out * grad_seed).sum().backward()

    ref_ms = time_cuda(ref_step)
    overlapped_ms = time_cuda(overlapped_step)

    if rank == 0:
        print(f"reference (TorchTokenDispatcher, fwd+bwd):  {ref_ms:.4f} ms/iter")
        print(f"overlapped_moe (OverlappedMoELayerFunction, fwd+bwd): {overlapped_ms:.4f} ms/iter")
        print(f"speedup: {ref_ms / overlapped_ms:.2f}x")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
