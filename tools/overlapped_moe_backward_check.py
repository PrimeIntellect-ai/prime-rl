import torch
import torch.distributed as dist
import torch.nn.functional as F

from prime_rl.trainer.distributed.overlapped_moe.autograd import (
    OverlappedMoELayerFunction,
    init_overlapped_moe_grad_buffers,
)
from prime_rl.trainer.distributed.overlapped_moe.buffers import init_overlapped_moe_buffers
from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    group = dist.group.WORLD

    num_local_tokens = 256
    num_experts = 4 * world_size
    num_local_experts = num_experts // world_size
    top_k = 2
    hidden_dim = 256
    intermediate = 128
    block_m = 128
    n_blocks = 16

    torch.manual_seed(7)
    gate_proj_data = (torch.randn(num_local_experts, intermediate, hidden_dim, device=device) * 0.02).to(torch.bfloat16)
    up_proj_data = (torch.randn(num_local_experts, intermediate, hidden_dim, device=device) * 0.02).to(torch.bfloat16)
    down_proj_data = (torch.randn(num_local_experts, hidden_dim, intermediate, device=device) * 0.02).to(torch.bfloat16)

    def reference_experts(rx: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        offs = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        up = torch._grouped_mm(rx.bfloat16(), up_proj_ref.transpose(-2, -1), offs=offs)
        gate = torch._grouped_mm(rx.bfloat16(), gate_proj_ref.transpose(-2, -1), offs=offs)
        act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        y = torch._grouped_mm(act, down_proj_ref.transpose(-2, -1), offs=offs)
        return y.type_as(rx)

    ref_dispatcher = TorchTokenDispatcher(
        num_experts=num_experts,
        top_k=top_k,
        token_group_alignment=1,
        group=group,
    )

    max_capacity = 16 * num_local_tokens * top_k
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

    n_iters = 5
    all_ok = True
    for it in range(n_iters):
        torch.manual_seed(1000 + rank + it * 7919)
        x_data = torch.randn(num_local_tokens, hidden_dim, device=device, dtype=torch.bfloat16) * 0.05
        logits = torch.randn(num_local_tokens, num_experts, device=device)
        top_scores_data, selected = torch.topk(logits, k=top_k, dim=1)
        top_scores_data = torch.softmax(top_scores_data, dim=-1)

        # reference path: independent leaves
        x_ref = x_data.clone().requires_grad_(True)
        top_scores_ref = top_scores_data.clone().float().requires_grad_(True)
        gate_proj_ref = gate_proj_data.clone().requires_grad_(True)
        up_proj_ref = up_proj_data.clone().requires_grad_(True)
        down_proj_ref = down_proj_data.clone().requires_grad_(True)

        ref_out = ref_dispatcher.run(x_ref, top_scores_ref, selected, reference_experts, score_before_experts=False)
        grad_seed = torch.randn_like(ref_out)
        (ref_out * grad_seed).sum().backward()

        # overlapped_moe path: independent leaves, same data/grad_seed
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
            nonlocal all_ok
            ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
            diff = (a.float() - b.float()).abs().max().item()
            if not ok:
                print(f"[rank {rank}] iter {it}: {name} MISMATCH (max diff {diff:.5f})", flush=True)
            all_ok = all_ok and ok

        cmp("forward output", cm_out, ref_out)
        cmp("grad_x", x_cm.grad, x_ref.grad)
        cmp("grad_top_scores", top_scores_cm.grad, top_scores_ref.grad)
        cmp("grad_gate_proj", gate_proj_cm.grad, gate_proj_ref.grad)
        cmp("grad_up_proj", up_proj_cm.grad, up_proj_ref.grad)
        cmp("grad_down_proj", down_proj_cm.grad, down_proj_ref.grad)

    print(f"[rank {rank}] {n_iters} iterations, all correct: {all_ok}", flush=True)

    dist.barrier()
    ok_tensor = torch.tensor([1 if all_ok else 0], device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        assert bool(ok_tensor.item()), "overlapped_moe forward/backward mismatch vs reference on some iteration"
        print(
            f"PASS: overlapped_moe forward+backward matches the plain-NCCL dispatcher reference over {n_iters} iterations",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
