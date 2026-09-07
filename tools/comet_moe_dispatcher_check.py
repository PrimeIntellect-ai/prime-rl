import torch
import torch.distributed as dist

from prime_rl.trainer.distributed.comet_moe.token_dispatcher import CometMoETokenDispatcher
from prime_rl.trainer.distributed.token_dispatcher import TorchTokenDispatcher
from prime_rl.trainer.models.layers.moe import GroupedExperts


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
    dim = 256
    intermediate = 128

    torch.manual_seed(7)
    experts_ref = (
        GroupedExperts(dim=dim, hidden_dim=intermediate, num_experts=num_local_experts).to(device).to(torch.bfloat16)
    )
    experts_cm = (
        GroupedExperts(dim=dim, hidden_dim=intermediate, num_experts=num_local_experts).to(device).to(torch.bfloat16)
    )
    experts_ref.init_weights(0.02)
    with torch.no_grad():
        experts_cm.gate_proj.copy_(experts_ref.gate_proj)
        experts_cm.up_proj.copy_(experts_ref.up_proj)
        experts_cm.down_proj.copy_(experts_ref.down_proj)
    for p in experts_ref.parameters():
        p.requires_grad_(True)
    for p in experts_cm.parameters():
        p.requires_grad_(True)

    ref_dispatcher = TorchTokenDispatcher(num_experts=num_experts, top_k=top_k, token_group_alignment=1, group=group)
    cm_dispatcher = CometMoETokenDispatcher(num_experts=num_experts, top_k=top_k, group=group, block_m=32, n_blocks=16)

    n_iters = 5
    all_ok = True
    for it in range(n_iters):
        torch.manual_seed(1000 + rank + it * 7919)
        x_data = torch.randn(num_local_tokens, dim, device=device, dtype=torch.bfloat16) * 0.05
        logits = torch.randn(num_local_tokens, num_experts, device=device)
        top_scores_data, selected = torch.topk(logits, k=top_k, dim=1)
        top_scores_data = torch.softmax(top_scores_data, dim=-1)

        x_ref = x_data.clone().requires_grad_(True)
        top_scores_ref = top_scores_data.clone().float().requires_grad_(True)
        ref_out = ref_dispatcher.run(x_ref, top_scores_ref, selected, experts_ref, score_before_experts=False)
        grad_seed = torch.randn_like(ref_out)
        (ref_out * grad_seed).sum().backward()

        x_cm = x_data.clone().requires_grad_(True)
        top_scores_cm = top_scores_data.clone().float().requires_grad_(True)
        cm_out = cm_dispatcher.run(x_cm, top_scores_cm, selected, experts_cm, score_before_experts=False)
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
        cmp("grad_gate_proj", experts_cm.gate_proj.grad, experts_ref.gate_proj.grad)
        cmp("grad_up_proj", experts_cm.up_proj.grad, experts_ref.up_proj.grad)
        cmp("grad_down_proj", experts_cm.down_proj.grad, experts_ref.down_proj.grad)
        experts_ref.zero_grad()
        experts_cm.zero_grad()

    print(f"[rank {rank}] {n_iters} iterations, all correct: {all_ok}", flush=True)

    dist.barrier()
    ok_tensor = torch.tensor([1 if all_ok else 0], device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        assert bool(ok_tensor.item()), "CometMoETokenDispatcher mismatch vs TorchTokenDispatcher reference"
        print(f"PASS: CometMoETokenDispatcher matches TorchTokenDispatcher over {n_iters} iterations", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
