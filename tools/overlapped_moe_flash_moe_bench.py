import torch
import torch.distributed as dist
import torch.nn.functional as F

from prime_rl.trainer.distributed.overlapped_moe.api import run_overlapped_moe_layer_with_buffers_flash_moe
from prime_rl.trainer.distributed.overlapped_moe.buffers import init_overlapped_moe_buffers
from prime_rl.trainer.distributed.overlapped_moe.flash_moe_compute import init_flash_moe_scratch
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
    hidden_dim = 256  # K, must be a multiple of 256 on the mxfp8 path
    intermediate = 128  # N = 2*intermediate must be a multiple of 256
    block_m = 128  # fixed by flash_moe

    torch.manual_seed(7)  # same weights on every rank
    gate_up_weight = (torch.randn(num_local_experts, 2 * intermediate, hidden_dim, device=device) * 0.02).to(
        torch.bfloat16
    )
    down_weight = (torch.randn(num_local_experts, hidden_dim, intermediate, device=device) * 0.02).to(torch.bfloat16)

    def reference_experts(rx: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        offs = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        h = torch._grouped_mm(rx.bfloat16(), gate_up_weight.transpose(-2, -1), offs=offs)
        gate, up = h.chunk(2, dim=-1)
        act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        y = torch._grouped_mm(act, down_weight.transpose(-2, -1), offs=offs)
        return y.type_as(rx)

    ref_dispatcher = TorchTokenDispatcher(
        num_experts=num_experts,
        top_k=top_k,
        token_group_alignment=1,
        group=group,
    )
    # 4x headroom for imbalanced routing, plus `num_experts` tiles of alignment-padding overhead
    # per rank (see `metadata.py`'s `max_dispatch_tiles`) -- at small `num_local_tokens` and large
    # `num_experts` (e.g. ep_size=8 here), that padding term alone can exceed a plain 4x multiplier.
    max_capacity = 4 * num_local_tokens * top_k + num_experts * block_m
    # round up to a multiple of block_m=128 so symmetric-memory buffers meet flash_moe's tiling
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
    scratch = init_flash_moe_scratch(max_capacity, num_local_tokens * top_k, hidden_dim, device)

    n_iters = 10
    all_ok = True
    for it in range(n_iters):
        torch.manual_seed(1000 + rank + it * 7919)
        x = torch.randn(num_local_tokens, hidden_dim, device=device, dtype=torch.bfloat16) * 0.05
        logits = torch.randn(num_local_tokens, num_experts, device=device)
        top_scores, selected = torch.topk(logits, k=top_k, dim=1)
        top_scores = torch.softmax(top_scores, dim=-1)

        ref_out = ref_dispatcher.run(x, top_scores, selected, reference_experts, score_before_experts=False)
        overlapped_out = run_overlapped_moe_layer_with_buffers_flash_moe(
            x,
            top_scores,
            selected,
            gate_up_weight,
            down_weight,
            bufs,
            scratch,
            num_experts=num_experts,
            top_k=top_k,
            group=group,
            n_blocks=132,
        )

        diff = (overlapped_out.float() - ref_out.float()).abs()
        rel = (diff / ref_out.float().abs().clamp_min(1e-3)).max().item()
        max_diff = diff.max().item()
        ok = max_diff < 0.05 or rel < 0.1
        if not ok:
            print(f"[rank {rank}] iter {it}: MISMATCH (max diff {max_diff:.5f}, max rel {rel:.5f})", flush=True)
        all_ok = all_ok and ok

    print(f"[rank {rank}] {n_iters} iterations, all correct: {all_ok}", flush=True)

    dist.barrier()
    ok_tensor = torch.tensor([1 if all_ok else 0], device=device)
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN, group=group)
    if rank == 0:
        assert bool(ok_tensor.item()), "flash_moe overlapped_moe output mismatch vs reference on some iteration"
        print(
            f"PASS: flash_moe overlapped_moe end-to-end matches the plain-NCCL dispatcher reference over {n_iters} iterations",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
