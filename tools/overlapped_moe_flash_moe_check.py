import torch
import torch.nn.functional as F

from prime_rl.trainer.distributed.overlapped_moe.flash_moe_compute import (
    init_flash_moe_scratch,
    run_flash_moe_expert_compute,
)


def reference_ffn(hidden, tile_to_local_expert, gate_up_weight, down_weight, block_m):
    n_tiles = tile_to_local_expert.numel()
    out = torch.zeros_like(hidden)
    for t in range(n_tiles):
        e = int(tile_to_local_expert[t].item())
        rows = slice(t * block_m, (t + 1) * block_m)
        h = torch._grouped_mm(
            hidden[rows],
            gate_up_weight[e : e + 1].transpose(-2, -1),
            offs=torch.tensor([block_m], dtype=torch.int32, device=hidden.device),
        )
        gate, up = h.chunk(2, dim=-1)
        act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        y = torch._grouped_mm(
            act,
            down_weight[e : e + 1].transpose(-2, -1),
            offs=torch.tensor([block_m], dtype=torch.int32, device=hidden.device),
        )
        out[rows] = y
    return out


def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    num_local_experts = 4
    hidden_dim = 256  # K, must be a multiple of 128 (256 on the split bf16 path)
    intermediate = 128  # N = 2*intermediate must be a multiple of 128
    block_m = 128  # required by the kernel
    n_tiles = 6
    capacity = n_tiles * block_m

    torch.manual_seed(0)
    hidden = (torch.randn(capacity, hidden_dim, device=device) * 0.05).to(torch.bfloat16)
    gate_up_weight = (torch.randn(num_local_experts, 2 * intermediate, hidden_dim, device=device) * 0.05).to(
        torch.bfloat16
    )
    down_weight = (torch.randn(num_local_experts, hidden_dim, intermediate, device=device) * 0.05).to(torch.bfloat16)
    tile_to_local_expert = torch.randint(0, num_local_experts, (n_tiles,), dtype=torch.int32, device=device)

    scratch = init_flash_moe_scratch(capacity, capacity, hidden_dim, device)
    out = run_flash_moe_expert_compute(
        hidden,
        tile_to_local_expert,
        gate_up_weight,
        down_weight,
        scratch,
        block_m=block_m,
    )
    ref = reference_ffn(hidden, tile_to_local_expert, gate_up_weight, down_weight, block_m)

    diff = (out.float() - ref.float()).abs()
    rel_1 = (diff / ref.float().abs().clamp_min(1.0)).max().item()
    finite = torch.isfinite(out.float()).all().item()
    print(f"finite: {finite}, max_abs_diff: {diff.max().item():.5f}, rel_clamp_1: {rel_1:.5f}")
    ok = finite and rel_1 < 0.05
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()
