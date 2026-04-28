"""Probe: does GDN output actually differ between multi-segment and
single-segment cu_seqlens?

Runs the *non-CP* forward of `Qwen3_5MoeGatedDeltaNet` twice on identical
input — once with the real packed multi-segment cu_seqlens, once with the
buggy single-segment fallback `[0, S]`. Reports the output diff.

If the diff is non-zero, that *mathematically* proves the multi-segment
plumbing matters (even when aggregate training loss doesn't show it at
random init). The CP all-to-all path uses the same conv1d + SSM scan
internally, so a divergence here is the same bug in the CP path.

Usage: uv run python scripts/probe_gdn_segments.py
"""

from __future__ import annotations

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from prime_rl.trainer.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet


def main() -> None:
    torch.manual_seed(0)

    config = Qwen3_5MoeTextConfig(
        vocab_size=2048,
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts_per_tok=2,
        num_experts=4,
        layer_types=["linear_attention", "full_attention"],
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    gdn = Qwen3_5MoeGatedDeltaNet(config).to(device=device, dtype=dtype)
    gdn.eval()

    seq_len = 1024
    hidden = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=dtype)

    # Multi-segment: 3 segments of lengths [200, 400, 424]
    cu_multi = torch.tensor([0, 200, 600, seq_len], dtype=torch.int32, device=device)
    # Single-segment fallback (the bug): one segment spanning the whole row
    cu_single = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

    with torch.no_grad():
        out_multi = gdn._forward_no_cp(hidden, cu_seqlens=cu_multi)
        out_single = gdn._forward_no_cp(hidden, cu_seqlens=cu_single)

    diff = (out_multi - out_single).abs()
    print(f"output shape: {tuple(out_multi.shape)}, dtype: {out_multi.dtype}")
    print(f"|out_multi|.mean = {out_multi.abs().mean().item():.6f}")
    print(f"|out_single|.mean = {out_single.abs().mean().item():.6f}")
    print(f"|diff|.max  = {diff.max().item():.6f}")
    print(f"|diff|.mean = {diff.mean().item():.6f}")
    print()

    # Per-segment view: where does the divergence concentrate?
    starts = cu_multi[:-1].tolist()
    ends = cu_multi[1:].tolist()
    print("Per-segment |diff|.mean (multi vs single):")
    for i, (s, e) in enumerate(zip(starts, ends)):
        seg_diff = diff[:, s:e].mean().item()
        print(f"  seg {i}: tokens [{s:4d}:{e:4d}] -> {seg_diff:.6f}")

    # Sanity: the FIRST segment should match (no prior state to leak from)
    first_seg_diff = diff[:, starts[0] : ends[0]].max().item()
    later_segs_diff = diff[:, ends[0] :].max().item()
    print()
    print(f"First-segment max diff:  {first_seg_diff:.6f}  (should be ~0)")
    print(f"Later-segment max diff:  {later_segs_diff:.6f}  (large => bug)")


if __name__ == "__main__":
    main()
