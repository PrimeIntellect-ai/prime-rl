"""Probe: does NemotronH Mamba2 SSM scan output actually differ between
multi-segment and single-segment seq_idx?

Mirrors scripts/probe_gdn_segments.py but for NemotronH's CP path. Calls
`mamba_chunk_scan_combined` (the kernel that `cp_mamba.py:mamba_cp_forward`
uses internally) twice on identical inputs:

  - seq_idx=None       → what `cp_mamba.py:202` currently passes (the bug)
  - seq_idx=segment_id → the correct multi-segment behaviour NVIDIA Megatron
                         threads via packed_seq_params.seq_idx

If outputs diverge in segments after the first, that proves the NemotronH
CP forward leaks SSM state across packed-sequence boundaries — same shape
of bug we just fixed in GDN.

Usage: uv run --group mamba-ssm python scripts/probe_nemotronh_mamba_segments.py
"""

from __future__ import annotations

import torch
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Mirror NemotronH-3-Nano-30B Mamba2 mixer dims (per its config.json):
    #   mamba_num_heads=64, mamba_head_dim=64, n_groups=8, ssm_state_size,
    # The exact ssm_state_size doesn't change the bug; pick a small one.
    batch_size = 1
    seq_len = 1024
    num_heads = 16        # smaller for faster probe (still divisible by groups)
    head_dim = 64
    n_groups = 4
    state_size = 64
    chunk_size = 128

    # Inputs to mamba_chunk_scan_combined — same layout the mamba_cp_forward
    # uses after slicing local heads/groups (see cp_mamba.py L193–207):
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    dt = torch.rand(batch_size, seq_len, num_heads, device=device, dtype=dtype) * 0.1
    A = -torch.rand(num_heads, device=device, dtype=torch.float32).abs()
    B = torch.randn(batch_size, seq_len, n_groups, state_size, device=device, dtype=dtype)
    C = torch.randn(batch_size, seq_len, n_groups, state_size, device=device, dtype=dtype)
    D = torch.randn(num_heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, device=device, dtype=torch.float32)

    # Multi-segment packed batch: 3 segments of lengths [200, 400, 424].
    cu_seqlens = torch.tensor([0, 200, 600, seq_len], dtype=torch.int32, device=device)
    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_idx = torch.repeat_interleave(
        torch.arange(seg_lens.numel(), dtype=torch.int32, device=device),
        seg_lens,
    ).unsqueeze(0)  # [1, S]

    common = dict(
        x=x, dt=dt, A=A, B=B, C=C,
        chunk_size=chunk_size,
        D=D, dt_bias=dt_bias, dt_softplus=True,
    )
    with torch.no_grad():
        out_correct = mamba_chunk_scan_combined(seq_idx=seq_idx, **common)
        out_broken = mamba_chunk_scan_combined(seq_idx=None, **common)

    diff = (out_correct - out_broken).abs()
    print(f"output shape: {tuple(out_correct.shape)}, dtype: {out_correct.dtype}")
    print(f"|out_correct|.mean = {out_correct.abs().mean().item():.6f}")
    print(f"|out_broken |.mean = {out_broken.abs().mean().item():.6f}")
    print(f"|diff|.max  = {diff.max().item():.6f}")
    print(f"|diff|.mean = {diff.mean().item():.6f}")
    print()

    starts = cu_seqlens[:-1].tolist()
    ends = cu_seqlens[1:].tolist()
    print("Per-segment |diff|.mean (correct vs broken):")
    for i, (s, e) in enumerate(zip(starts, ends)):
        seg_diff = diff[:, s:e].mean().item()
        print(f"  seg {i}: tokens [{s:4d}:{e:4d}] -> {seg_diff:.6f}")

    first_seg_diff = diff[:, starts[0] : ends[0]].max().item()
    later_segs_diff = diff[:, ends[0] :].max().item()
    print()
    print(f"First-segment max diff:  {first_seg_diff:.6f}  (should be ~0)")
    print(f"Later-segment max diff:  {later_segs_diff:.6f}  (large => bug)")


if __name__ == "__main__":
    main()
