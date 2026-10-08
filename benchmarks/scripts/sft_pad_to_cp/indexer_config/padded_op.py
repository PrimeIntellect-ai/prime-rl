"""Full fp8_indexer op: stock vs key axis padded to a multiple of 16, bitwise check and timing per config."""

import argparse
import json
import random
from pathlib import Path

import torch
import triton

from prime_rl.trainer.models.kernels import fp8_indexer as mod

from grid import ALL_CONFIGS, H, D, doc_lens, kernel, name, windows

TOPK = 512


def padded_indexer(q, k, w, ks, ke, topk, pad_to=16):
    S_q, H_, D_ = q.shape
    S_k = k.shape[0]
    S_k_pad = triton.cdiv(S_k, pad_to) * pad_to
    q_fp8, q_scales = mod.per_token_group_quant_fp8(q.reshape(S_q * H_, D_).contiguous(), group_size=D_)
    k_fp8, k_scales = mod.per_token_group_quant_fp8(k.contiguous(), group_size=D_)
    if S_k_pad != S_k:
        k_fp8 = torch.cat([k_fp8, k_fp8.new_zeros(S_k_pad - S_k, D_)])
        k_scales = torch.cat([k_scales, k_scales.new_zeros(S_k_pad - S_k, 1)])
    q_fp8 = q_fp8.view(S_q, H_, D_).permute(1, 0, 2).contiguous()
    w = w * q_scales.view(S_q, H_)
    logits = torch.empty(S_q, S_k_pad, dtype=torch.float32, device=q.device)
    grid = lambda meta: (triton.cdiv(S_q, meta["BLOCK_M"]), triton.cdiv(S_k_pad, meta["BLOCK_N"]))
    kernel[grid](
        q_fp8, k_fp8, k_scales, w, logits, ks, ke, S_q, S_k_pad,
        q_fp8.stride(0), q_fp8.stride(1), k_fp8.stride(0), w.stride(0),
        H=H_, D=D_, S_Q_BUCKET=triton.next_power_of_2(S_q), S_K_BUCKET=triton.next_power_of_2(S_k_pad),
    )
    actual_topk = min(topk, S_k)
    _, indices = torch.topk(logits, actual_topk, dim=-1)
    if actual_topk < topk:
        indices = torch.cat([indices, indices.new_full((S_q, topk - actual_topk), S_k)], dim=-1)
    out_of_range = (indices < ks.unsqueeze(1)) | (indices >= ke.unsqueeze(1))
    return indices.masked_fill(out_of_range, S_k).to(torch.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = []
    cases = [("one", 252408, 7), ("fixed16384", 252408, 7), ("fixed4096", 252408, 7), ("mixed", 228776, 3),
             ("fixed4096", 80984, 0), ("one", 262144, 7)]
    for layout, total, rank in cases:
        lens = doc_lens(layout, total, random.Random(total))
        s_q, s_k, ks, ke = windows(lens, rank)
        g = torch.Generator(device="cuda").manual_seed(1)
        q = torch.randn(s_q, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(s_k, D, device="cuda", dtype=torch.bfloat16, generator=g)
        w = torch.randn(s_q, H, device="cuda", dtype=torch.bfloat16, generator=g)
        per_config = {}
        try:
            for c in ALL_CONFIGS:
                kernel.configs = [c]
                ref = mod.fp8_indexer(q, k, w, ks, ke, TOPK)
                new = padded_indexer(q, k, w, ks, ke, TOPK)
                t_ref = triton.testing.do_bench(lambda: mod.fp8_indexer(q, k, w, ks, ke, TOPK), warmup=25, rep=200, return_mode="median")
                t_new = triton.testing.do_bench(lambda: padded_indexer(q, k, w, ks, ke, TOPK), warmup=25, rep=200, return_mode="median")
                per_config[name(c)] = dict(bitwise_equal=bool(torch.equal(ref, new)),
                                           rows_differ=int((ref != new).any(-1).sum()), stock_ms=t_ref, padded_ms=t_new)
        finally:
            kernel.configs = ALL_CONFIGS
        row = dict(layout=layout, total=total, rank=rank, s_q=s_q, s_k=s_k, per_config=per_config)
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
