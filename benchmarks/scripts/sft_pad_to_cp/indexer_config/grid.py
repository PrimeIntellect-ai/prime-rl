"""Per-config kernel timings of _triton_fp8_indexer_kernel over realistic DSv4 packed layouts.

For each (layout, row length, cp rank) the script times every autotune config with the same
do_bench the autotuner uses (warmup=25, rep=100, median), on the stock output layout and on a
variant whose key axis is padded to a multiple of 16 (zero keys, so the extra logits are -inf).
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import triton

from prime_rl.trainer.models.kernels import fp8_indexer as mod

H, D, CP, RATE = 64, 128, 8, 4
kernel = mod._triton_fp8_indexer_kernel
ALL_CONFIGS = list(kernel.configs)


def name(c):
    return f"M{c.kwargs['BLOCK_M']}_N{c.kwargs['BLOCK_N']}_w{c.num_warps}_s{c.num_stages}"


def doc_lens(layout: str, total: int, rng: random.Random) -> list[int]:
    if layout == "one":
        return [total]
    if layout.startswith("fixed"):
        size = int(layout[5:])
        lens = [size] * (total // size)
        if total % size:
            lens.append(total % size)
        return lens
    if layout == "mixed":
        lens = []
        while sum(lens) < total:
            lens.append(int(min(2 ** rng.uniform(9, 17), total - sum(lens))))
        return lens
    raise ValueError(layout)


def windows(lens: list[int], rank: int):
    total = sum(lens)
    s_q = total // CP
    lens_t = torch.tensor(lens, device="cuda")
    cu = torch.cat([torch.zeros(1, device="cuda", dtype=torch.long), lens_t.cumsum(0)])
    counts = lens_t // RATE
    first_entry = counts.cumsum(0) - counts
    tok = torch.arange(rank * s_q, (rank + 1) * s_q, device="cuda")
    doc = torch.searchsorted(cu[1:], tok, right=True)
    pos = tok - cu[doc]
    ks = first_entry[doc].int()
    ke = (first_entry[doc] + (pos + 1) // RATE).int()
    return s_q, int(counts.sum()), ks, ke


def active_fraction(ks, ke, s_k, block_m=64, block_n=128):
    s_q = ks.numel()
    n_m = triton.cdiv(s_q, block_m)
    pad = n_m * block_m - s_q
    ks_p = torch.cat([ks, ks.new_full((pad,), s_k)]).view(n_m, block_m).amin(1)
    ke_p = torch.cat([ke, ke.new_zeros(pad)]).view(n_m, block_m).amax(1)
    n_n = triton.cdiv(s_k, block_n)
    lo = torch.arange(n_n, device=ks.device) * block_n
    live = (lo[None, :] < ke_p[:, None]) & (lo[None, :] + block_n > ks_p[:, None])
    return float(live.float().mean())


def prepare(s_q, s_k, ks, ke, k_pad_to: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(s_q, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(s_k, D, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(s_q, H, device="cuda", dtype=torch.bfloat16, generator=g)
    q_fp8, q_sc = mod.per_token_group_quant_fp8(q.reshape(s_q * H, D).contiguous(), group_size=D)
    k_fp8, k_sc = mod.per_token_group_quant_fp8(k, group_size=D)
    q_fp8 = q_fp8.view(s_q, H, D).permute(1, 0, 2).contiguous()
    w = w * q_sc.view(s_q, H)
    s_k_run = triton.cdiv(s_k, k_pad_to) * k_pad_to
    if s_k_run != s_k:
        k_fp8 = torch.cat([k_fp8, k_fp8.new_zeros(s_k_run - s_k, D)])
        k_sc = torch.cat([k_sc, k_sc.new_zeros(s_k_run - s_k, 1)])
    out = torch.empty(s_q, s_k_run, dtype=torch.float32, device="cuda")
    return q_fp8, k_fp8, k_sc, w, out, s_k_run


def time_configs(s_q, s_k, ks, ke, k_pad_to):
    q_fp8, k_fp8, k_sc, w, out, s_k_run = prepare(s_q, s_k, ks, ke, k_pad_to)
    grid = lambda meta: (triton.cdiv(s_q, meta["BLOCK_M"]), triton.cdiv(s_k_run, meta["BLOCK_N"]))
    args = (
        q_fp8,
        k_fp8,
        k_sc,
        w,
        out,
        ks,
        ke,
        s_q,
        s_k_run,
        q_fp8.stride(0),
        q_fp8.stride(1),
        k_fp8.stride(0),
        w.stride(0),
    )
    meta = dict(H=H, D=D, S_Q_BUCKET=triton.next_power_of_2(s_q), S_K_BUCKET=triton.next_power_of_2(s_k_run))
    res, wall = {}, {}
    try:
        for c in ALL_CONFIGS:
            kernel.configs = [c]
            fn = lambda: kernel[grid](*args, **meta)
            fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res[name(c)] = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
            wall[name(c)] = time.perf_counter() - t0
    finally:
        kernel.configs = ALL_CONFIGS
    return res, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layouts", default="one,fixed65536,fixed16384,fixed4096,mixed")
    p.add_argument("--totals", default="262144,252408,228776,201352,161928,80984")
    p.add_argument("--ranks", default="0,3,7")
    p.add_argument("--pads", default="1,16")
    args = p.parse_args()
    rows = []
    for layout in args.layouts.split(","):
        for total in map(int, args.totals.split(",")):
            lens = doc_lens(layout, total, random.Random(total))
            for rank in map(int, args.ranks.split(",")):
                s_q, s_k, ks, ke = windows(lens, rank)
                frac = active_fraction(ks, ke, s_k)
                for pad in map(int, args.pads.split(",")):
                    ms, wall = time_configs(s_q, s_k, ks, ke, pad)
                    row = dict(
                        layout=layout,
                        total=total,
                        rank=rank,
                        s_q=s_q,
                        s_k=s_k,
                        k_pad_to=pad,
                        n_docs=len(lens),
                        mean_doc=total / len(lens),
                        max_doc=max(lens),
                        active_frac=frac,
                        ms=ms,
                        do_bench_wall_s=wall,
                    )
                    rows.append(row)
                    best = min(ms.values())
                    print(
                        layout,
                        total,
                        rank,
                        s_q,
                        s_k,
                        pad,
                        f"frac={frac:.3f} best={best:.2f}",
                        " ".join(f"{k}:{v / best:.2f}" for k, v in ms.items()),
                        flush=True,
                    )
                args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
