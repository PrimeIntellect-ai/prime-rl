"""Proposed fixed-config, key-aligned fp8_indexer vs stock per-config op times, plus bitwise checks."""

import json
import random

import torch
import triton

import grid
import proposed_mod
from grid import ALL_CONFIGS, D, doc_lens, kernel, name
from prime_rl.trainer.models.kernels import fp8_indexer as mod

CASES = [
    (64, 4, "one", 252408, 7), (64, 4, "fixed16384", 252408, 7), (64, 4, "fixed4096", 252408, 7),
    (64, 4, "mixed", 228776, 3), (64, 4, "fixed4096", 80984, 0), (64, 4, "one", 262144, 7),
    (64, 4, "mixed", 201352, 7), (64, 4, "one", 8008, 7), (64, 4, "mixed", 30002, 0), (64, 4, "one", 30002, 7),
    (32, 1, "one", 65000, 7), (32, 1, "fixed4096", 65000, 7), (32, 1, "mixed", 65000, 3), (32, 1, "one", 12345, 5),
]


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=200, return_mode="median")


rows = []
for h, rate, layout, total, rank in CASES:
    grid.RATE = rate
    lens = doc_lens(layout, total, random.Random(total))
    s_q, s_k, ks, ke = grid.windows(lens, rank)
    g = torch.Generator(device="cuda").manual_seed(3)
    q = torch.randn(s_q, h, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(s_k, D, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(s_q, h, device="cuda", dtype=torch.bfloat16, generator=g)
    new = proposed_mod.fp8_indexer(q, k, w, ks, ke, 512)
    t_new = bench(lambda: proposed_mod.fp8_indexer(q, k, w, ks, ke, 512))
    stock_ms, equal = {}, {}
    try:
        for c in ALL_CONFIGS:
            kernel.configs = [c]
            kernel.cache.clear()
            ref = mod.fp8_indexer(q, k, w, ks, ke, 512)
            equal[name(c)] = bool(torch.equal(ref, new))
            stock_ms[name(c)] = bench(lambda: mod.fp8_indexer(q, k, w, ks, ke, 512))
    finally:
        kernel.configs = ALL_CONFIGS
        kernel.cache.clear()
    best = min(stock_ms, key=stock_ms.get)
    row = dict(h=h, rate=rate, layout=layout, total=total, rank=rank, s_q=s_q, s_k=s_k, proposed_ms=t_new,
               stock_best=best, stock_best_ms=stock_ms[best], stock_worst_ms=max(stock_ms.values()),
               proposed_over_best=t_new / stock_ms[best], all_bitwise_equal=all(equal.values()), stock_ms=stock_ms)
    rows.append(row)
    print(json.dumps({k_: v for k_, v in row.items() if k_ != "stock_ms"}), flush=True)
    json.dump(rows, open("validate_ext.json", "w"), indent=1)
