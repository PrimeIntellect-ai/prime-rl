"""Autotune bench_time and chosen config under different config lists and do_bench budgets."""

import argparse
import functools
import json
import random
from pathlib import Path

import torch
import triton
from grid import ALL_CONFIGS, D, H, doc_lens, kernel, name, windows
from padded_op import padded_indexer

from prime_rl.trainer.models.kernels import fp8_indexer as mod

PRUNED = [c for c in ALL_CONFIGS if name(c) in ("M64_N64_w4_s2", "M64_N128_w4_s2", "M64_N128_w8_s3")]
BENCHES = {
    "default_25_100": None,
    "fast_5_20": functools.partial(triton.testing.do_bench, warmup=5, rep=20, return_mode="median"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for layout, total, rank in [("one", 252408, 7), ("fixed16384", 252408, 7), ("fixed4096", 252408, 7)]:
        lens = doc_lens(layout, total, random.Random(total))
        s_q, s_k, ks, ke = windows(lens, rank)
        g = torch.Generator(device="cuda").manual_seed(1)
        q = torch.randn(s_q, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
        k = torch.randn(s_k, D, device="cuda", dtype=torch.bfloat16, generator=g)
        w = torch.randn(s_q, H, device="cuda", dtype=torch.bfloat16, generator=g)
        for op_name, op in [("stock", mod.fp8_indexer), ("padded16", padded_indexer)]:
            for cfg_name, cfgs in [("all6", ALL_CONFIGS), ("pruned3", PRUNED)]:
                for bench_name, bench in BENCHES.items():
                    for trial in range(2):
                        kernel.configs = cfgs
                        kernel.cache.clear()
                        kernel.__dict__.pop("do_bench", None)
                        if bench is not None:
                            kernel.__dict__["do_bench"] = lambda fn, quantiles, b=bench: b(fn)
                        op(q, k, w, ks, ke, 512)
                        torch.cuda.synchronize()
                        row = dict(
                            layout=layout,
                            total=total,
                            rank=rank,
                            op=op_name,
                            configs=cfg_name,
                            bench=bench_name,
                            trial=trial,
                            bench_time_s=kernel.bench_time,
                            chosen=name(kernel.best_config),
                        )
                        rows.append(row)
                        print(json.dumps(row), flush=True)
    kernel.configs = ALL_CONFIGS
    kernel.__dict__.pop("do_bench", None)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
