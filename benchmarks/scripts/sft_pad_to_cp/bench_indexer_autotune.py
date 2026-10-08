"""Cost of re-tuning the fp8_indexer Triton kernel when the per-rank query count S_Q changes.

Shapes follow DeepSeek-V4-Flash at seq_len 262144 with cp=8: H=64, D=128, topk=512, S_Q near
32768 queries per rank and S_K = 2 * S_Q compressed entries (compress rate 4 over the whole row).
Rank `rank` holds the contiguous query chunk [rank * S_Q, (rank + 1) * S_Q) of the row.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
import triton

from prime_rl.trainer.models.kernels import fp8_indexer as fp8_indexer_module
from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer

H, D, TOPK, CP, COMPRESS_RATE = 64, 128, 512, 8, 4

kernel = fp8_indexer_module._triton_fp8_indexer_kernel
ALL_CONFIGS = list(kernel.configs)


def make_inputs(s_q: int, rank: int, doc_len: int | None, seed: int = 0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    s_k = s_q * CP // COMPRESS_RATE
    q = torch.randn(s_q, H, D, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(s_k, D, device="cuda", dtype=torch.bfloat16, generator=generator)
    w = torch.randn(s_q, H, device="cuda", dtype=torch.bfloat16, generator=generator)
    global_pos = torch.arange(rank * s_q, (rank + 1) * s_q, device="cuda")
    doc_start = torch.zeros_like(global_pos) if doc_len is None else (global_pos // doc_len) * doc_len
    ks = (doc_start // COMPRESS_RATE).int()
    ke = (doc_start // COMPRESS_RATE + (global_pos - doc_start + 1) // COMPRESS_RATE).int()
    return q, k, w, ks, ke


def timed_call(inputs) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fp8_indexer(*inputs, TOPK)
    torch.cuda.synchronize()
    return time.perf_counter() - start, out


def config_name(config: triton.Config) -> str:
    return f"M{config.kwargs['BLOCK_M']}_N{config.kwargs['BLOCK_N']}_w{config.num_warps}_s{config.num_stages}"


def retune_sweep(s_q_values: list[int], rank: int, doc_len: int | None, steady_reps: int) -> list[dict]:
    rows = []
    for s_q in s_q_values:
        inputs = make_inputs(s_q, rank, doc_len)
        tunes_before = len(kernel.cache)
        first_s, _ = timed_call(inputs)
        retuned = len(kernel.cache) > tunes_before
        steady = statistics.median(timed_call(inputs)[0] for _ in range(steady_reps))
        best = kernel.best_config
        rows.append(
            {
                "s_q": s_q,
                "s_q_mod16": s_q % 16,
                "first_call_s": first_s,
                "steady_call_s": steady,
                "retune_s": first_s - steady,
                "retuned": retuned,
                "bench_time_s": getattr(kernel, "bench_time", None) if retuned else None,
                "best_config": config_name(best),
                "config_median_ms": (
                    {config_name(c): t[0] for c, t in kernel.configs_timings.items()} if retuned else None
                ),
                "num_tuning_keys": len(kernel.cache),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        del inputs
    return rows


def topk_identity(s_q: int, rank: int, doc_len: int | None) -> dict:
    inputs = make_inputs(s_q, rank, doc_len)
    reference = None
    results = {}
    try:
        for config in ALL_CONFIGS:
            kernel.configs = [config]
            indices = fp8_indexer(*inputs, TOPK)
            if reference is None:
                reference = indices
            sorted_ref, sorted_idx = reference.sort(dim=-1).values, indices.sort(dim=-1).values
            results[config_name(config)] = {
                "bitwise_equal": bool(torch.equal(indices, reference)),
                "rows_differ": int((indices != reference).any(dim=-1).sum()),
                "set_rows_differ": int((sorted_idx != sorted_ref).any(dim=-1).sum()),
            }
    finally:
        kernel.configs = ALL_CONFIGS
    return {"s_q": s_q, "reference": config_name(ALL_CONFIGS[0]), "per_config": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=CP - 1)
    parser.add_argument("--doc-len", type=int, default=None, help="Tokens per packed document; one document if unset.")
    parser.add_argument("--num-distinct", type=int, default=8)
    parser.add_argument("--steady-reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    # One throwaway S_Q per Triton integer-specialization class compiles every config up front, so the
    # sweep below times re-tuning alone. Classes: S_Q % 16 == 0; S_Q % 8 == 0 only (S_K % 16 == 0); neither.
    jit_warm = [32768 - 16 * 7, 32768 - 8 * 7, 32768 - 7]
    distinct = sorted({rng.randint(28000, 32767) for _ in range(args.num_distinct)}, reverse=True)
    sweep = [32768] + distinct + [distinct[0]] * 2

    print(f"jit warm-up over {jit_warm}", flush=True)
    jit_rows = retune_sweep(jit_warm, args.rank, args.doc_len, 1)
    print(f"retune sweep over {sweep}", flush=True)
    sweep_rows = retune_sweep(sweep, args.rank, args.doc_len, args.steady_reps)
    identity = [topk_identity(s_q, args.rank, args.doc_len) for s_q in (32768, distinct[0])]
    print(json.dumps(identity, indent=1), flush=True)

    retunes = [row["retune_s"] for row in sweep_rows if row["retuned"]]
    summary = {
        "device": torch.cuda.get_device_name(),
        "triton": triton.__version__,
        "rank": args.rank,
        "doc_len": args.doc_len,
        "median_retune_s": statistics.median(retunes) if retunes else None,
        "median_steady_call_s": statistics.median(row["steady_call_s"] for row in sweep_rows),
        "num_retunes_in_sweep": len(retunes),
    }
    print(json.dumps(summary, indent=1), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "jit_warm": jit_rows, "sweep": sweep_rows, "topk_identity": identity}, indent=1)
    )


if __name__ == "__main__":
    main()
