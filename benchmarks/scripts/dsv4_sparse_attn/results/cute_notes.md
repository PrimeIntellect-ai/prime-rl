# CuTe DSL forwards: results notes

These results compare four arms at kernel commit `8af8d58fb`:
- `tilelang`: `main`'s TileLang forward and backward.
- `cute`: a one-to-one CuTe DSL port of that forward, with the TileLang backward.
- `cute_ws`: the warp-specialized, persistent CuTe DSL forward, with the TileLang backward.
- `flashmla_fwd_ref`: FlashMLA's sparse prefill forward, a reference rather than a backend.

They were measured on 2026-10-10 on `prime-nebius-puku-h200-gpu-059` (H200, SLURM job 3569, GPU 0) during a quiet
window: no other process used any GPU. The run's git state was dirty only because of `summarize.py`'s new
per-arm table; no kernel or harness timing code differed from the commit.

Caveat: the corpus is synthetic. Its CSA picks come from a random-weight indexer and are near-uniform over the
readable entries. A trained indexer favors recent and neighboring entries, so CSA gather locality here is
pessimistic for every arm.

## Files

- `cute.json`: `bench.py --backends tilelang cute cute_ws flashmla_fwd_ref --label cute`, over the 240 grid items
  (7 ABBA rounds x 10 calls).
- `cute_tables.md`: `bench.py --compare cute.json`, with every item's times, FLOPs and peak memory for every arm.
- `cute_stream.json`, `cute_stream_tables.md`: `stream.py --backends tilelang cute cute_ws --label cute` and its
  tables.
- `cute_summary.md`: `summarize.py cute.json cute_stream.json`, including the per-arm table of the single-row
  items (absolute times, TFLOP/s on valid slots, executed over useful FLOPs).
- `cute_vs_main.md`: `bench.py --compare main.json cute.json`, the TileLang regression check against `main.json`.

## Findings

Geometric means over the 240 items of the ratio between two arms' times. Lower is better for the first arm.

| ratio | time | csa | hca | sliding | all (min-max) |
|---|---|---|---|---|---|
| cute / tilelang, fwd | GPU | 0.94 | 0.94 | 0.93 | 0.94 (0.87-0.97) |
| cute_ws / cute, fwd | GPU | 0.52 | 0.48 | 0.42 | 0.47 (0.38-0.64) |
| cute_ws / tilelang, fwd | GPU | 0.49 | 0.45 | 0.39 | 0.44 (0.34-0.61) |
| cute_ws / flashmla_fwd_ref, fwd | GPU | 0.87 | 0.80 | 0.75 | 0.81 (0.61-0.93) |
| cute_ws / tilelang, fwd | op boundary | 0.25 | 0.21 | 0.19 | 0.21 (0.10-0.51) |
| cute_ws / flashmla_fwd_ref, fwd | op boundary | 0.70 | 0.63 | 0.58 | 0.63 (0.38-0.95) |
| cute_ws / tilelang, fwd+bwd | GPU | 0.87 | 0.83 | 0.82 | 0.84 (0.79-0.91) |
| cute_ws / tilelang, fwd+bwd | op boundary | 0.72 | 0.68 | 0.67 | 0.69 (0.57-0.88) |

- The `cute_ws` forward's GPU time is below FlashMLA's on all 240 items.
- Largest row: on single-65536-csa-cp1 the forward takes 9.77 ms of GPU time with `cute_ws`, 11.08 ms with
  FlashMLA, 20.14 ms with `cute` and 21.73 ms with TileLang.
- Forward+backward on the 64k-token CSA rows reaches 213-226 TFLOP/s with `cute_ws` (183-199 with TileLang). That is
  21.5-23% of the 989.5 TFLOP/s dense BF16 peak, counting useful FLOPs of valid slots over op-boundary time. The
  TileLang backward is now most of the time.
- Executed over useful forward FLOPs is higher for `cute_ws` than for TileLang (for example 1.18 vs 1.09 on
  single-2048-csa-cp1): it reads whole 128-slot pairs where TileLang reads 64-slot tiles. Its GPU time is still
  lower.
- Host overhead per forward call (op-boundary minus GPU busy time, median): TileLang 697 µs, `cute` 123 µs,
  `cute_ws` 53 µs, FlashMLA reference 147 µs.
- Correctness: every arm passes every item. The tightest margin is `cute_ws`'s LSE against TileLang: 4.2e-7 against
  a 5e-7 bound.
- TileLang regression check: TileLang's times in this run over those in `main.json` have a geometric mean of 1.001
  (forward GPU time) and 1.000 (forward+backward GPU time), and per item fall within 0.90-1.10 on every time
  measure. No regression.
- Dynamic stream: every arm compiles on the first item only, with no compiles per shape afterwards.
  - The other 31 items take 0.617 s with TileLang, 0.583 s with `cute` and 0.510 s with `cute_ws`.
  - A warm process still recompiles the CuTe forward once (1.4-3.2 s first item), because `cute.compile` keeps no
    disk cache. TileLang instead loads its kernels from disk.
- Forward kernel source lines: TileLang 289, `cute` 372, `cute_ws` 643.

The ratio table and the per-row summaries above came from `~/tmp/dsv4-sparse-attn-bench/planB/m3_ratios.py
cute.json main.json` (Plan B's scratch directory).
