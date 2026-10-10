# DSv4 sparse attention: cross-backend comparison (`compare`)

## Answer

- **Fastest forward: `cute_ws`** (warp-specialized persistent CuTe DSL forward). It is the fastest arm on all 240
  items, on GPU time and on op-boundary time. Its forward GPU time is 0.44 of TileLang's and 0.81 of FlashMLA's
  (geometric means). FlashMLA here means `flashmla_fwd_ref`, FlashMLA's sparse prefill forward called directly.
- **Fastest measured forward+backward: `cudnn_flashmla`** (FlashMLA fwd + cuDNN bwd). Its GPU time is 0.70 of
  TileLang's, against 0.84 for `cute_ws` (CuTe fwd + TileLang bwd). It has the lowest GPU time on 225 of 240
  items. The 15 exceptions are 2048- and 4096-token HCA and sliding cp8 slices, where `cute_ws` wins. On a 64k
  CSA item it takes 53 ms against TileLang's 95 ms. The cuDNN backward does the work: alone it takes 0.75 of
  TileLang's backward GPU time.
- **CuTe fwd + cuDNN bwd (an estimate, not measured):** about 0.66 of TileLang's forward+backward GPU time, and
  0.94 of `cudnn_flashmla`'s. It would be the fastest pairing, but on large items it would beat
  `cudnn_flashmla` by only 1-5%. The backward dominates there, and the two forwards differ by about 6%. The gain
  is larger on small items, where `cute_ws`'s 54 µs host overhead per forward replaces FlashMLA's 198 µs.
- Every arm passes the correctness gate on every item. TileLang shows no regression on any branch: its time over
  `main.json`'s has a geometric mean between 0.999 and 1.017.

Measured 2026-10-10 on `prime-nebius-puku-h200-gpu-059` (H200, job 3569, GPU 0). The node was quiet: no other
process ran on any GPU. Clean tree at `dfb6b9bd3`, a throwaway merge of `feat/ds-v4-cudnn-flashmla-attn` and
`feat/ds-v4-cute-sparse-attn-fwd`. The harness defaults applied: 240 grid items, 7 ABBA rounds of 10 calls, and
L2 flushed with the GPU idle before each call. All five arms ran in one process.

**Synthetic-corpus caveat.** The CSA picks come from a random-weight Lightning Indexer, so they are near-uniform
over the readable entries. A trained indexer favors recent and neighboring entries. Gather locality on CSA items
is therefore pessimistic here, for every arm. Sliding and HCA indices do not depend on weights.

Arms:

| arm | forward | backward |
|---|---|---|
| `tilelang` | TileLang | TileLang |
| `cudnn_flashmla` | FlashMLA sparse prefill | cuDNN frontend SM90 DSA, plus TileLang delta kernel |
| `cute` | CuTe DSL port of the TileLang design (M1) | TileLang |
| `cute_ws` | CuTe DSL, warp-specialized and persistent | TileLang |
| `flashmla_fwd_ref` | FlashMLA sparse prefill, called directly | none (forward-only reference) |

Definitions. *GPU time* is the summed duration of the kernels, memsets and memcpys that one call launches, from a
torch.profiler trace. *Op-boundary time* is CUDA events around one call that starts with the GPU idle, so it
includes host launch overhead that the GPU does not hide. Every table below uses the median over rounds. Rows
are keyed by document shape (`single`, `heavy`, `tiny`), layer type, total tokens in the sequence, and context
parallelism (`cp8 r7` is rank 7's slice of an 8-way split, the most loaded rank under causal masking).

## Representative items

Forward GPU time per call, ms (lower is better):

| shape | layer | tokens | cp | tilelang | cudnn_flashmla | cute | cute_ws | flashmla_fwd_ref |
|---|---|---|---|---|---|---|---|---|
| single | csa | 65536 | cp1 | 21.69 | 10.39 | 20.30 | 9.77 | 11.43 |
| single | hca | 65536 | cp1 | 16.23 | 7.99 | 15.39 | 7.11 | 8.37 |
| single | sliding | 65536 | cp1 | 9.08 | 4.15 | 8.35 | 3.16 | 4.08 |
| heavy | csa | 65536 | cp1 | 17.34 | 8.44 | 16.20 | 7.72 | 8.79 |
| single | csa | 65536 | cp8 r7 | 2.76 | 1.35 | 2.59 | 1.24 | 1.34 |
| single | hca | 65536 | cp8 r7 | 2.75 | 1.32 | 2.59 | 1.21 | 1.31 |
| tiny | csa | 4096 | cp1 | 0.68 | 0.38 | 0.64 | 0.33 | 0.37 |
| tiny | hca | 4096 | cp1 | 0.53 | 0.28 | 0.49 | 0.22 | 0.27 |
| tiny | sliding | 4096 | cp1 | 0.53 | 0.28 | 0.49 | 0.23 | 0.27 |

Forward op-boundary time per call, ms (lower is better):

| shape | layer | tokens | cp | tilelang | cudnn_flashmla | cute | cute_ws | flashmla_fwd_ref |
|---|---|---|---|---|---|---|---|---|
| single | csa | 65536 | cp1 | 21.72 | 11.46 | 20.27 | 10.95 | 11.47 |
| single | hca | 65536 | cp1 | 16.55 | 8.30 | 15.30 | 7.65 | 8.43 |
| single | sliding | 65536 | cp1 | 9.71 | 4.23 | 8.42 | 3.22 | 4.24 |
| heavy | csa | 65536 | cp1 | 17.46 | 8.81 | 16.18 | 8.37 | 9.04 |
| single | csa | 65536 | cp8 r7 | 3.43 | 1.51 | 2.70 | 1.31 | 1.43 |
| single | hca | 65536 | cp8 r7 | 3.41 | 1.47 | 2.70 | 1.27 | 1.39 |
| tiny | csa | 4096 | cp1 | 1.35 | 0.55 | 0.75 | 0.38 | 0.49 |
| tiny | hca | 4096 | cp1 | 1.23 | 0.48 | 0.61 | 0.28 | 0.42 |
| tiny | sliding | 4096 | cp1 | 1.24 | 0.48 | 0.61 | 0.29 | 0.41 |

Forward+backward time per call, ms (lower is better). The `est` column is the CuTe fwd + cuDNN bwd *estimate*
(defined in its own section below), not a measurement.

| shape | layer | tokens | cp | time | tilelang | cudnn_flashmla | cute | cute_ws | est |
|---|---|---|---|---|---|---|---|---|---|
| single | csa | 65536 | cp1 | GPU | 94.65 | 53.02 | 94.45 | 84.19 | 52.40 |
| single | hca | 65536 | cp1 | GPU | 63.72 | 37.44 | 63.29 | 54.90 | 36.57 |
| single | sliding | 65536 | cp1 | GPU | 29.33 | 19.99 | 28.65 | 23.50 | 19.00 |
| heavy | csa | 65536 | cp1 | GPU | 69.03 | 40.24 | 68.63 | 59.95 | 39.52 |
| single | csa | 65536 | cp8 r7 | GPU | 13.22 | 6.82 | 12.79 | 11.60 | 6.72 |
| single | hca | 65536 | cp8 r7 | GPU | 11.84 | 6.25 | 11.71 | 10.35 | 6.14 |
| tiny | csa | 4096 | cp1 | GPU | 2.07 | 1.39 | 2.03 | 1.72 | 1.34 |
| tiny | hca | 4096 | cp1 | GPU | 1.43 | 1.09 | 1.40 | 1.15 | 1.03 |
| tiny | sliding | 4096 | cp1 | GPU | 1.43 | 1.09 | 1.39 | 1.15 | 1.04 |
| single | csa | 65536 | cp1 | op | 95.51 | 53.34 | 94.01 | 84.00 | 52.82 |
| single | hca | 65536 | cp1 | op | 64.49 | 37.42 | 63.17 | 55.06 | 36.78 |
| single | sliding | 65536 | cp1 | op | 30.15 | 20.29 | 28.89 | 23.64 | 19.28 |
| heavy | csa | 65536 | cp1 | op | 69.81 | 40.16 | 68.45 | 60.10 | 39.72 |
| single | csa | 65536 | cp8 r7 | op | 14.17 | 7.21 | 13.40 | 11.91 | 7.01 |
| single | hca | 65536 | cp8 r7 | op | 12.79 | 6.57 | 12.01 | 10.54 | 6.37 |
| tiny | csa | 4096 | cp1 | op | 2.96 | 1.98 | 2.30 | 2.16 | 1.80 |
| tiny | hca | 4096 | cp1 | op | 2.44 | 1.77 | 1.81 | 1.65 | 1.57 |
| tiny | sliding | 4096 | cp1 | op | 2.47 | 1.78 | 1.81 | 1.68 | 1.59 |

On single-65536-csa-cp1, forward+backward reaches 356 TFLOP/s for `cudnn_flashmla`, 226 for `cute_ws` and 199
for `tilelang`. This counts useful FLOPs (valid slots only) over op-boundary time. Higher is better, and the
dense BF16 peak is 989.5.

## Ratios against tilelang over all 240 items

Ratio = the arm's time / tilelang's time on the same item. Each cell is the geometric mean over the items of
that layer type, and the last column is over all items with the min and max in brackets. Lower is better, and
below 1 means faster than tilelang.

| arm | mode | time | csa | hca | sliding | all [min-max] |
|---|---|---|---|---|---|---|
| cudnn_flashmla | fwd | GPU | 0.58 | 0.60 | 0.57 | 0.58 [0.45-0.91] |
| cudnn_flashmla | fwd | op | 0.42 | 0.41 | 0.40 | 0.41 [0.35-0.53] |
| cudnn_flashmla | fwd+bwd | GPU | 0.64 | 0.73 | 0.74 | 0.70 [0.52-0.92] |
| cudnn_flashmla | fwd+bwd | op | 0.61 | 0.65 | 0.68 | 0.65 [0.51-0.77] |
| cute | fwd | GPU | 0.94 | 0.94 | 0.93 | 0.94 [0.90-0.97] |
| cute | fwd | op | 0.50 | 0.47 | 0.44 | 0.47 [0.22-0.93] |
| cute | fwd+bwd | GPU | 0.99 | 0.98 | 0.98 | 0.98 [0.97-1.00] |
| cute | fwd+bwd | op | 0.81 | 0.78 | 0.77 | 0.79 [0.67-0.99] |
| cute_ws | fwd | GPU | 0.49 | 0.45 | 0.39 | 0.44 [0.34-0.61] |
| cute_ws | fwd | op | 0.25 | 0.21 | 0.19 | 0.21 [0.10-0.50] |
| cute_ws | fwd+bwd | GPU | 0.87 | 0.84 | 0.82 | 0.84 [0.79-0.90] |
| cute_ws | fwd+bwd | op | 0.72 | 0.68 | 0.67 | 0.69 [0.58-0.88] |
| flashmla_fwd_ref | fwd | GPU | 0.56 | 0.57 | 0.53 | 0.55 [0.45-0.75] |
| flashmla_fwd_ref | fwd | op | 0.35 | 0.33 | 0.32 | 0.33 [0.23-0.53] |
| est (CuTe fwd + cuDNN bwd) | fwd+bwd | GPU | 0.61 | 0.68 | 0.69 | 0.66 [0.51-0.79] |
| est (CuTe fwd + cuDNN bwd) | fwd+bwd | op | 0.56 | 0.58 | 0.60 | 0.58 [0.49-0.74] |

Other ratios, as geometric means over all 240 items (lower is better):

- Forward GPU time over `flashmla_fwd_ref`'s: `cute_ws` 0.81 (0.60-0.93, faster on 240 of 240),
  `cudnn_flashmla` 1.06, `cute` 1.71. `cudnn_flashmla` calls the same FlashMLA kernel. Its extra 6% comes from the
  index flattening and the conversion to the public LSE (19 launches per forward against 13).
- Backward GPU time alone, defined as fwd+bwd GPU time minus fwd GPU time, over tilelang's: `cudnn_flashmla`
  0.75 (0.52-0.95). `cute` and `cute_ws` give 1.00, as expected, since they reuse the TileLang backward.
- Arm with the lowest fwd+bwd *op-boundary* time: `cudnn_flashmla` on 144 items, `cute_ws` on 96. `cute_ws`
  wins 70 of the 72 cp8 slices of 2048 and 4096 tokens, plus 26 HCA and sliding items at other lengths. Those
  calls are host-bound, and `cute_ws` launches 21-22 kernels per fwd+bwd call against `cudnn_flashmla`'s 44-45.

## Estimate: CuTe fwd + cuDNN bwd (not measured)

This pairing was not run. Per item, the estimate is

```
est = T(cudnn_flashmla, fwd+bwd) - T(cudnn_flashmla, fwd) + T(cute_ws, fwd)
```

Each `T` is a measured median, and the formula is applied separately to GPU time and to op-boundary time. Why it
is plausible:

- `_cudnn_flashmla_backward` takes only (q, kv, out, dO, indices, public LSE). It rebuilds its own flattened
  indices and converts the public base-2 LSE itself, and `cute_ws` returns that same public LSE (it passes the
  5e-7 LSE gate against TileLang). Switching would amount to pointing `BACKWARD_BACKENDS["cute_ws"]` at
  `_cudnn_flashmla_backward`. The cuDNN shape gate (kv_group 1, 64 or 128 heads, dim 512, SM90) would then also
  apply.
- The cuDNN backward's work does not depend on which forward produced `out` and the LSE.

What it ignores: L2 contents and launch overlap at the fwd/bwd boundary, and the fact that the op-boundary
subtraction treats host overhead as additive. It also ignores correctness: cuDNN's gradients were gated only on
FlashMLA's outputs. Results:

- GPU: 0.66 of tilelang (0.51-0.79), 0.94 of `cudnn_flashmla` (0.85-0.99), and 0.79 of `cute_ws` (0.58-0.96).
- Op-boundary: 0.58 of tilelang, 0.90 of `cudnn_flashmla` (0.83-0.99), and 0.84 of `cute_ws` (0.59-1.02).
- Compiles: the cuDNN per-bucket backward compiles (below) plus one `cute.compile` of about 2.9 s per process.

## Host overhead per call

Op-boundary minus GPU time, µs (lower is better). For fwd+bwd, the backward's launches partly overlap the
forward's kernels, so this is the exposed host time. Negative minimums are timer disagreement between CUDA
events and the profiler on short GPU-bound calls.

| arm | fwd median | fwd max | fwd+bwd median | fwd+bwd max | items where fwd host > fwd GPU |
|---|---|---|---|---|---|
| tilelang | 700 | 807 | 1021 | 2041 | 127 of 240 |
| cudnn_flashmla | 198 | 1072 | 681 | 1356 | 96 of 240 |
| cute | 121 | 212 | 418 | 1375 | 70 of 240 |
| cute_ws | 54 | 1172 | 539 | 1218 | 58 of 240 |
| flashmla_fwd_ref | 143 | 452 | - | - | 79 of 240 |

## Compiles over the dynamic stream

The stream has 32 items of varying lengths, layer types and cp slices. Each arm replays it in a fresh process
with fresh caches (cold), then again in a second process that reuses those caches (warm). Times are in seconds,
and lower is better. `rest` sums the first calls of items 2 to 32 and excludes per-process costs. Every process
also pays about 41 s to import `prime_rl.trainer.models` (not in the table).

| arm | phase | compiles | first item s | rest s | where the compiles happen |
|---|---|---|---|---|---|
| tilelang | cold | 4 | 20.73 | 0.615 | item 0: 4 TileLang kernels |
| tilelang | warm | 0 | 0.28 | 0.616 | none (4 disk loads) |
| cudnn_flashmla | cold | 7 | 6.98 | 7.968 | item 0: 4, then items 1, 2 and 8: 1 each |
| cudnn_flashmla | warm | 6 | 3.45 | 7.940 | same items, and only the TileLang delta loads from disk |
| cute | cold | 4 | 14.03 | 0.586 | item 0: 1 CuTe + 3 TileLang bwd |
| cute | warm | 1 | 1.44 | 0.583 | item 0: the CuTe forward again (no disk cache) |
| cute_ws | cold | 4 | 15.72 | 0.512 | item 0: 1 CuTe + 3 TileLang bwd |
| cute_ws | warm | 1 | 3.16 | 0.512 | item 0: the CuTe forward again, 2.9 s |

`cudnn_flashmla` compiles one cuDNN main backward kernel per slot-width bucket the stream reaches: 640, 128, 256
and 512 slots, at items 0, 1, 2 and 8. Each takes about 2.5 s, and they recur in the warm process because
`cute.compile` keeps no file cache. That is why its `rest` is 7.9 s against about 0.5-0.6 s for the TileLang
backward arms. In a training run, each process pays those compiles once per bucket. The CuTe forward arms
recompile once per process, also because `cute.compile` keeps no file cache.

## Correctness margins (tightest)

Every arm passed every check on all 240 items, and none raised. Each margin is the largest relative error over
the bound (1 would fail).

| arm | tightest check | error / bound | margin used | item |
|---|---|---|---|---|
| tilelang | dsink vs fp32 dense | 9.56e-3 / 1e-2 | 0.96 | single-2048-hca-cp8r4 |
| cudnn_flashmla | dsink vs tilelang | 9.52e-3 / 1e-2 | 0.95 | single-65536-sliding-cp8r4 |
| cute | dsink vs fp32 dense | 9.56e-3 / 1e-2 | 0.96 | single-2048-hca-cp8r4 |
| cute_ws | dsink vs fp32 dense | 9.04e-3 / 1e-2 | 0.90 | single-2048-hca-cp8r4 |
| cute_ws | lse vs tilelang | 4.17e-7 / 5e-7 | 0.83 | heavy-65536-sliding-cp1 |
| flashmla_fwd_ref | out vs tilelang | 5.15e-3 / 1e-2 | 0.52 | short-2048-csa-cp8r7 |

The dsink margin is thin for every arm with a backward, TileLang included, so it reflects bf16 kernels against
fp32 math rather than any one backend. `cudnn_flashmla`'s dsink against the fp32 dense reference is 6.67e-3.

## Peak memory

Peak allocation above the pre-call level, in MiB (lower is better). The arms differ by at most 8%.

| shape | layer | tokens | cp | mode | tilelang | cudnn_flashmla | cute | cute_ws | flashmla_fwd_ref |
|---|---|---|---|---|---|---|---|---|---|
| single | csa | 65536 | cp1 | fwd | 4112 | 4324 | 4112 | 4112 | 4448 |
| single | csa | 65536 | cp1 | fwd+bwd | 8480 | 8672 | 8480 | 8480 | - |
| single | csa | 65536 | cp8 r7 | fwd+bwd | 1270 | 1294 | 1270 | 1270 | - |
| tiny | csa | 4096 | cp1 | fwd+bwd | 530 | 542 | 530 | 530 | - |

Over all items, the peak relative to tilelang is: `cudnn_flashmla` 1.01-1.05 (fwd) and 1.00-1.02 (fwd+bwd),
from padding to the slot-width bucket and the flattened indices. `cute` and `cute_ws` give 0.98-1.00, and
`flashmla_fwd_ref` 1.02-1.08.

## Regression check: tilelang column vs `main.json`

Ratio = this file's tilelang time / `main.json`'s tilelang time on the same item. The table gives the geometric
mean, with the range in brackets, over 240 items. Ideally it is 1, and above 1 means slower than main.

| result file | tip measured | fwd GPU | fwd op | fwd+bwd GPU | fwd+bwd op |
|---|---|---|---|---|---|
| `cudnn_flashmla.json` | `e28cc525e` | 1.000 [0.98-1.04] | 1.007 [0.93-1.08] | 1.001 [0.99-1.04] | 1.015 [0.98-1.08] |
| `cute.json` | `8af8d58fb` | 1.001 [0.99-1.08] | 1.003 [0.90-1.09] | 1.000 [0.99-1.01] | 1.012 [0.97-1.10] |
| `compare.json` | `dfb6b9bd3` | 1.001 [0.99-1.04] | 1.005 [0.94-1.11] | 0.999 [0.99-1.01] | 1.017 [0.97-1.07] |

GPU time is unchanged. The 0.5-1.7% on op-boundary time sits inside the host jitter that `main_notes.md`
measured on small items, up to about 9% for TileLang. The other arms also reproduce their own branch runs: GPU
time geometric means against `cudnn_flashmla.json` and `cute.json` fall within 0.998-1.002 for every arm.

## Files

- `compare.json`: `bench.py --backends tilelang cudnn_flashmla cute cute_ws flashmla_fwd_ref --label compare`.
- `compare_tables.md`: `bench.py --compare compare.json`, with every item and arm.
- `compare_stream.json`, `compare_stream_tables.md`: `stream.py --backends tilelang cudnn_flashmla cute cute_ws
  --label compare` and its tables.
- `compare_vs_main.md`: `bench.py --compare main.json compare.json`.
- The rest of this file is `summarize.py compare.json compare_stream.json`, unedited apart from its title level.

## summarize.py output: DSv4 sparse attention baseline: `compare`

Caveat: the corpus is synthetic. Its CSA picks come from a random-weight Lightning Indexer and are
near-uniform over the readable entries, while a trained indexer favors recent and neighboring entries,
so CSA gather locality here is pessimistic. Sliding and HCA indices do not depend on weights.

- GPU: NVIDIA H200, driver 580.173.02, power limit 700.00 W,
  max SM clock 1980 MHz, host `prime-nebius-puku-h200-gpu-059`.
- Code: git `dfb6b9bd3`, torch 2.13.0+cu130, tilelang 0.1.12.
- Corpus hash `b5b289171983c8c6`, 240 grid items.
- Settings: 7 ABBA rounds x 10 calls, 3 warmup calls, L2 flushed and GPU idle before each call.

## Single-row items (cp1)

Op-boundary time per call in ms (lower is better). `gpu` is the GPU busy time of the same call; the
difference is host overhead. TFLOP/s counts useful FLOPs over op-boundary time (higher is better);
`% peak` is against 989.5 dense BF16 TFLOP/s. `ref/TL` is the FlashMLA
forward reference's op-boundary time over tilelang's forward (below 1 means the reference is faster).

| item | fwd | fwd gpu | f+b | f+b gpu | f+b TFLOP/s | % peak | ref/TL fwd |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | 1.21 | 0.54 | 3.27 | 2.16 | 109 | 11.0 | 0.32 |
| single-2048-hca-cp1 | 1.10 | 0.36 | 2.57 | 1.17 | 48 | 4.9 | 0.31 |
| single-2048-sliding-cp1 | 1.03 | 0.31 | 2.39 | 1.03 | 49 | 4.9 | 0.29 |
| short-2048-csa-cp1 | 1.13 | 0.44 | 2.87 | 1.63 | 81 | 8.2 | 0.33 |
| short-2048-hca-cp1 | 1.07 | 0.35 | 2.51 | 1.14 | 46 | 4.7 | 0.32 |
| short-2048-sliding-cp1 | 1.00 | 0.31 | 2.36 | 1.01 | 48 | 4.8 | 0.29 |
| heavy-2048-csa-cp1 | 1.15 | 0.47 | 3.02 | 1.82 | 90 | 9.1 | 0.34 |
| heavy-2048-hca-cp1 | 1.08 | 0.34 | 2.50 | 1.11 | 46 | 4.6 | 0.31 |
| heavy-2048-sliding-cp1 | 1.01 | 0.30 | 2.35 | 0.99 | 46 | 4.7 | 0.29 |
| tiny-2048-csa-cp1 | 1.06 | 0.36 | 2.42 | 1.10 | 22 | 2.2 | 0.32 |
| tiny-2048-hca-cp1 | 0.96 | 0.27 | 2.15 | 0.77 | 20 | 2.0 | 0.30 |
| tiny-2048-sliding-cp1 | 0.96 | 0.27 | 2.21 | 0.77 | 20 | 2.0 | 0.31 |
| single-4096-csa-cp1 | 1.90 | 1.21 | 6.01 | 5.15 | 159 | 16.1 | 0.38 |
| single-4096-hca-cp1 | 1.43 | 0.69 | 3.19 | 2.23 | 83 | 8.4 | 0.35 |
| single-4096-sliding-cp1 | 1.34 | 0.59 | 2.94 | 1.95 | 81 | 8.1 | 0.32 |
| short-4096-csa-cp1 | 1.51 | 0.83 | 3.90 | 3.05 | 115 | 11.7 | 0.37 |
| short-4096-hca-cp1 | 1.40 | 0.67 | 3.09 | 2.13 | 74 | 7.5 | 0.36 |
| short-4096-sliding-cp1 | 1.29 | 0.59 | 2.87 | 1.90 | 77 | 7.8 | 0.32 |
| heavy-4096-csa-cp1 | 1.47 | 0.77 | 3.60 | 2.72 | 99 | 10.0 | 0.36 |
| heavy-4096-hca-cp1 | 1.42 | 0.65 | 3.03 | 2.03 | 68 | 6.9 | 0.34 |
| heavy-4096-sliding-cp1 | 1.28 | 0.58 | 2.81 | 1.84 | 72 | 7.3 | 0.33 |
| tiny-4096-csa-cp1 | 1.35 | 0.68 | 2.96 | 2.07 | 35 | 3.6 | 0.36 |
| tiny-4096-hca-cp1 | 1.23 | 0.53 | 2.44 | 1.43 | 34 | 3.5 | 0.34 |
| tiny-4096-sliding-cp1 | 1.24 | 0.53 | 2.47 | 1.43 | 34 | 3.5 | 0.34 |
| single-16384-csa-cp1 | 5.90 | 5.52 | 23.71 | 22.67 | 193 | 19.5 | 0.47 |
| single-16384-hca-cp1 | 3.54 | 2.85 | 10.88 | 9.95 | 132 | 13.3 | 0.42 |
| single-16384-sliding-cp1 | 2.97 | 2.27 | 8.32 | 7.44 | 115 | 11.6 | 0.39 |
| short-16384-csa-cp1 | 4.50 | 3.92 | 15.99 | 15.01 | 163 | 16.5 | 0.44 |
| short-16384-hca-cp1 | 3.32 | 2.59 | 9.14 | 8.24 | 105 | 10.7 | 0.44 |
| short-16384-sliding-cp1 | 2.96 | 2.26 | 8.12 | 7.29 | 112 | 11.4 | 0.39 |
| heavy-16384-csa-cp1 | 3.97 | 3.36 | 12.92 | 12.03 | 139 | 14.1 | 0.44 |
| heavy-16384-hca-cp1 | 3.21 | 2.49 | 8.61 | 7.76 | 98 | 9.9 | 0.43 |
| heavy-16384-sliding-cp1 | 2.96 | 2.26 | 7.86 | 7.01 | 104 | 10.5 | 0.40 |
| tiny-16384-csa-cp1 | 3.30 | 2.64 | 8.78 | 7.89 | 45 | 4.5 | 0.45 |
| tiny-16384-hca-cp1 | 2.72 | 2.03 | 6.18 | 5.35 | 51 | 5.2 | 0.43 |
| tiny-16384-sliding-cp1 | 2.73 | 2.03 | 6.20 | 5.35 | 51 | 5.1 | 0.43 |
| single-49208-csa-cp1 | 16.49 | 16.42 | 70.85 | 70.06 | 200 | 20.3 | 0.52 |
| single-49208-hca-cp1 | 11.54 | 11.15 | 42.64 | 41.83 | 169 | 17.1 | 0.49 |
| single-49208-sliding-cp1 | 7.46 | 6.90 | 22.93 | 22.09 | 126 | 12.7 | 0.42 |
| short-49208-csa-cp1 | 13.13 | 12.94 | 51.17 | 50.33 | 181 | 18.3 | 0.51 |
| short-49208-hca-cp1 | 8.53 | 7.93 | 26.19 | 25.37 | 120 | 12.1 | 0.48 |
| short-49208-sliding-cp1 | 7.49 | 6.95 | 22.66 | 21.79 | 123 | 12.4 | 0.42 |
| heavy-49208-csa-cp1 | 14.94 | 14.90 | 61.75 | 60.94 | 194 | 19.6 | 0.52 |
| heavy-49208-hca-cp1 | 9.10 | 8.53 | 29.68 | 28.87 | 137 | 13.9 | 0.48 |
| heavy-49208-sliding-cp1 | 7.46 | 6.78 | 22.62 | 21.74 | 123 | 12.4 | 0.42 |
| tiny-49208-csa-cp1 | 8.38 | 7.83 | 24.16 | 23.44 | 50 | 5.0 | 0.51 |
| tiny-49208-hca-cp1 | 6.71 | 6.04 | 16.76 | 15.89 | 58 | 5.8 | 0.47 |
| tiny-49208-sliding-cp1 | 6.72 | 6.04 | 16.81 | 15.90 | 58 | 5.8 | 0.47 |
| single-65536-csa-cp1 | 21.72 | 21.69 | 95.51 | 94.65 | 199 | 20.1 | 0.53 |
| single-65536-hca-cp1 | 16.55 | 16.23 | 64.49 | 63.72 | 179 | 18.1 | 0.51 |
| single-65536-sliding-cp1 | 9.71 | 9.08 | 30.15 | 29.33 | 127 | 12.9 | 0.44 |
| short-65536-csa-cp1 | 16.35 | 16.33 | 63.22 | 62.49 | 177 | 17.9 | 0.51 |
| short-65536-hca-cp1 | 10.95 | 10.39 | 33.59 | 32.79 | 117 | 11.8 | 0.49 |
| short-65536-sliding-cp1 | 9.66 | 9.06 | 29.68 | 28.83 | 124 | 12.5 | 0.43 |
| heavy-65536-csa-cp1 | 17.46 | 17.34 | 69.81 | 69.03 | 183 | 18.5 | 0.52 |
| heavy-65536-hca-cp1 | 11.77 | 11.26 | 38.34 | 37.64 | 134 | 13.6 | 0.49 |
| heavy-65536-sliding-cp1 | 9.62 | 8.99 | 29.27 | 28.48 | 121 | 12.2 | 0.43 |
| tiny-65536-csa-cp1 | 10.92 | 10.43 | 31.83 | 31.18 | 51 | 5.1 | 0.52 |
| tiny-65536-hca-cp1 | 8.69 | 8.04 | 22.03 | 21.20 | 59 | 6.0 | 0.48 |
| tiny-65536-sliding-cp1 | 8.69 | 8.03 | 22.01 | 21.20 | 59 | 6.0 | 0.48 |

## Every arm on the single-row items (cp1)

Time per call in ms (lower is better): op-boundary, then GPU busy time. TFLOP/s counts useful FLOPs (valid
slots only) over op-boundary time (higher is better). `exec/useful` is the FLOPs of the slots the arm's
tiles touch over useful FLOPs (1 is no wasted work). `-` marks a mode the arm does not have.

| item | arm | fwd | fwd gpu | fwd TFLOP/s | exec/useful fwd | f+b | f+b gpu | f+b TFLOP/s |
|---|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang | 1.21 | 0.54 | 84 | 1.09 | 3.27 | 2.16 | 109 |
| single-2048-csa-cp1 | cudnn_flashmla | 0.44 | 0.29 | 233 | 1.09 | 1.94 | 1.26 | 184 |
| single-2048-csa-cp1 | cute | 0.61 | 0.51 | 168 | 1.09 | 2.63 | 2.13 | 136 |
| single-2048-csa-cp1 | cute_ws | 0.30 | 0.25 | 335 | 1.18 | 2.49 | 1.87 | 144 |
| single-2048-csa-cp1 | flashmla_fwd_ref | 0.38 | 0.28 | 267 | 1.69 | - | - | - |
| single-2048-hca-cp1 | tilelang | 1.10 | 0.36 | 32 | 1.41 | 2.57 | 1.17 | 48 |
| single-2048-hca-cp1 | cudnn_flashmla | 0.40 | 0.21 | 87 | 1.41 | 1.59 | 0.81 | 78 |
| single-2048-hca-cp1 | cute | 0.48 | 0.34 | 73 | 1.41 | 1.92 | 1.14 | 64 |
| single-2048-hca-cp1 | cute_ws | 0.22 | 0.17 | 159 | 1.89 | 1.73 | 0.98 | 71 |
| single-2048-hca-cp1 | flashmla_fwd_ref | 0.34 | 0.20 | 103 | 1.95 | - | - | - |
| single-2048-sliding-cp1 | tilelang | 1.03 | 0.31 | 32 | 1.02 | 2.39 | 1.03 | 49 |
| single-2048-sliding-cp1 | cudnn_flashmla | 0.36 | 0.16 | 92 | 1.02 | 1.53 | 0.71 | 76 |
| single-2048-sliding-cp1 | cute | 0.41 | 0.29 | 81 | 1.02 | 1.75 | 1.01 | 66 |
| single-2048-sliding-cp1 | cute_ws | 0.17 | 0.12 | 194 | 1.03 | 1.60 | 0.84 | 73 |
| single-2048-sliding-cp1 | flashmla_fwd_ref | 0.29 | 0.15 | 113 | 1.03 | - | - | - |
| short-2048-csa-cp1 | tilelang | 1.13 | 0.44 | 59 | 1.16 | 2.87 | 1.63 | 81 |
| short-2048-csa-cp1 | cudnn_flashmla | 0.43 | 0.24 | 155 | 1.16 | 1.75 | 1.01 | 134 |
| short-2048-csa-cp1 | cute | 0.54 | 0.42 | 124 | 1.16 | 2.22 | 1.61 | 105 |
| short-2048-csa-cp1 | cute_ws | 0.26 | 0.21 | 255 | 1.30 | 2.07 | 1.40 | 113 |
| short-2048-csa-cp1 | flashmla_fwd_ref | 0.37 | 0.23 | 181 | 2.58 | - | - | - |
| short-2048-hca-cp1 | tilelang | 1.07 | 0.35 | 31 | 1.46 | 2.51 | 1.14 | 46 |
| short-2048-hca-cp1 | cudnn_flashmla | 0.40 | 0.21 | 82 | 1.46 | 1.56 | 0.79 | 74 |
| short-2048-hca-cp1 | cute | 0.47 | 0.33 | 70 | 1.46 | 1.89 | 1.11 | 62 |
| short-2048-hca-cp1 | cute_ws | 0.22 | 0.17 | 151 | 1.94 | 1.67 | 0.95 | 69 |
| short-2048-hca-cp1 | flashmla_fwd_ref | 0.34 | 0.20 | 97 | 2.07 | - | - | - |
| short-2048-sliding-cp1 | tilelang | 1.00 | 0.31 | 32 | 1.03 | 2.36 | 1.01 | 48 |
| short-2048-sliding-cp1 | cudnn_flashmla | 0.35 | 0.16 | 91 | 1.03 | 1.53 | 0.70 | 74 |
| short-2048-sliding-cp1 | cute | 0.41 | 0.28 | 80 | 1.03 | 1.72 | 0.99 | 65 |
| short-2048-sliding-cp1 | cute_ws | 0.17 | 0.12 | 190 | 1.07 | 1.58 | 0.82 | 71 |
| short-2048-sliding-cp1 | flashmla_fwd_ref | 0.29 | 0.15 | 111 | 1.07 | - | - | - |
| heavy-2048-csa-cp1 | tilelang | 1.15 | 0.47 | 68 | 1.16 | 3.02 | 1.82 | 90 |
| heavy-2048-csa-cp1 | cudnn_flashmla | 0.46 | 0.26 | 171 | 1.16 | 1.81 | 1.09 | 150 |
| heavy-2048-csa-cp1 | cute | 0.57 | 0.45 | 137 | 1.16 | 2.36 | 1.79 | 115 |
| heavy-2048-csa-cp1 | cute_ws | 0.28 | 0.22 | 280 | 1.29 | 2.22 | 1.57 | 123 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref | 0.39 | 0.26 | 200 | 2.21 | - | - | - |
| heavy-2048-hca-cp1 | tilelang | 1.08 | 0.34 | 30 | 1.44 | 2.50 | 1.11 | 46 |
| heavy-2048-hca-cp1 | cudnn_flashmla | 0.40 | 0.20 | 82 | 1.44 | 1.55 | 0.77 | 74 |
| heavy-2048-hca-cp1 | cute | 0.47 | 0.32 | 69 | 1.44 | 1.85 | 1.08 | 61 |
| heavy-2048-hca-cp1 | cute_ws | 0.21 | 0.16 | 152 | 1.92 | 1.67 | 0.92 | 68 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref | 0.34 | 0.19 | 96 | 2.11 | - | - | - |
| heavy-2048-sliding-cp1 | tilelang | 1.01 | 0.30 | 31 | 1.05 | 2.35 | 0.99 | 46 |
| heavy-2048-sliding-cp1 | cudnn_flashmla | 0.35 | 0.16 | 88 | 1.05 | 1.52 | 0.70 | 72 |
| heavy-2048-sliding-cp1 | cute | 0.40 | 0.28 | 77 | 1.05 | 1.71 | 0.97 | 64 |
| heavy-2048-sliding-cp1 | cute_ws | 0.17 | 0.12 | 185 | 1.10 | 1.56 | 0.82 | 70 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref | 0.29 | 0.15 | 107 | 1.10 | - | - | - |
| tiny-2048-csa-cp1 | tilelang | 1.06 | 0.36 | 14 | 3.28 | 2.42 | 1.10 | 22 |
| tiny-2048-csa-cp1 | cudnn_flashmla | 0.40 | 0.21 | 39 | 3.28 | 1.53 | 0.76 | 35 |
| tiny-2048-csa-cp1 | cute | 0.45 | 0.33 | 34 | 3.28 | 1.76 | 1.07 | 30 |
| tiny-2048-csa-cp1 | cute_ws | 0.23 | 0.17 | 67 | 4.40 | 1.59 | 0.92 | 34 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref | 0.34 | 0.20 | 45 | 11.22 | - | - | - |
| tiny-2048-hca-cp1 | tilelang | 0.96 | 0.27 | 13 | 1.79 | 2.15 | 0.77 | 20 |
| tiny-2048-hca-cp1 | cudnn_flashmla | 0.36 | 0.16 | 34 | 1.79 | 1.49 | 0.60 | 29 |
| tiny-2048-hca-cp1 | cute | 0.38 | 0.25 | 33 | 1.79 | 1.51 | 0.75 | 29 |
| tiny-2048-hca-cp1 | cute_ws | 0.17 | 0.12 | 73 | 2.79 | 1.36 | 0.63 | 32 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref | 0.29 | 0.15 | 42 | 2.79 | - | - | - |
| tiny-2048-sliding-cp1 | tilelang | 0.96 | 0.27 | 13 | 1.79 | 2.21 | 0.77 | 20 |
| tiny-2048-sliding-cp1 | cudnn_flashmla | 0.36 | 0.16 | 34 | 1.79 | 1.52 | 0.60 | 28 |
| tiny-2048-sliding-cp1 | cute | 0.38 | 0.25 | 33 | 1.79 | 1.54 | 0.75 | 28 |
| tiny-2048-sliding-cp1 | cute_ws | 0.17 | 0.12 | 73 | 2.79 | 1.38 | 0.63 | 31 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref | 0.30 | 0.15 | 42 | 2.79 | - | - | - |
| single-4096-csa-cp1 | tilelang | 1.90 | 1.21 | 144 | 1.03 | 6.01 | 5.15 | 159 |
| single-4096-csa-cp1 | cudnn_flashmla | 0.78 | 0.60 | 350 | 1.03 | 3.14 | 2.79 | 305 |
| single-4096-csa-cp1 | cute | 1.27 | 1.15 | 215 | 1.03 | 5.33 | 5.10 | 180 |
| single-4096-csa-cp1 | cute_ws | 0.60 | 0.55 | 453 | 1.07 | 4.73 | 4.48 | 203 |
| single-4096-csa-cp1 | flashmla_fwd_ref | 0.71 | 0.59 | 383 | 1.26 | - | - | - |
| single-4096-hca-cp1 | tilelang | 1.43 | 0.69 | 53 | 1.34 | 3.19 | 2.23 | 83 |
| single-4096-hca-cp1 | cudnn_flashmla | 0.57 | 0.37 | 133 | 1.34 | 2.11 | 1.51 | 126 |
| single-4096-hca-cp1 | cute | 0.79 | 0.64 | 96 | 1.34 | 2.55 | 2.18 | 104 |
| single-4096-hca-cp1 | cute_ws | 0.37 | 0.32 | 204 | 1.78 | 2.36 | 1.86 | 113 |
| single-4096-hca-cp1 | flashmla_fwd_ref | 0.51 | 0.36 | 150 | 1.81 | - | - | - |
| single-4096-sliding-cp1 | tilelang | 1.34 | 0.59 | 51 | 1.01 | 2.94 | 1.95 | 81 |
| single-4096-sliding-cp1 | cudnn_flashmla | 0.48 | 0.28 | 140 | 1.01 | 1.99 | 1.30 | 119 |
| single-4096-sliding-cp1 | cute | 0.67 | 0.55 | 101 | 1.01 | 2.28 | 1.90 | 104 |
| single-4096-sliding-cp1 | cute_ws | 0.27 | 0.22 | 251 | 1.02 | 2.12 | 1.58 | 112 |
| single-4096-sliding-cp1 | flashmla_fwd_ref | 0.42 | 0.27 | 160 | 1.02 | - | - | - |
| short-4096-csa-cp1 | tilelang | 1.51 | 0.83 | 85 | 1.18 | 3.90 | 3.05 | 115 |
| short-4096-csa-cp1 | cudnn_flashmla | 0.61 | 0.44 | 209 | 1.18 | 2.36 | 1.85 | 190 |
| short-4096-csa-cp1 | cute | 0.90 | 0.78 | 143 | 1.18 | 3.23 | 3.00 | 139 |
| short-4096-csa-cp1 | cute_ws | 0.43 | 0.37 | 300 | 1.33 | 3.00 | 2.60 | 150 |
| short-4096-csa-cp1 | flashmla_fwd_ref | 0.56 | 0.43 | 232 | 2.67 | - | - | - |
| short-4096-hca-cp1 | tilelang | 1.40 | 0.67 | 47 | 1.46 | 3.09 | 2.13 | 74 |
| short-4096-hca-cp1 | cudnn_flashmla | 0.56 | 0.37 | 116 | 1.46 | 2.04 | 1.44 | 112 |
| short-4096-hca-cp1 | cute | 0.76 | 0.62 | 85 | 1.46 | 2.46 | 2.09 | 93 |
| short-4096-hca-cp1 | cute_ws | 0.36 | 0.30 | 182 | 1.95 | 2.27 | 1.77 | 100 |
| short-4096-hca-cp1 | flashmla_fwd_ref | 0.50 | 0.36 | 130 | 2.11 | - | - | - |
| short-4096-sliding-cp1 | tilelang | 1.29 | 0.59 | 49 | 1.04 | 2.87 | 1.90 | 77 |
| short-4096-sliding-cp1 | cudnn_flashmla | 0.48 | 0.28 | 131 | 1.04 | 1.98 | 1.29 | 112 |
| short-4096-sliding-cp1 | cute | 0.67 | 0.54 | 95 | 1.04 | 2.23 | 1.85 | 100 |
| short-4096-sliding-cp1 | cute_ws | 0.27 | 0.22 | 233 | 1.08 | 2.07 | 1.53 | 107 |
| short-4096-sliding-cp1 | flashmla_fwd_ref | 0.41 | 0.27 | 153 | 1.08 | - | - | - |
| heavy-4096-csa-cp1 | tilelang | 1.47 | 0.77 | 69 | 1.29 | 3.60 | 2.72 | 99 |
| heavy-4096-csa-cp1 | cudnn_flashmla | 0.60 | 0.41 | 170 | 1.29 | 2.28 | 1.70 | 156 |
| heavy-4096-csa-cp1 | cute | 0.84 | 0.73 | 121 | 1.29 | 2.90 | 2.67 | 123 |
| heavy-4096-csa-cp1 | cute_ws | 0.41 | 0.36 | 248 | 1.52 | 2.72 | 2.31 | 131 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref | 0.53 | 0.41 | 192 | 3.37 | - | - | - |
| heavy-4096-hca-cp1 | tilelang | 1.42 | 0.65 | 42 | 1.47 | 3.03 | 2.03 | 68 |
| heavy-4096-hca-cp1 | cudnn_flashmla | 0.55 | 0.35 | 107 | 1.47 | 2.02 | 1.39 | 102 |
| heavy-4096-hca-cp1 | cute | 0.75 | 0.60 | 79 | 1.47 | 2.39 | 1.98 | 87 |
| heavy-4096-hca-cp1 | cute_ws | 0.34 | 0.29 | 172 | 1.96 | 2.19 | 1.67 | 94 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref | 0.49 | 0.34 | 122 | 2.32 | - | - | - |
| heavy-4096-sliding-cp1 | tilelang | 1.28 | 0.58 | 45 | 1.09 | 2.81 | 1.84 | 72 |
| heavy-4096-sliding-cp1 | cudnn_flashmla | 0.48 | 0.28 | 120 | 1.09 | 1.93 | 1.26 | 105 |
| heavy-4096-sliding-cp1 | cute | 0.66 | 0.54 | 88 | 1.09 | 2.16 | 1.80 | 94 |
| heavy-4096-sliding-cp1 | cute_ws | 0.27 | 0.22 | 214 | 1.18 | 2.01 | 1.48 | 101 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref | 0.42 | 0.27 | 139 | 1.18 | - | - | - |
| tiny-4096-csa-cp1 | tilelang | 1.35 | 0.68 | 22 | 3.35 | 2.96 | 2.07 | 35 |
| tiny-4096-csa-cp1 | cudnn_flashmla | 0.55 | 0.38 | 54 | 3.35 | 1.98 | 1.39 | 53 |
| tiny-4096-csa-cp1 | cute | 0.75 | 0.64 | 40 | 3.35 | 2.30 | 2.03 | 45 |
| tiny-4096-csa-cp1 | cute_ws | 0.38 | 0.33 | 79 | 4.50 | 2.16 | 1.72 | 48 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref | 0.49 | 0.37 | 61 | 11.49 | - | - | - |
| tiny-4096-hca-cp1 | tilelang | 1.23 | 0.53 | 20 | 1.81 | 2.44 | 1.43 | 34 |
| tiny-4096-hca-cp1 | cudnn_flashmla | 0.48 | 0.28 | 50 | 1.81 | 1.77 | 1.09 | 48 |
| tiny-4096-hca-cp1 | cute | 0.61 | 0.49 | 39 | 1.81 | 1.81 | 1.40 | 47 |
| tiny-4096-hca-cp1 | cute_ws | 0.28 | 0.22 | 86 | 2.85 | 1.65 | 1.15 | 51 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref | 0.42 | 0.27 | 57 | 2.85 | - | - | - |
| tiny-4096-sliding-cp1 | tilelang | 1.24 | 0.53 | 19 | 1.81 | 2.47 | 1.43 | 34 |
| tiny-4096-sliding-cp1 | cudnn_flashmla | 0.48 | 0.28 | 50 | 1.81 | 1.78 | 1.09 | 47 |
| tiny-4096-sliding-cp1 | cute | 0.61 | 0.49 | 39 | 1.81 | 1.81 | 1.39 | 46 |
| tiny-4096-sliding-cp1 | cute_ws | 0.29 | 0.23 | 84 | 2.85 | 1.68 | 1.15 | 50 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref | 0.41 | 0.27 | 58 | 2.85 | - | - | - |
| single-16384-csa-cp1 | tilelang | 5.90 | 5.52 | 221 | 1.01 | 23.71 | 22.67 | 193 |
| single-16384-csa-cp1 | cudnn_flashmla | 2.63 | 2.57 | 495 | 1.01 | 12.69 | 12.14 | 360 |
| single-16384-csa-cp1 | cute | 5.17 | 4.95 | 253 | 1.01 | 22.85 | 22.61 | 200 |
| single-16384-csa-cp1 | cute_ws | 2.43 | 2.37 | 536 | 1.01 | 20.13 | 19.99 | 227 |
| single-16384-csa-cp1 | flashmla_fwd_ref | 2.76 | 2.54 | 472 | 1.05 | - | - | - |
| single-16384-hca-cp1 | tilelang | 3.54 | 2.85 | 116 | 1.17 | 10.88 | 9.95 | 132 |
| single-16384-hca-cp1 | cudnn_flashmla | 1.56 | 1.40 | 263 | 1.17 | 6.53 | 6.23 | 220 |
| single-16384-hca-cp1 | cute | 2.77 | 2.67 | 148 | 1.17 | 10.05 | 9.77 | 143 |
| single-16384-hca-cp1 | cute_ws | 1.25 | 1.20 | 328 | 1.34 | 8.41 | 8.28 | 171 |
| single-16384-hca-cp1 | flashmla_fwd_ref | 1.49 | 1.40 | 276 | 1.34 | - | - | - |
| single-16384-sliding-cp1 | tilelang | 2.97 | 2.27 | 92 | 1.00 | 8.32 | 7.44 | 115 |
| single-16384-sliding-cp1 | cudnn_flashmla | 1.24 | 1.06 | 220 | 1.00 | 5.34 | 5.01 | 179 |
| single-16384-sliding-cp1 | cute | 2.21 | 2.10 | 124 | 1.00 | 7.50 | 7.26 | 128 |
| single-16384-sliding-cp1 | cute_ws | 0.85 | 0.79 | 322 | 1.00 | 6.07 | 5.95 | 158 |
| single-16384-sliding-cp1 | flashmla_fwd_ref | 1.16 | 1.03 | 236 | 1.00 | - | - | - |
| short-16384-csa-cp1 | tilelang | 4.50 | 3.92 | 166 | 1.11 | 15.99 | 15.01 | 163 |
| short-16384-csa-cp1 | cudnn_flashmla | 2.02 | 1.93 | 369 | 1.11 | 8.84 | 8.56 | 295 |
| short-16384-csa-cp1 | cute | 3.69 | 3.62 | 202 | 1.11 | 15.15 | 14.80 | 172 |
| short-16384-csa-cp1 | cute_ws | 1.79 | 1.73 | 416 | 1.20 | 13.19 | 12.91 | 198 |
| short-16384-csa-cp1 | flashmla_fwd_ref | 1.99 | 1.92 | 376 | 1.84 | - | - | - |
| short-16384-hca-cp1 | tilelang | 3.32 | 2.59 | 83 | 1.42 | 9.14 | 8.24 | 105 |
| short-16384-hca-cp1 | cudnn_flashmla | 1.55 | 1.37 | 178 | 1.42 | 5.88 | 5.59 | 164 |
| short-16384-hca-cp1 | cute | 2.54 | 2.42 | 108 | 1.42 | 8.31 | 8.07 | 116 |
| short-16384-hca-cp1 | cute_ws | 1.23 | 1.18 | 224 | 1.90 | 6.94 | 6.82 | 139 |
| short-16384-hca-cp1 | flashmla_fwd_ref | 1.46 | 1.37 | 189 | 2.00 | - | - | - |
| short-16384-sliding-cp1 | tilelang | 2.96 | 2.26 | 88 | 1.03 | 8.12 | 7.29 | 112 |
| short-16384-sliding-cp1 | cudnn_flashmla | 1.25 | 1.06 | 209 | 1.03 | 5.26 | 4.93 | 174 |
| short-16384-sliding-cp1 | cute | 2.20 | 2.09 | 118 | 1.03 | 7.34 | 7.12 | 125 |
| short-16384-sliding-cp1 | cute_ws | 0.85 | 0.79 | 307 | 1.05 | 5.94 | 5.82 | 154 |
| short-16384-sliding-cp1 | flashmla_fwd_ref | 1.17 | 1.04 | 224 | 1.05 | - | - | - |
| heavy-16384-csa-cp1 | tilelang | 3.97 | 3.36 | 129 | 1.22 | 12.92 | 12.03 | 139 |
| heavy-16384-csa-cp1 | cudnn_flashmla | 1.79 | 1.71 | 287 | 1.22 | 7.45 | 7.22 | 241 |
| heavy-16384-csa-cp1 | cute | 3.19 | 3.12 | 161 | 1.22 | 12.11 | 11.82 | 148 |
| heavy-16384-csa-cp1 | cute_ws | 1.56 | 1.51 | 330 | 1.40 | 10.43 | 10.24 | 172 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref | 1.73 | 1.70 | 296 | 2.68 | - | - | - |
| heavy-16384-hca-cp1 | tilelang | 3.21 | 2.49 | 75 | 1.45 | 8.61 | 7.76 | 98 |
| heavy-16384-hca-cp1 | cudnn_flashmla | 1.47 | 1.32 | 164 | 1.45 | 5.63 | 5.33 | 151 |
| heavy-16384-hca-cp1 | cute | 2.45 | 2.32 | 99 | 1.45 | 7.83 | 7.57 | 108 |
| heavy-16384-hca-cp1 | cute_ws | 1.16 | 1.10 | 208 | 1.94 | 6.48 | 6.37 | 131 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref | 1.40 | 1.30 | 173 | 2.27 | - | - | - |
| heavy-16384-sliding-cp1 | tilelang | 2.96 | 2.26 | 79 | 1.09 | 7.86 | 7.01 | 104 |
| heavy-16384-sliding-cp1 | cudnn_flashmla | 1.25 | 1.07 | 187 | 1.09 | 5.15 | 4.81 | 159 |
| heavy-16384-sliding-cp1 | cute | 2.20 | 2.09 | 107 | 1.09 | 7.07 | 6.85 | 116 |
| heavy-16384-sliding-cp1 | cute_ws | 0.87 | 0.81 | 269 | 1.17 | 5.70 | 5.59 | 144 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref | 1.17 | 1.05 | 200 | 1.17 | - | - | - |
| tiny-16384-csa-cp1 | tilelang | 3.30 | 2.64 | 34 | 3.56 | 8.78 | 7.89 | 45 |
| tiny-16384-csa-cp1 | cudnn_flashmla | 1.54 | 1.45 | 73 | 3.56 | 5.56 | 5.33 | 70 |
| tiny-16384-csa-cp1 | cute | 2.54 | 2.47 | 44 | 3.56 | 7.90 | 7.72 | 50 |
| tiny-16384-csa-cp1 | cute_ws | 1.30 | 1.24 | 86 | 4.79 | 6.61 | 6.50 | 59 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref | 1.48 | 1.44 | 76 | 12.29 | - | - | - |
| tiny-16384-hca-cp1 | tilelang | 2.72 | 2.03 | 33 | 1.89 | 6.18 | 5.35 | 51 |
| tiny-16384-hca-cp1 | cudnn_flashmla | 1.26 | 1.08 | 72 | 1.89 | 4.48 | 4.16 | 70 |
| tiny-16384-hca-cp1 | cute | 1.98 | 1.87 | 46 | 1.89 | 5.40 | 5.19 | 58 |
| tiny-16384-hca-cp1 | cute_ws | 0.98 | 0.92 | 92 | 3.05 | 4.36 | 4.26 | 72 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref | 1.18 | 1.05 | 76 | 3.05 | - | - | - |
| tiny-16384-sliding-cp1 | tilelang | 2.73 | 2.03 | 33 | 1.89 | 6.20 | 5.35 | 51 |
| tiny-16384-sliding-cp1 | cudnn_flashmla | 1.26 | 1.08 | 72 | 1.89 | 4.47 | 4.14 | 71 |
| tiny-16384-sliding-cp1 | cute | 1.98 | 1.86 | 46 | 1.89 | 5.40 | 5.19 | 58 |
| tiny-16384-sliding-cp1 | cute_ws | 0.98 | 0.93 | 92 | 3.05 | 4.36 | 4.26 | 72 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref | 1.18 | 1.05 | 76 | 3.05 | - | - | - |
| single-49208-csa-cp1 | tilelang | 16.49 | 16.42 | 246 | 1.00 | 70.85 | 70.06 | 200 |
| single-49208-csa-cp1 | cudnn_flashmla | 8.30 | 7.76 | 489 | 1.00 | 39.49 | 38.52 | 360 |
| single-49208-csa-cp1 | cute | 15.34 | 15.56 | 265 | 1.00 | 69.67 | 69.84 | 204 |
| single-49208-csa-cp1 | cute_ws | 7.88 | 7.27 | 515 | 1.00 | 61.74 | 61.48 | 230 |
| single-49208-csa-cp1 | flashmla_fwd_ref | 8.53 | 8.40 | 476 | 1.02 | - | - | - |
| single-49208-hca-cp1 | tilelang | 11.54 | 11.15 | 179 | 1.10 | 42.64 | 41.83 | 169 |
| single-49208-hca-cp1 | cudnn_flashmla | 5.45 | 5.37 | 378 | 1.10 | 25.01 | 24.86 | 288 |
| single-49208-hca-cp1 | cute | 10.44 | 10.26 | 198 | 1.10 | 41.53 | 41.45 | 174 |
| single-49208-hca-cp1 | cute_ws | 5.01 | 4.75 | 412 | 1.20 | 35.92 | 35.71 | 201 |
| single-49208-hca-cp1 | flashmla_fwd_ref | 5.64 | 5.53 | 366 | 1.60 | - | - | - |
| single-49208-sliding-cp1 | tilelang | 7.46 | 6.90 | 110 | 1.00 | 22.93 | 22.09 | 126 |
| single-49208-sliding-cp1 | cudnn_flashmla | 3.24 | 3.13 | 255 | 1.00 | 15.38 | 15.04 | 188 |
| single-49208-sliding-cp1 | cute | 6.36 | 6.28 | 130 | 1.00 | 21.82 | 21.58 | 132 |
| single-49208-sliding-cp1 | cute_ws | 2.43 | 2.37 | 340 | 1.00 | 17.83 | 17.67 | 162 |
| single-49208-sliding-cp1 | flashmla_fwd_ref | 3.15 | 3.08 | 262 | 1.00 | - | - | - |
| short-49208-csa-cp1 | tilelang | 13.13 | 12.94 | 202 | 1.07 | 51.17 | 50.33 | 181 |
| short-49208-csa-cp1 | cudnn_flashmla | 6.30 | 6.17 | 420 | 1.07 | 29.43 | 28.64 | 315 |
| short-49208-csa-cp1 | cute | 11.95 | 11.89 | 222 | 1.07 | 50.18 | 50.14 | 185 |
| short-49208-csa-cp1 | cute_ws | 5.82 | 5.61 | 455 | 1.13 | 43.76 | 43.54 | 212 |
| short-49208-csa-cp1 | flashmla_fwd_ref | 6.67 | 6.24 | 397 | 1.56 | - | - | - |
| short-49208-hca-cp1 | tilelang | 8.53 | 7.93 | 105 | 1.36 | 26.19 | 25.37 | 120 |
| short-49208-hca-cp1 | cudnn_flashmla | 4.20 | 4.14 | 213 | 1.36 | 17.41 | 17.21 | 180 |
| short-49208-hca-cp1 | cute | 7.38 | 7.39 | 121 | 1.36 | 25.04 | 24.85 | 125 |
| short-49208-hca-cp1 | cute_ws | 3.57 | 3.52 | 250 | 1.78 | 21.18 | 20.99 | 148 |
| short-49208-hca-cp1 | flashmla_fwd_ref | 4.11 | 4.11 | 217 | 1.85 | - | - | - |
| short-49208-sliding-cp1 | tilelang | 7.49 | 6.95 | 106 | 1.02 | 22.66 | 21.79 | 123 |
| short-49208-sliding-cp1 | cudnn_flashmla | 3.25 | 3.13 | 245 | 1.02 | 15.22 | 14.91 | 183 |
| short-49208-sliding-cp1 | cute | 6.35 | 6.27 | 125 | 1.02 | 21.52 | 21.28 | 129 |
| short-49208-sliding-cp1 | cute_ws | 2.42 | 2.37 | 329 | 1.04 | 17.57 | 17.38 | 159 |
| short-49208-sliding-cp1 | flashmla_fwd_ref | 3.15 | 3.09 | 252 | 1.04 | - | - | - |
| heavy-49208-csa-cp1 | tilelang | 14.94 | 14.90 | 229 | 1.03 | 61.75 | 60.94 | 194 |
| heavy-49208-csa-cp1 | cudnn_flashmla | 7.40 | 7.07 | 462 | 1.03 | 34.67 | 34.23 | 345 |
| heavy-49208-csa-cp1 | cute | 13.84 | 13.74 | 247 | 1.03 | 60.82 | 60.71 | 197 |
| heavy-49208-csa-cp1 | cute_ws | 6.87 | 6.54 | 497 | 1.05 | 53.46 | 53.35 | 224 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref | 7.73 | 7.27 | 442 | 1.21 | - | - | - |
| heavy-49208-hca-cp1 | tilelang | 9.10 | 8.53 | 128 | 1.21 | 29.68 | 28.87 | 137 |
| heavy-49208-hca-cp1 | cudnn_flashmla | 4.42 | 4.37 | 264 | 1.21 | 19.01 | 18.82 | 214 |
| heavy-49208-hca-cp1 | cute | 7.98 | 7.96 | 146 | 1.21 | 28.50 | 28.34 | 143 |
| heavy-49208-hca-cp1 | cute_ws | 3.69 | 3.62 | 315 | 1.43 | 24.21 | 24.03 | 168 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref | 4.37 | 4.27 | 267 | 2.13 | - | - | - |
| heavy-49208-sliding-cp1 | tilelang | 7.46 | 6.78 | 107 | 1.02 | 22.62 | 21.74 | 123 |
| heavy-49208-sliding-cp1 | cudnn_flashmla | 3.25 | 3.13 | 244 | 1.02 | 15.23 | 14.87 | 183 |
| heavy-49208-sliding-cp1 | cute | 6.35 | 6.26 | 125 | 1.02 | 21.49 | 21.25 | 129 |
| heavy-49208-sliding-cp1 | cute_ws | 2.43 | 2.37 | 327 | 1.04 | 17.55 | 17.36 | 159 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref | 3.14 | 3.09 | 253 | 1.04 | - | - | - |
| tiny-49208-csa-cp1 | tilelang | 8.38 | 7.83 | 41 | 3.49 | 24.16 | 23.44 | 50 |
| tiny-49208-csa-cp1 | cudnn_flashmla | 4.34 | 4.29 | 79 | 3.49 | 16.27 | 16.10 | 74 |
| tiny-49208-csa-cp1 | cute | 7.36 | 7.33 | 47 | 3.49 | 23.05 | 22.93 | 52 |
| tiny-49208-csa-cp1 | cute_ws | 3.72 | 3.65 | 92 | 4.69 | 19.43 | 19.24 | 62 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref | 4.27 | 4.28 | 80 | 12.01 | - | - | - |
| tiny-49208-hca-cp1 | tilelang | 6.71 | 6.04 | 41 | 1.86 | 16.76 | 15.89 | 58 |
| tiny-49208-hca-cp1 | cudnn_flashmla | 3.29 | 3.15 | 84 | 1.86 | 12.81 | 12.47 | 76 |
| tiny-49208-hca-cp1 | cute | 5.62 | 5.54 | 49 | 1.86 | 15.67 | 15.39 | 62 |
| tiny-49208-hca-cp1 | cute_ws | 2.74 | 2.71 | 101 | 2.98 | 12.74 | 12.58 | 76 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref | 3.17 | 3.12 | 87 | 2.98 | - | - | - |
| tiny-49208-sliding-cp1 | tilelang | 6.72 | 6.04 | 41 | 1.86 | 16.81 | 15.90 | 58 |
| tiny-49208-sliding-cp1 | cudnn_flashmla | 3.28 | 3.15 | 84 | 1.86 | 12.79 | 12.46 | 76 |
| tiny-49208-sliding-cp1 | cute | 5.63 | 5.54 | 49 | 1.86 | 15.67 | 15.40 | 62 |
| tiny-49208-sliding-cp1 | cute_ws | 2.74 | 2.71 | 101 | 2.98 | 12.73 | 12.56 | 76 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref | 3.17 | 3.11 | 87 | 2.98 | - | - | - |
| single-65536-csa-cp1 | tilelang | 21.72 | 21.69 | 250 | 1.00 | 95.51 | 94.65 | 199 |
| single-65536-csa-cp1 | cudnn_flashmla | 11.46 | 10.39 | 473 | 1.00 | 53.34 | 53.02 | 356 |
| single-65536-csa-cp1 | cute | 20.27 | 20.30 | 268 | 1.00 | 94.01 | 94.45 | 202 |
| single-65536-csa-cp1 | cute_ws | 10.95 | 9.77 | 496 | 1.00 | 84.00 | 84.19 | 226 |
| single-65536-csa-cp1 | flashmla_fwd_ref | 11.47 | 11.43 | 473 | 1.01 | - | - | - |
| single-65536-hca-cp1 | tilelang | 16.55 | 16.23 | 199 | 1.08 | 64.49 | 63.72 | 179 |
| single-65536-hca-cp1 | cudnn_flashmla | 8.30 | 7.99 | 397 | 1.08 | 37.42 | 37.44 | 308 |
| single-65536-hca-cp1 | cute | 15.30 | 15.39 | 215 | 1.08 | 63.17 | 63.29 | 182 |
| single-65536-hca-cp1 | cute_ws | 7.65 | 7.11 | 430 | 1.17 | 55.06 | 54.90 | 209 |
| single-65536-hca-cp1 | flashmla_fwd_ref | 8.43 | 8.37 | 391 | 1.67 | - | - | - |
| single-65536-sliding-cp1 | tilelang | 9.71 | 9.08 | 113 | 1.00 | 30.15 | 29.33 | 127 |
| single-65536-sliding-cp1 | cudnn_flashmla | 4.23 | 4.15 | 260 | 1.00 | 20.29 | 19.99 | 190 |
| single-65536-sliding-cp1 | cute | 8.42 | 8.35 | 130 | 1.00 | 28.89 | 28.65 | 133 |
| single-65536-sliding-cp1 | cute_ws | 3.22 | 3.16 | 341 | 1.00 | 23.64 | 23.50 | 163 |
| single-65536-sliding-cp1 | flashmla_fwd_ref | 4.24 | 4.08 | 259 | 1.00 | - | - | - |
| short-65536-csa-cp1 | tilelang | 16.35 | 16.33 | 196 | 1.08 | 63.22 | 62.49 | 177 |
| short-65536-csa-cp1 | cudnn_flashmla | 8.14 | 7.85 | 393 | 1.08 | 36.74 | 36.54 | 305 |
| short-65536-csa-cp1 | cute | 15.00 | 14.92 | 214 | 1.08 | 62.07 | 62.10 | 181 |
| short-65536-csa-cp1 | cute_ws | 7.66 | 7.10 | 418 | 1.16 | 54.09 | 53.90 | 207 |
| short-65536-csa-cp1 | flashmla_fwd_ref | 8.32 | 8.07 | 385 | 1.72 | - | - | - |
| short-65536-hca-cp1 | tilelang | 10.95 | 10.39 | 103 | 1.40 | 33.59 | 32.79 | 117 |
| short-65536-hca-cp1 | cudnn_flashmla | 5.49 | 5.45 | 205 | 1.40 | 22.68 | 22.54 | 173 |
| short-65536-hca-cp1 | cute | 9.68 | 9.69 | 116 | 1.40 | 32.26 | 32.05 | 122 |
| short-65536-hca-cp1 | cute_ws | 4.74 | 4.72 | 237 | 1.87 | 27.29 | 27.11 | 144 |
| short-65536-hca-cp1 | flashmla_fwd_ref | 5.40 | 5.38 | 208 | 1.96 | - | - | - |
| short-65536-sliding-cp1 | tilelang | 9.66 | 9.06 | 109 | 1.02 | 29.68 | 28.83 | 124 |
| short-65536-sliding-cp1 | cudnn_flashmla | 4.22 | 4.15 | 249 | 1.02 | 20.01 | 19.79 | 184 |
| short-65536-sliding-cp1 | cute | 8.37 | 8.34 | 125 | 1.02 | 28.39 | 28.17 | 129 |
| short-65536-sliding-cp1 | cute_ws | 3.23 | 3.16 | 325 | 1.05 | 23.19 | 23.01 | 158 |
| short-65536-sliding-cp1 | flashmla_fwd_ref | 4.12 | 4.10 | 255 | 1.05 | - | - | - |
| heavy-65536-csa-cp1 | tilelang | 17.46 | 17.34 | 209 | 1.07 | 69.81 | 69.03 | 183 |
| heavy-65536-csa-cp1 | cudnn_flashmla | 8.81 | 8.44 | 415 | 1.07 | 40.16 | 40.24 | 318 |
| heavy-65536-csa-cp1 | cute | 16.18 | 16.20 | 226 | 1.07 | 68.45 | 68.63 | 187 |
| heavy-65536-csa-cp1 | cute_ws | 8.37 | 7.72 | 436 | 1.12 | 60.10 | 59.95 | 213 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref | 9.04 | 8.79 | 404 | 1.50 | - | - | - |
| heavy-65536-hca-cp1 | tilelang | 11.77 | 11.26 | 125 | 1.25 | 38.34 | 37.64 | 134 |
| heavy-65536-hca-cp1 | cudnn_flashmla | 5.86 | 5.85 | 251 | 1.25 | 24.88 | 24.74 | 207 |
| heavy-65536-hca-cp1 | cute | 10.55 | 10.55 | 139 | 1.25 | 37.19 | 37.13 | 138 |
| heavy-65536-hca-cp1 | cute_ws | 5.01 | 4.90 | 293 | 1.52 | 31.45 | 31.26 | 163 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref | 5.80 | 5.71 | 253 | 2.25 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang | 9.62 | 8.99 | 105 | 1.04 | 29.27 | 28.48 | 121 |
| heavy-65536-sliding-cp1 | cudnn_flashmla | 4.23 | 4.16 | 239 | 1.04 | 19.87 | 19.58 | 178 |
| heavy-65536-sliding-cp1 | cute | 8.33 | 8.31 | 121 | 1.04 | 28.00 | 27.79 | 127 |
| heavy-65536-sliding-cp1 | cute_ws | 3.23 | 3.17 | 314 | 1.09 | 22.84 | 22.64 | 155 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref | 4.16 | 4.11 | 243 | 1.09 | - | - | - |
| tiny-65536-csa-cp1 | tilelang | 10.92 | 10.43 | 42 | 3.45 | 31.83 | 31.18 | 51 |
| tiny-65536-csa-cp1 | cudnn_flashmla | 5.73 | 5.70 | 81 | 3.45 | 21.54 | 21.43 | 75 |
| tiny-65536-csa-cp1 | cute | 9.78 | 9.74 | 47 | 3.45 | 30.61 | 30.52 | 53 |
| tiny-65536-csa-cp1 | cute_ws | 4.92 | 4.88 | 94 | 4.64 | 25.80 | 25.61 | 63 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref | 5.67 | 5.69 | 82 | 11.87 | - | - | - |
| tiny-65536-hca-cp1 | tilelang | 8.69 | 8.04 | 43 | 1.85 | 22.03 | 21.20 | 59 |
| tiny-65536-hca-cp1 | cudnn_flashmla | 4.28 | 4.18 | 87 | 1.85 | 16.85 | 16.56 | 78 |
| tiny-65536-hca-cp1 | cute | 7.43 | 7.36 | 50 | 1.85 | 20.74 | 20.54 | 63 |
| tiny-65536-hca-cp1 | cute_ws | 3.65 | 3.62 | 102 | 2.95 | 16.95 | 16.76 | 77 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref | 4.16 | 4.11 | 90 | 2.95 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang | 8.69 | 8.03 | 43 | 1.85 | 22.01 | 21.20 | 59 |
| tiny-65536-sliding-cp1 | cudnn_flashmla | 4.27 | 4.20 | 87 | 1.85 | 16.86 | 16.58 | 77 |
| tiny-65536-sliding-cp1 | cute | 7.43 | 7.37 | 50 | 1.85 | 20.74 | 20.53 | 63 |
| tiny-65536-sliding-cp1 | cute_ws | 3.67 | 3.57 | 102 | 2.95 | 16.93 | 16.75 | 77 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref | 4.17 | 4.13 | 90 | 2.95 | - | - | - |

## Host overhead (op-boundary minus GPU busy time), µs

| arm | mode | min | median | max | items where it exceeds GPU time |
|---|---|---|---|---|---|
| cudnn_flashmla | fwd | 15 | 198 | 1072 | 96 of 240 |
| cudnn_flashmla | fwd_bwd | -78 | 681 | 1356 | 96 of 240 |
| cute | fwd | -222 | 121 | 212 | 70 of 240 |
| cute | fwd_bwd | -440 | 418 | 1375 | 79 of 240 |
| cute_ws | fwd | 20 | 54 | 1172 | 58 of 240 |
| cute_ws | fwd_bwd | -188 | 539 | 1218 | 80 of 240 |
| flashmla_fwd_ref | fwd | -20 | 143 | 452 | 79 of 240 |
| tilelang | fwd | 16 | 700 | 807 | 127 of 240 |
| tilelang | fwd_bwd | 656 | 1021 | 2041 | 106 of 240 |

## Correctness gate

Largest relative error over all items (max deviation over the reference's max magnitude), and its bound.

| arm | basis | tensor | max relative error | bound | failures |
|---|---|---|---|---|---|
| cudnn_flashmla | vs_dense_fp32 | dkv | 4.40e-03 | 1e-02 | 0 |
| cudnn_flashmla | vs_dense_fp32 | dq | 6.81e-03 | 1e-02 | 0 |
| cudnn_flashmla | vs_dense_fp32 | dsink | 6.67e-03 | 1e-02 | 0 |
| cudnn_flashmla | vs_dense_fp32 | lse | 6.62e-07 | 1e-06 | 0 |
| cudnn_flashmla | vs_dense_fp32 | out | 3.62e-03 | 1e-02 | 0 |
| cudnn_flashmla | vs_tilelang | dkv | 7.30e-03 | 1e-02 | 0 |
| cudnn_flashmla | vs_tilelang | dq | 7.14e-03 | 1e-02 | 0 |
| cudnn_flashmla | vs_tilelang | dsink | 9.52e-03 | 1e-02 | 0 |
| cudnn_flashmla | vs_tilelang | lse | 4.17e-07 | 1e-06 | 0 |
| cudnn_flashmla | vs_tilelang | out | 5.15e-03 | 1e-02 | 0 |
| cute | vs_dense_fp32 | dkv | 4.40e-03 | 1e-02 | 0 |
| cute | vs_dense_fp32 | dq | 6.81e-03 | 1e-02 | 0 |
| cute | vs_dense_fp32 | dsink | 9.56e-03 | 1e-02 | 0 |
| cute | vs_dense_fp32 | lse | 6.20e-07 | 1e-06 | 0 |
| cute | vs_dense_fp32 | out | 3.62e-03 | 1e-02 | 0 |
| cute | vs_tilelang | dkv | 5.88e-03 | 1e-02 | 0 |
| cute | vs_tilelang | dq | 2.99e-03 | 1e-02 | 0 |
| cute | vs_tilelang | dsink | 6.05e-05 | 1e-02 | 0 |
| cute | vs_tilelang | lse | 1.20e-07 | 5e-07 | 0 |
| cute | vs_tilelang | out | 3.62e-03 | 1e-02 | 0 |
| cute_ws | vs_dense_fp32 | dkv | 4.40e-03 | 1e-02 | 0 |
| cute_ws | vs_dense_fp32 | dq | 6.81e-03 | 1e-02 | 0 |
| cute_ws | vs_dense_fp32 | dsink | 9.04e-03 | 1e-02 | 0 |
| cute_ws | vs_dense_fp32 | lse | 6.30e-07 | 1e-06 | 0 |
| cute_ws | vs_dense_fp32 | out | 3.62e-03 | 1e-02 | 0 |
| cute_ws | vs_tilelang | dkv | 6.49e-03 | 1e-02 | 0 |
| cute_ws | vs_tilelang | dq | 7.14e-03 | 1e-02 | 0 |
| cute_ws | vs_tilelang | dsink | 4.57e-03 | 1e-02 | 0 |
| cute_ws | vs_tilelang | lse | 4.17e-07 | 5e-07 | 0 |
| cute_ws | vs_tilelang | out | 5.15e-03 | 1e-02 | 0 |
| flashmla_fwd_ref | vs_dense_fp32 | out | 3.62e-03 | 1e-02 | 0 |
| flashmla_fwd_ref | vs_tilelang | out | 5.15e-03 | 1e-02 | 0 |
| tilelang | vs_dense_fp32 | dkv | 4.40e-03 | 1e-02 | 0 |
| tilelang | vs_dense_fp32 | dq | 6.81e-03 | 1e-02 | 0 |
| tilelang | vs_dense_fp32 | dsink | 9.56e-03 | 1e-02 | 0 |
| tilelang | vs_dense_fp32 | lse | 6.20e-07 | 1e-06 | 0 |
| tilelang | vs_dense_fp32 | out | 3.62e-03 | 1e-02 | 0 |

Arms that raised: 0. 

## Dynamic stream

32 items replayed once each in a fresh process with fresh compile caches; seconds,
lower is better. `rest` sums every item after the first, so it excludes per-process costs.

| arm | phase | first item | rest | compiles | disk loads | prime_rl import | TileLang compile |
|---|---|---|---|---|---|---|---|
| tilelang | cold | 20.73 | 0.615 | 4 | 0 | 41.5 | 15.9 |
| tilelang | warm | 0.28 | 0.616 | 0 | 4 | 41.0 | 0.0 |
| cudnn_flashmla | cold | 6.98 | 7.968 | 7 | 0 | 41.5 | 6.0 |
| cudnn_flashmla | warm | 3.45 | 7.940 | 6 | 1 | 40.5 | 3.3 |
| cute | cold | 14.03 | 0.586 | 4 | 0 | 41.1 | 9.5 |
| cute | warm | 1.44 | 0.583 | 1 | 3 | 40.5 | 0.0 |
| cute_ws | cold | 15.72 | 0.512 | 4 | 0 | 40.9 | 9.5 |
| cute_ws | warm | 3.16 | 0.512 | 1 | 3 | 40.4 | 0.0 |

## Valid slots per query

Share of an item's queries (%) whose valid-slot count falls in each 64-slot bin; the last
bin is closed. `slots` is the item's gather width, `med` the median valid count per query.

| item | slots | med | 0- | 64- | 128- | 192- | 256- | 320- | 384- | 448- | 512- | 576- |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | 640 | 384 | 2 | 2 | 7 | 12 | 12 | 12 | 12 | 12 | 12 | 13 |
| single-2048-csa-cp8r0 | 640 | 160 | 20 | 20 | 60 | 0 |  |  |  |  |  |  |
| single-2048-csa-cp8r4 | 640 | 416 |  |  |  |  |  |  | 100 | 0 |  |  |
| single-2048-csa-cp8r7 | 640 | 608 |  |  |  |  |  |  |  |  |  | 100 |
| single-2048-hca-cp1 | 144 | 136 | 3 | 3 | 94 |  |  |  |  |  |  |  |
| single-2048-hca-cp8r0 | 144 | 129 | 25 | 25 | 50 |  |  |  |  |  |  |  |
| single-2048-hca-cp8r4 | 144 | 137 |  |  | 100 |  |  |  |  |  |  |  |
| single-2048-hca-cp8r7 | 144 | 143 |  |  | 100 |  |  |  |  |  |  |  |
| single-2048-sliding-cp1 | 128 | 128 | 3 | 3 | 94 |  |  |  |  |  |  |  |
| single-2048-sliding-cp8r0 | 128 | 128 | 25 | 25 | 50 |  |  |  |  |  |  |  |
| single-2048-sliding-cp8r4 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| single-2048-sliding-cp8r7 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| short-2048-csa-cp1 | 640 | 256 | 5 | 5 | 15 | 25 | 25 | 23 | 2 |  |  |  |
| short-2048-csa-cp8r0 | 640 | 160 | 20 | 20 | 60 | 0 |  |  |  |  |  |  |
| short-2048-csa-cp8r4 | 640 | 160 | 20 | 20 | 41 |  |  |  | 19 |  |  |  |
| short-2048-csa-cp8r7 | 640 | 339 |  |  |  |  | 19 | 81 |  |  |  |  |
| short-2048-hca-cp1 | 136 | 132 | 6 | 6 | 88 |  |  |  |  |  |  |  |
| short-2048-hca-cp8r0 | 136 | 129 | 25 | 25 | 50 |  |  |  |  |  |  |  |
| short-2048-hca-cp8r4 | 136 | 129 | 25 | 25 | 50 |  |  |  |  |  |  |  |
| short-2048-hca-cp8r7 | 136 | 134 |  |  | 100 |  |  |  |  |  |  |  |
| short-2048-sliding-cp1 | 128 | 128 | 6 | 6 | 88 |  |  |  |  |  |  |  |
| short-2048-sliding-cp8r0 | 128 | 128 | 25 | 25 | 50 |  |  |  |  |  |  |  |
| short-2048-sliding-cp8r4 | 128 | 128 | 25 | 25 | 50 |  |  |  |  |  |  |  |
| short-2048-sliding-cp8r7 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| heavy-2048-csa-cp1 | 640 | 289 | 7 | 7 | 16 | 12 | 12 | 12 | 12 | 12 | 6 |  |
| heavy-2048-csa-cp8r0 | 640 | 80 | 40 | 30 | 30 |  |  |  |  |  |  |  |
| heavy-2048-csa-cp8r4 | 640 | 321 |  |  |  |  | 48 | 52 |  |  |  |  |
| heavy-2048-csa-cp8r7 | 640 | 513 |  |  |  |  |  |  |  | 48 | 52 |  |
| heavy-2048-hca-cp1 | 141 | 133 | 9 | 9 | 81 |  |  |  |  |  |  |  |
| heavy-2048-hca-cp8r0 | 141 | 64 | 49 | 31 | 20 |  |  |  |  |  |  |  |
| heavy-2048-hca-cp8r4 | 141 | 134 |  |  | 100 |  |  |  |  |  |  |  |
| heavy-2048-hca-cp8r7 | 141 | 140 |  |  | 100 |  |  |  |  |  |  |  |
| heavy-2048-sliding-cp1 | 128 | 128 | 9 | 9 | 81 |  |  |  |  |  |  |  |
| heavy-2048-sliding-cp8r0 | 128 | 64 | 49 | 31 | 20 |  |  |  |  |  |  |  |
| heavy-2048-sliding-cp8r4 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| heavy-2048-sliding-cp8r7 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| tiny-2048-csa-cp1 | 640 | 51 | 61 | 34 | 5 |  |  |  |  |  |  |  |
| tiny-2048-csa-cp8r0 | 640 | 47 | 65 | 25 | 10 |  |  |  |  |  |  |  |
| tiny-2048-csa-cp8r4 | 640 | 33 | 80 | 20 |  |  |  |  |  |  |  |  |
| tiny-2048-csa-cp8r7 | 640 | 77 | 40 | 52 | 8 |  |  |  |  |  |  |  |
| tiny-2048-hca-cp1 | 128 | 41 | 71 | 29 |  |  |  |  |  |  |  |  |
| tiny-2048-hca-cp8r0 | 128 | 38 | 75 | 25 |  |  |  |  |  |  |  |  |
| tiny-2048-hca-cp8r4 | 128 | 27 | 86 | 14 |  |  |  |  |  |  |  |  |
| tiny-2048-hca-cp8r7 | 128 | 62 | 52 | 48 |  |  |  |  |  |  |  |  |
| tiny-2048-sliding-cp1 | 128 | 41 | 71 | 29 |  |  |  |  |  |  |  |  |
| tiny-2048-sliding-cp8r0 | 128 | 38 | 75 | 25 |  |  |  |  |  |  |  |  |
| tiny-2048-sliding-cp8r4 | 128 | 27 | 86 | 14 |  |  |  |  |  |  |  |  |
| tiny-2048-sliding-cp8r7 | 128 | 62 | 52 | 48 |  |  |  |  |  |  |  |  |
| single-4096-csa-cp1 | 640 | 640 | 1 | 1 | 4 | 6 | 6 | 6 | 6 | 6 | 6 | 56 |
| single-4096-csa-cp8r0 | 640 | 192 | 10 | 10 | 30 | 50 | 0 |  |  |  |  |  |
| single-4096-csa-cp8r4 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| single-4096-csa-cp8r7 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| single-4096-hca-cp1 | 160 | 144 | 2 | 2 | 97 |  |  |  |  |  |  |  |
| single-4096-hca-cp8r0 | 160 | 130 | 12 | 12 | 75 |  |  |  |  |  |  |  |
| single-4096-hca-cp8r4 | 160 | 146 |  |  | 100 |  |  |  |  |  |  |  |
| single-4096-hca-cp8r7 | 160 | 158 |  |  | 100 |  |  |  |  |  |  |  |
| single-4096-sliding-cp1 | 128 | 128 | 2 | 2 | 97 |  |  |  |  |  |  |  |
| single-4096-sliding-cp8r0 | 128 | 128 | 12 | 12 | 75 |  |  |  |  |  |  |  |
| single-4096-sliding-cp8r4 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| single-4096-sliding-cp8r7 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| short-4096-csa-cp1 | 640 | 238 | 6 | 6 | 19 | 26 | 20 | 18 | 6 |  |  |  |
| short-4096-csa-cp8r0 | 640 | 192 | 10 | 10 | 30 | 50 | 0 |  |  |  |  |  |
| short-4096-csa-cp8r4 | 640 | 199 | 4 | 10 | 30 | 50 | 6 |  |  |  |  |  |
| short-4096-csa-cp8r7 | 640 | 192 | 10 | 10 | 30 | 43 | 8 |  |  |  |  |  |
| short-4096-hca-cp1 | 137 | 131 | 8 | 8 | 84 |  |  |  |  |  |  |  |
| short-4096-hca-cp8r0 | 137 | 130 | 12 | 12 | 75 |  |  |  |  |  |  |  |
| short-4096-hca-cp8r4 | 137 | 130 | 6 | 12 | 81 |  |  |  |  |  |  |  |
| short-4096-hca-cp8r7 | 137 | 130 | 12 | 12 | 75 |  |  |  |  |  |  |  |
| short-4096-sliding-cp1 | 128 | 128 | 8 | 8 | 84 |  |  |  |  |  |  |  |
| short-4096-sliding-cp8r0 | 128 | 128 | 12 | 12 | 75 |  |  |  |  |  |  |  |
| short-4096-sliding-cp8r4 | 128 | 128 | 6 | 12 | 81 |  |  |  |  |  |  |  |
| short-4096-sliding-cp8r7 | 128 | 128 | 12 | 12 | 75 |  |  |  |  |  |  |  |
| heavy-4096-csa-cp1 | 640 | 187 | 12 | 12 | 27 | 24 | 16 | 6 | 2 |  |  |  |
| heavy-4096-csa-cp8r0 | 640 | 192 | 10 | 10 | 30 | 50 | 0 |  |  |  |  |  |
| heavy-4096-csa-cp8r4 | 640 | 201 | 2 | 10 | 30 | 50 | 8 |  |  |  |  |  |
| heavy-4096-csa-cp8r7 | 640 | 160 | 20 | 20 | 60 |  |  |  |  |  |  |  |
| heavy-4096-hca-cp1 | 136 | 129 | 15 | 16 | 69 |  |  |  |  |  |  |  |
| heavy-4096-hca-cp8r0 | 136 | 130 | 12 | 12 | 75 |  |  |  |  |  |  |  |
| heavy-4096-hca-cp8r4 | 136 | 130 | 5 | 12 | 83 |  |  |  |  |  |  |  |
| heavy-4096-hca-cp8r7 | 136 | 129 | 25 | 25 | 50 |  |  |  |  |  |  |  |
| heavy-4096-sliding-cp1 | 128 | 128 | 15 | 16 | 69 |  |  |  |  |  |  |  |
| heavy-4096-sliding-cp8r0 | 128 | 128 | 12 | 12 | 75 |  |  |  |  |  |  |  |
| heavy-4096-sliding-cp8r4 | 128 | 128 | 5 | 12 | 83 |  |  |  |  |  |  |  |
| heavy-4096-sliding-cp8r7 | 128 | 128 | 25 | 25 | 50 |  |  |  |  |  |  |  |
| tiny-4096-csa-cp1 | 640 | 48 | 63 | 31 | 6 |  |  |  |  |  |  |  |
| tiny-4096-csa-cp8r0 | 640 | 43 | 68 | 29 | 4 |  |  |  |  |  |  |  |
| tiny-4096-csa-cp8r4 | 640 | 58 | 54 | 37 | 9 |  |  |  |  |  |  |  |
| tiny-4096-csa-cp8r7 | 640 | 52 | 58 | 33 | 10 |  |  |  |  |  |  |  |
| tiny-4096-hca-cp1 | 128 | 39 | 73 | 27 |  |  |  |  |  |  |  |  |
| tiny-4096-hca-cp8r0 | 128 | 35 | 77 | 23 |  |  |  |  |  |  |  |  |
| tiny-4096-hca-cp8r4 | 128 | 47 | 65 | 35 |  |  |  |  |  |  |  |  |
| tiny-4096-hca-cp8r7 | 128 | 42 | 67 | 33 |  |  |  |  |  |  |  |  |
| tiny-4096-sliding-cp1 | 128 | 39 | 73 | 27 |  |  |  |  |  |  |  |  |
| tiny-4096-sliding-cp8r0 | 128 | 35 | 77 | 23 |  |  |  |  |  |  |  |  |
| tiny-4096-sliding-cp8r4 | 128 | 47 | 65 | 35 |  |  |  |  |  |  |  |  |
| tiny-4096-sliding-cp8r7 | 128 | 42 | 67 | 33 |  |  |  |  |  |  |  |  |
| single-16384-csa-cp1 | 640 | 640 | 0 | 0 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 89 |
| single-16384-csa-cp8r0 | 640 | 384 | 2 | 2 | 7 | 12 | 12 | 12 | 12 | 12 | 12 | 13 |
| single-16384-csa-cp8r4 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| single-16384-csa-cp8r7 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| single-16384-hca-cp1 | 256 | 192 | 0 | 0 | 49 | 50 | 0 |  |  |  |  |  |
| single-16384-hca-cp8r0 | 256 | 136 | 3 | 3 | 94 |  |  |  |  |  |  |  |
| single-16384-hca-cp8r4 | 256 | 200 |  |  |  | 100 |  |  |  |  |  |  |
| single-16384-hca-cp8r7 | 256 | 248 |  |  |  | 100 | 0 |  |  |  |  |  |
| single-16384-sliding-cp1 | 128 | 128 | 0 | 0 | 99 |  |  |  |  |  |  |  |
| single-16384-sliding-cp8r0 | 128 | 128 | 3 | 3 | 94 |  |  |  |  |  |  |  |
| single-16384-sliding-cp8r4 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| single-16384-sliding-cp8r7 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| short-16384-csa-cp1 | 640 | 330 | 4 | 4 | 11 | 16 | 13 | 11 | 11 | 9 | 8 | 13 |
| short-16384-csa-cp8r0 | 640 | 288 | 5 | 5 | 15 | 19 | 12 | 12 | 12 | 12 | 6 |  |
| short-16384-csa-cp8r4 | 640 | 466 | 2 | 2 | 7 | 10 |  | 11 | 12 | 12 | 12 | 29 |
| short-16384-csa-cp8r7 | 640 | 408 | 5 | 5 | 10 | 12 | 11 | 2 | 12 | 12 | 12 | 17 |
| short-16384-hca-cp1 | 147 | 134 | 5 | 5 | 90 |  |  |  |  |  |  |  |
| short-16384-hca-cp8r0 | 147 | 133 | 6 | 6 | 88 |  |  |  |  |  |  |  |
| short-16384-hca-cp8r4 | 147 | 138 | 3 | 3 | 94 |  |  |  |  |  |  |  |
| short-16384-hca-cp8r7 | 147 | 136 | 6 | 6 | 88 |  |  |  |  |  |  |  |
| short-16384-sliding-cp1 | 128 | 128 | 5 | 5 | 90 |  |  |  |  |  |  |  |
| short-16384-sliding-cp8r0 | 128 | 128 | 6 | 6 | 88 |  |  |  |  |  |  |  |
| short-16384-sliding-cp8r4 | 128 | 128 | 3 | 3 | 94 |  |  |  |  |  |  |  |
| short-16384-sliding-cp8r7 | 128 | 128 | 6 | 6 | 88 |  |  |  |  |  |  |  |
| heavy-16384-csa-cp1 | 640 | 197 | 12 | 12 | 24 | 18 | 9 | 6 | 5 | 5 | 4 | 5 |
| heavy-16384-csa-cp8r0 | 640 | 174 | 15 | 15 | 31 | 27 | 13 |  |  |  |  |  |
| heavy-16384-csa-cp8r4 | 640 | 220 | 7 | 7 | 22 | 27 | 12 | 12 | 11 |  |  |  |
| heavy-16384-csa-cp8r7 | 640 | 440 |  |  | 1 | 12 | 12 | 12 | 12 | 12 | 12 | 24 |
| heavy-16384-hca-cp1 | 145 | 130 | 15 | 15 | 71 |  |  |  |  |  |  |  |
| heavy-16384-hca-cp8r0 | 145 | 129 | 18 | 19 | 63 |  |  |  |  |  |  |  |
| heavy-16384-hca-cp8r4 | 145 | 130 | 9 | 9 | 82 |  |  |  |  |  |  |  |
| heavy-16384-hca-cp8r7 | 145 | 137 |  |  | 100 |  |  |  |  |  |  |  |
| heavy-16384-sliding-cp1 | 128 | 128 | 15 | 15 | 71 |  |  |  |  |  |  |  |
| heavy-16384-sliding-cp8r0 | 128 | 128 | 18 | 19 | 63 |  |  |  |  |  |  |  |
| heavy-16384-sliding-cp8r4 | 128 | 128 | 9 | 9 | 82 |  |  |  |  |  |  |  |
| heavy-16384-sliding-cp8r7 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| tiny-16384-csa-cp1 | 640 | 45 | 65 | 31 | 4 |  |  |  |  |  |  |  |
| tiny-16384-csa-cp8r0 | 640 | 46 | 62 | 33 | 4 |  |  |  |  |  |  |  |
| tiny-16384-csa-cp8r4 | 640 | 45 | 66 | 31 | 3 |  |  |  |  |  |  |  |
| tiny-16384-csa-cp8r7 | 640 | 40 | 71 | 27 | 2 |  |  |  |  |  |  |  |
| tiny-16384-hca-cp1 | 128 | 36 | 75 | 25 |  |  |  |  |  |  |  |  |
| tiny-16384-hca-cp8r0 | 128 | 37 | 73 | 27 |  |  |  |  |  |  |  |  |
| tiny-16384-hca-cp8r4 | 128 | 36 | 77 | 23 |  |  |  |  |  |  |  |  |
| tiny-16384-hca-cp8r7 | 128 | 32 | 81 | 19 |  |  |  |  |  |  |  |  |
| tiny-16384-sliding-cp1 | 128 | 36 | 75 | 25 |  |  |  |  |  |  |  |  |
| tiny-16384-sliding-cp8r0 | 128 | 37 | 73 | 27 |  |  |  |  |  |  |  |  |
| tiny-16384-sliding-cp8r4 | 128 | 36 | 77 | 23 |  |  |  |  |  |  |  |  |
| tiny-16384-sliding-cp8r7 | 128 | 32 | 81 | 19 |  |  |  |  |  |  |  |  |
| single-49208-csa-cp1 | 640 | 640 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 96 |
| single-49208-csa-cp8r0 | 640 | 640 | 1 | 1 | 2 | 4 | 4 | 4 | 4 | 4 | 4 | 71 |
| single-49208-csa-cp8r4 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| single-49208-csa-cp8r7 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| single-49208-hca-cp1 | 512 | 320 | 0 | 0 | 16 | 17 | 17 | 17 | 17 | 17 | 0 |  |
| single-49208-hca-cp8r0 | 512 | 152 | 1 | 1 | 98 |  |  |  |  |  |  |  |
| single-49208-hca-cp8r4 | 512 | 344 |  |  |  |  |  | 100 |  |  |  |  |
| single-49208-hca-cp8r7 | 512 | 488 |  |  |  |  |  |  |  | 99 | 1 |  |
| single-49208-sliding-cp1 | 128 | 128 | 0 | 0 | 100 |  |  |  |  |  |  |  |
| single-49208-sliding-cp8r0 | 128 | 128 | 1 | 1 | 98 |  |  |  |  |  |  |  |
| single-49208-sliding-cp8r4 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| single-49208-sliding-cp8r7 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| short-49208-csa-cp1 | 640 | 400 | 3 | 3 | 9 | 13 | 11 | 10 | 8 | 7 | 5 | 32 |
| short-49208-csa-cp8r0 | 640 | 265 | 5 | 5 | 15 | 23 | 13 | 12 | 8 | 8 | 8 | 2 |
| short-49208-csa-cp8r4 | 640 | 581 | 2 | 2 | 5 | 8 | 8 | 8 | 8 | 4 | 4 | 50 |
| short-49208-csa-cp8r7 | 640 | 346 | 2 | 2 | 7 | 15 | 17 | 14 | 12 | 12 | 8 | 10 |
| short-49208-hca-cp1 | 210 | 136 | 4 | 4 | 88 | 5 |  |  |  |  |  |  |
| short-49208-hca-cp8r0 | 210 | 132 | 6 | 6 | 88 |  |  |  |  |  |  |  |
| short-49208-hca-cp8r4 | 210 | 142 | 2 | 2 | 96 |  |  |  |  |  |  |  |
| short-49208-hca-cp8r7 | 210 | 134 | 3 | 3 | 94 |  |  |  |  |  |  |  |
| short-49208-sliding-cp1 | 128 | 128 | 4 | 4 | 93 |  |  |  |  |  |  |  |
| short-49208-sliding-cp8r0 | 128 | 128 | 6 | 6 | 88 |  |  |  |  |  |  |  |
| short-49208-sliding-cp8r4 | 128 | 128 | 2 | 2 | 96 |  |  |  |  |  |  |  |
| short-49208-sliding-cp8r7 | 128 | 128 | 3 | 3 | 94 |  |  |  |  |  |  |  |
| heavy-49208-csa-cp1 | 640 | 640 | 3 | 3 | 7 | 5 | 3 | 2 | 2 | 2 | 2 | 73 |
| heavy-49208-csa-cp8r0 | 640 | 199 | 12 | 12 | 24 | 18 | 12 | 5 | 4 | 4 | 4 | 4 |
| heavy-49208-csa-cp8r4 | 640 | 640 |  |  |  | 1 | 4 | 4 | 4 | 4 | 4 | 79 |
| heavy-49208-csa-cp8r7 | 640 | 640 | 2 | 2 | 5 | 8 | 4 | 4 | 4 | 4 | 4 | 63 |
| heavy-49208-hca-cp1 | 289 | 178 | 4 | 4 | 50 | 33 | 10 |  |  |  |  |  |
| heavy-49208-hca-cp8r0 | 289 | 130 | 14 | 15 | 71 |  |  |  |  |  |  |  |
| heavy-49208-hca-cp8r4 | 289 | 155 |  |  | 100 |  |  |  |  |  |  |  |
| heavy-49208-hca-cp8r7 | 289 | 148 | 2 | 2 | 67 |  | 29 |  |  |  |  |  |
| heavy-49208-sliding-cp1 | 128 | 128 | 4 | 4 | 93 |  |  |  |  |  |  |  |
| heavy-49208-sliding-cp8r0 | 128 | 128 | 14 | 15 | 71 |  |  |  |  |  |  |  |
| heavy-49208-sliding-cp8r4 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| heavy-49208-sliding-cp8r7 | 128 | 128 | 2 | 2 | 96 |  |  |  |  |  |  |  |
| tiny-49208-csa-cp1 | 640 | 47 | 64 | 32 | 4 |  |  |  |  |  |  |  |
| tiny-49208-csa-cp8r0 | 640 | 47 | 64 | 32 | 4 |  |  |  |  |  |  |  |
| tiny-49208-csa-cp8r4 | 640 | 50 | 62 | 33 | 5 |  |  |  |  |  |  |  |
| tiny-49208-csa-cp8r7 | 640 | 45 | 66 | 30 | 3 |  |  |  |  |  |  |  |
| tiny-49208-hca-cp1 | 128 | 38 | 74 | 26 |  |  |  |  |  |  |  |  |
| tiny-49208-hca-cp8r0 | 128 | 38 | 75 | 25 |  |  |  |  |  |  |  |  |
| tiny-49208-hca-cp8r4 | 128 | 40 | 71 | 29 |  |  |  |  |  |  |  |  |
| tiny-49208-hca-cp8r7 | 128 | 36 | 77 | 23 |  |  |  |  |  |  |  |  |
| tiny-49208-sliding-cp1 | 128 | 38 | 74 | 26 |  |  |  |  |  |  |  |  |
| tiny-49208-sliding-cp8r0 | 128 | 38 | 75 | 25 |  |  |  |  |  |  |  |  |
| tiny-49208-sliding-cp8r4 | 128 | 40 | 71 | 29 |  |  |  |  |  |  |  |  |
| tiny-49208-sliding-cp8r7 | 128 | 36 | 77 | 23 |  |  |  |  |  |  |  |  |
| single-65536-csa-cp1 | 640 | 640 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 97 |
| single-65536-csa-cp8r0 | 640 | 640 | 1 | 1 | 2 | 3 | 3 | 3 | 3 | 3 | 3 | 78 |
| single-65536-csa-cp8r4 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| single-65536-csa-cp8r7 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| single-65536-hca-cp1 | 640 | 384 | 0 | 0 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 13 |
| single-65536-hca-cp8r0 | 640 | 160 | 1 | 1 | 98 | 0 |  |  |  |  |  |  |
| single-65536-hca-cp8r4 | 640 | 416 |  |  |  |  |  |  | 100 | 0 |  |  |
| single-65536-hca-cp8r7 | 640 | 608 |  |  |  |  |  |  |  |  |  | 100 |
| single-65536-sliding-cp1 | 128 | 128 | 0 | 0 | 100 |  |  |  |  |  |  |  |
| single-65536-sliding-cp8r0 | 128 | 128 | 1 | 1 | 98 |  |  |  |  |  |  |  |
| single-65536-sliding-cp8r4 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| single-65536-sliding-cp8r7 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| short-65536-csa-cp1 | 640 | 338 | 4 | 4 | 11 | 16 | 13 | 10 | 8 | 6 | 5 | 25 |
| short-65536-csa-cp8r0 | 640 | 236 | 6 | 6 | 19 | 27 | 22 | 15 | 5 |  |  |  |
| short-65536-csa-cp8r4 | 640 | 469 | 2 | 2 | 6 | 9 | 9 | 9 | 9 | 9 | 6 | 38 |
| short-65536-csa-cp8r7 | 640 | 331 | 4 | 4 | 11 | 16 | 13 | 8 | 6 | 6 | 6 | 26 |
| short-65536-hca-cp1 | 160 | 134 | 5 | 5 | 91 |  |  |  |  |  |  |  |
| short-65536-hca-cp8r0 | 160 | 131 | 8 | 8 | 84 |  |  |  |  |  |  |  |
| short-65536-hca-cp8r4 | 160 | 138 | 2 | 2 | 95 |  |  |  |  |  |  |  |
| short-65536-hca-cp8r7 | 160 | 134 | 5 | 5 | 91 |  |  |  |  |  |  |  |
| short-65536-sliding-cp1 | 128 | 128 | 5 | 5 | 91 |  |  |  |  |  |  |  |
| short-65536-sliding-cp8r0 | 128 | 128 | 8 | 8 | 84 |  |  |  |  |  |  |  |
| short-65536-sliding-cp8r4 | 128 | 128 | 2 | 2 | 95 |  |  |  |  |  |  |  |
| short-65536-sliding-cp8r7 | 128 | 128 | 5 | 5 | 91 |  |  |  |  |  |  |  |
| heavy-65536-csa-cp1 | 640 | 542 | 6 | 6 | 12 | 9 | 6 | 4 | 3 | 2 | 2 | 49 |
| heavy-65536-csa-cp8r0 | 640 | 179 | 14 | 14 | 29 | 19 | 6 | 6 | 4 | 3 | 3 | 2 |
| heavy-65536-csa-cp8r4 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| heavy-65536-csa-cp8r7 | 640 | 286 | 9 | 9 | 17 | 10 | 11 | 8 | 6 | 3 | 3 | 25 |
| heavy-65536-hca-cp1 | 355 | 140 | 8 | 8 | 52 | 12 | 12 | 7 |  |  |  |  |
| heavy-65536-hca-cp8r0 | 355 | 129 | 17 | 17 | 66 |  |  |  |  |  |  |  |
| heavy-65536-hca-cp8r4 | 355 | 257 |  |  |  | 48 | 52 |  |  |  |  |  |
| heavy-65536-hca-cp8r7 | 355 | 132 | 11 | 11 | 78 |  |  |  |  |  |  |  |
| heavy-65536-sliding-cp1 | 128 | 128 | 8 | 8 | 84 |  |  |  |  |  |  |  |
| heavy-65536-sliding-cp8r0 | 128 | 128 | 17 | 17 | 66 |  |  |  |  |  |  |  |
| heavy-65536-sliding-cp8r4 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| heavy-65536-sliding-cp8r7 | 128 | 128 | 11 | 11 | 78 |  |  |  |  |  |  |  |
| tiny-65536-csa-cp1 | 640 | 47 | 63 | 33 | 4 |  |  |  |  |  |  |  |
| tiny-65536-csa-cp8r0 | 640 | 47 | 63 | 33 | 4 |  |  |  |  |  |  |  |
| tiny-65536-csa-cp8r4 | 640 | 46 | 64 | 32 | 5 |  |  |  |  |  |  |  |
| tiny-65536-csa-cp8r7 | 640 | 47 | 64 | 32 | 4 |  |  |  |  |  |  |  |
| tiny-65536-hca-cp1 | 128 | 38 | 74 | 26 |  |  |  |  |  |  |  |  |
| tiny-65536-hca-cp8r0 | 128 | 38 | 73 | 27 |  |  |  |  |  |  |  |  |
| tiny-65536-hca-cp8r4 | 128 | 37 | 74 | 26 |  |  |  |  |  |  |  |  |
| tiny-65536-hca-cp8r7 | 128 | 38 | 74 | 26 |  |  |  |  |  |  |  |  |
| tiny-65536-sliding-cp1 | 128 | 38 | 74 | 26 |  |  |  |  |  |  |  |  |
| tiny-65536-sliding-cp8r0 | 128 | 38 | 73 | 27 |  |  |  |  |  |  |  |  |
| tiny-65536-sliding-cp8r4 | 128 | 37 | 74 | 26 |  |  |  |  |  |  |  |  |
| tiny-65536-sliding-cp8r7 | 128 | 38 | 74 | 26 |  |  |  |  |  |  |  |  |
| stream00-short-14224-csa-cp8r1 | 640 | 256 | 6 | 6 | 17 | 21 | 8 |  |  | 14 | 14 | 14 |
| stream01-tiny-48096-hca-cp1 | 128 | 38 | 74 | 26 |  |  |  |  |  |  |  |  |
| stream02-heavy-38136-hca-cp8r6 | 173 | 129 | 17 | 17 | 65 |  |  |  |  |  |  |  |
| stream03-short-37152-sliding-cp8r1 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| stream04-short-34872-csa-cp8r5 | 640 | 291 | 5 | 5 | 14 | 19 | 10 | 6 | 6 | 6 | 6 | 24 |
| stream05-single-25016-csa-cp8r4 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| stream06-heavy-55096-csa-cp1 | 640 | 188 | 13 | 13 | 26 | 16 | 8 | 5 | 2 | 2 | 1 | 14 |
| stream07-short-17664-csa-cp8r2 | 640 | 266 | 5 | 5 | 14 | 23 | 23 | 18 | 13 |  |  |  |
| stream08-single-28000-hca-cp1 | 346 | 237 | 0 | 0 | 29 | 29 | 29 | 12 |  |  |  |  |
| stream09-tiny-33656-hca-cp1 | 128 | 38 | 74 | 26 |  |  |  |  |  |  |  |  |
| stream10-heavy-48264-csa-cp1 | 640 | 640 | 6 | 6 | 14 | 9 | 5 | 3 | 2 | 2 | 2 | 52 |
| stream11-tiny-12904-hca-cp1 | 128 | 35 | 78 | 22 |  |  |  |  |  |  |  |  |
| stream12-tiny-39552-sliding-cp1 | 128 | 38 | 73 | 27 |  |  |  |  |  |  |  |  |
| stream13-single-64144-csa-cp8r1 | 640 | 640 |  |  |  |  |  |  |  |  |  | 100 |
| stream14-heavy-15304-csa-cp1 | 640 | 238 | 10 | 10 | 20 | 13 | 6 | 5 | 5 | 5 | 5 | 20 |
| stream15-heavy-48400-sliding-cp1 | 128 | 128 | 8 | 8 | 84 |  |  |  |  |  |  |  |
| stream16-tiny-65360-hca-cp1 | 128 | 38 | 74 | 26 |  |  |  |  |  |  |  |  |
| stream17-heavy-4848-sliding-cp8r5 | 128 | 128 |  |  | 100 |  |  |  |  |  |  |  |
| stream18-short-24576-sliding-cp1 | 128 | 128 | 3 | 3 | 94 |  |  |  |  |  |  |  |
| stream19-single-36016-hca-cp1 | 409 | 268 | 0 | 0 | 22 | 23 | 23 | 23 | 9 |  |  |  |
| stream20-short-24768-hca-cp1 | 152 | 132 | 6 | 6 | 88 |  |  |  |  |  |  |  |
| stream21-heavy-39048-sliding-cp1 | 128 | 128 | 9 | 9 | 82 |  |  |  |  |  |  |  |
| stream22-single-60624-sliding-cp1 | 128 | 128 | 0 | 0 | 100 |  |  |  |  |  |  |  |
| stream23-single-61784-csa-cp1 | 640 | 640 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 97 |
| stream24-single-6944-hca-cp8r6 | 182 | 172 |  |  | 100 |  |  |  |  |  |  |  |
| stream25-single-56616-hca-cp1 | 570 | 349 | 0 | 0 | 14 | 14 | 14 | 14 | 14 | 14 | 13 |  |
| stream26-tiny-60696-sliding-cp8r2 | 128 | 38 | 75 | 25 |  |  |  |  |  |  |  |  |
| stream27-tiny-62600-sliding-cp8r4 | 128 | 39 | 74 | 26 |  |  |  |  |  |  |  |  |
| stream28-single-42136-csa-cp1 | 640 | 640 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 96 |
| stream29-tiny-22480-sliding-cp8r4 | 128 | 37 | 76 | 24 |  |  |  |  |  |  |  |  |
| stream30-heavy-38080-csa-cp1 | 640 | 207 | 11 | 11 | 23 | 14 | 8 | 7 | 6 | 4 | 3 | 14 |
| stream31-short-12824-hca-cp8r5 | 154 | 144 |  |  | 100 |  |  |  |  |  |  |  |

