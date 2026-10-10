# DSv4 sparse attention baseline: `main`

Caveat: the corpus is synthetic. Its CSA picks come from a random-weight Lightning Indexer and are
near-uniform over the readable entries, while a trained indexer favors recent and neighboring entries,
so CSA gather locality here is pessimistic. Sliding and HCA indices do not depend on weights.

- GPU: NVIDIA H200, driver 580.173.02, power limit 700.00 W,
  max SM clock 1980 MHz, host `prime-nebius-puku-h200-gpu-059`.
- Code: git `6d5cf0180`, torch 2.13.0+cu130, tilelang 0.1.12.
- Corpus hash `b5b289171983c8c6`, 240 grid items.
- Settings: 7 ABBA rounds x 10 calls, 3 warmup calls, L2 flushed and GPU idle before each call.

## Single-row items (cp1)

Op-boundary time per call in ms (lower is better). `gpu` is the GPU busy time of the same call; the
difference is host overhead. TFLOP/s counts useful FLOPs over op-boundary time (higher is better);
`% peak` is against 989.5 dense BF16 TFLOP/s. `ref/TL` is the FlashMLA
forward reference's op-boundary time over tilelang's forward (below 1 means the reference is faster).

| item | fwd | fwd gpu | f+b | f+b gpu | f+b TFLOP/s | % peak | ref/TL fwd |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | 1.20 | 0.54 | 3.19 | 2.16 | 112 | 11.3 | 0.32 |
| single-2048-hca-cp1 | 1.09 | 0.36 | 2.50 | 1.17 | 49 | 5.0 | 0.31 |
| single-2048-sliding-cp1 | 1.02 | 0.31 | 2.29 | 1.03 | 51 | 5.1 | 0.29 |
| short-2048-csa-cp1 | 1.13 | 0.44 | 2.76 | 1.64 | 85 | 8.5 | 0.33 |
| short-2048-hca-cp1 | 1.08 | 0.35 | 2.49 | 1.14 | 47 | 4.7 | 0.32 |
| short-2048-sliding-cp1 | 1.00 | 0.31 | 2.33 | 1.01 | 48 | 4.9 | 0.29 |
| heavy-2048-csa-cp1 | 1.17 | 0.48 | 2.95 | 1.82 | 92 | 9.3 | 0.33 |
| heavy-2048-hca-cp1 | 1.07 | 0.34 | 2.46 | 1.11 | 46 | 4.7 | 0.31 |
| heavy-2048-sliding-cp1 | 1.00 | 0.30 | 2.29 | 1.00 | 48 | 4.8 | 0.29 |
| tiny-2048-csa-cp1 | 1.04 | 0.36 | 2.36 | 1.10 | 23 | 2.3 | 0.32 |
| tiny-2048-hca-cp1 | 0.96 | 0.27 | 2.09 | 0.77 | 21 | 2.1 | 0.30 |
| tiny-2048-sliding-cp1 | 0.96 | 0.27 | 2.11 | 0.77 | 20 | 2.1 | 0.30 |
| single-4096-csa-cp1 | 1.89 | 1.21 | 5.97 | 5.15 | 160 | 16.2 | 0.38 |
| single-4096-hca-cp1 | 1.43 | 0.69 | 3.13 | 2.23 | 85 | 8.6 | 0.35 |
| single-4096-sliding-cp1 | 1.30 | 0.59 | 2.86 | 1.95 | 83 | 8.4 | 0.32 |
| short-4096-csa-cp1 | 1.52 | 0.83 | 3.87 | 3.05 | 116 | 11.7 | 0.36 |
| short-4096-hca-cp1 | 1.41 | 0.67 | 3.09 | 2.13 | 74 | 7.5 | 0.35 |
| short-4096-sliding-cp1 | 1.31 | 0.58 | 2.84 | 1.90 | 78 | 7.9 | 0.32 |
| heavy-4096-csa-cp1 | 1.46 | 0.77 | 3.55 | 2.72 | 101 | 10.2 | 0.36 |
| heavy-4096-hca-cp1 | 1.37 | 0.65 | 2.98 | 2.03 | 69 | 7.0 | 0.35 |
| heavy-4096-sliding-cp1 | 1.28 | 0.58 | 2.91 | 1.84 | 70 | 7.1 | 0.33 |
| tiny-4096-csa-cp1 | 1.37 | 0.68 | 2.92 | 2.07 | 36 | 3.6 | 0.37 |
| tiny-4096-hca-cp1 | 1.23 | 0.53 | 2.41 | 1.44 | 35 | 3.5 | 0.34 |
| tiny-4096-sliding-cp1 | 1.23 | 0.53 | 2.41 | 1.44 | 35 | 3.5 | 0.33 |
| single-16384-csa-cp1 | 5.92 | 5.36 | 23.56 | 22.67 | 194 | 19.6 | 0.43 |
| single-16384-hca-cp1 | 3.57 | 2.87 | 10.90 | 10.01 | 132 | 13.3 | 0.42 |
| single-16384-sliding-cp1 | 2.97 | 2.29 | 8.29 | 7.47 | 116 | 11.7 | 0.39 |
| short-16384-csa-cp1 | 4.51 | 3.85 | 15.98 | 15.07 | 163 | 16.5 | 0.43 |
| short-16384-hca-cp1 | 3.34 | 2.60 | 9.22 | 8.30 | 104 | 10.6 | 0.44 |
| short-16384-sliding-cp1 | 2.99 | 2.26 | 8.16 | 7.32 | 112 | 11.3 | 0.39 |
| heavy-16384-csa-cp1 | 3.97 | 3.30 | 13.00 | 12.08 | 138 | 14.0 | 0.43 |
| heavy-16384-hca-cp1 | 3.25 | 2.51 | 8.71 | 7.78 | 97 | 9.8 | 0.43 |
| heavy-16384-sliding-cp1 | 2.98 | 2.25 | 7.89 | 7.05 | 104 | 10.5 | 0.39 |
| tiny-16384-csa-cp1 | 3.29 | 2.64 | 8.73 | 7.89 | 45 | 4.5 | 0.45 |
| tiny-16384-hca-cp1 | 2.71 | 2.02 | 6.18 | 5.36 | 51 | 5.2 | 0.43 |
| tiny-16384-sliding-cp1 | 2.73 | 2.03 | 6.16 | 5.35 | 51 | 5.2 | 0.43 |
| single-49208-csa-cp1 | 16.42 | 16.24 | 70.71 | 69.99 | 201 | 20.3 | 0.51 |
| single-49208-hca-cp1 | 11.50 | 11.19 | 42.56 | 41.81 | 170 | 17.1 | 0.47 |
| single-49208-sliding-cp1 | 7.43 | 6.77 | 22.89 | 22.04 | 126 | 12.7 | 0.42 |
| short-49208-csa-cp1 | 12.99 | 12.69 | 51.07 | 50.33 | 181 | 18.3 | 0.48 |
| short-49208-hca-cp1 | 8.50 | 7.88 | 26.17 | 25.34 | 120 | 12.1 | 0.48 |
| short-49208-sliding-cp1 | 7.42 | 6.74 | 22.63 | 21.76 | 123 | 12.4 | 0.42 |
| heavy-49208-csa-cp1 | 14.91 | 14.88 | 61.71 | 60.98 | 194 | 19.6 | 0.50 |
| heavy-49208-hca-cp1 | 9.09 | 8.49 | 29.60 | 28.84 | 138 | 13.9 | 0.47 |
| heavy-49208-sliding-cp1 | 7.45 | 6.75 | 22.58 | 21.73 | 123 | 12.4 | 0.42 |
| tiny-49208-csa-cp1 | 8.39 | 7.84 | 24.10 | 23.41 | 50 | 5.0 | 0.51 |
| tiny-49208-hca-cp1 | 6.74 | 6.06 | 16.79 | 15.95 | 58 | 5.8 | 0.47 |
| tiny-49208-sliding-cp1 | 6.73 | 6.06 | 16.80 | 15.95 | 58 | 5.8 | 0.47 |
| single-65536-csa-cp1 | 21.69 | 21.80 | 95.44 | 94.80 | 199 | 20.1 | 0.52 |
| single-65536-hca-cp1 | 16.50 | 16.32 | 64.33 | 63.70 | 179 | 18.1 | 0.50 |
| single-65536-sliding-cp1 | 9.65 | 9.00 | 30.15 | 29.31 | 128 | 12.9 | 0.43 |
| short-65536-csa-cp1 | 16.24 | 16.00 | 63.17 | 62.51 | 177 | 17.9 | 0.50 |
| short-65536-hca-cp1 | 10.92 | 10.31 | 33.53 | 32.76 | 117 | 11.8 | 0.49 |
| short-65536-sliding-cp1 | 9.61 | 8.97 | 29.64 | 28.84 | 124 | 12.5 | 0.43 |
| heavy-65536-csa-cp1 | 17.37 | 17.41 | 69.70 | 69.04 | 183 | 18.5 | 0.51 |
| heavy-65536-hca-cp1 | 11.71 | 11.20 | 38.27 | 37.59 | 134 | 13.6 | 0.49 |
| heavy-65536-sliding-cp1 | 9.62 | 8.94 | 29.28 | 28.44 | 121 | 12.2 | 0.43 |
| tiny-65536-csa-cp1 | 10.88 | 10.41 | 31.80 | 31.18 | 51 | 5.2 | 0.52 |
| tiny-65536-hca-cp1 | 8.69 | 8.04 | 22.02 | 21.19 | 59 | 6.0 | 0.48 |
| tiny-65536-sliding-cp1 | 8.72 | 8.03 | 22.04 | 21.19 | 59 | 6.0 | 0.48 |

## Host overhead (op-boundary minus GPU busy time), µs

| arm | mode | min | median | max | items where it exceeds GPU time |
|---|---|---|---|---|---|
| flashmla_fwd_ref | fwd | -31 | 140 | 1033 | 77 of 240 |
| tilelang | fwd | -110 | 692 | 762 | 127 of 240 |
| tilelang | fwd_bwd | 622 | 971 | 1973 | 106 of 240 |

## Correctness gate

Largest relative error over all items (max deviation over the reference's max magnitude), and its bound.

| arm | basis | tensor | max relative error | bound | failures |
|---|---|---|---|---|---|
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
| tilelang | cold | 20.87 | 0.614 | 4 | 0 | 42.6 | 16.0 |
| tilelang | warm | 0.29 | 0.614 | 0 | 4 | 42.1 | 0.0 |

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

