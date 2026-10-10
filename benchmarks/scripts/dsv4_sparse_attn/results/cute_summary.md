# DSv4 sparse attention baseline: `cute`

Caveat: the corpus is synthetic. Its CSA picks come from a random-weight Lightning Indexer and are
near-uniform over the readable entries, while a trained indexer favors recent and neighboring entries,
so CSA gather locality here is pessimistic. Sliding and HCA indices do not depend on weights.

- GPU: NVIDIA H200, driver 580.173.02, power limit 700.00 W,
  max SM clock 1980 MHz, host `prime-nebius-puku-h200-gpu-059`.
- Code: git `8af8d58fb` (dirty), torch 2.13.0+cu130, tilelang 0.1.12.
- Corpus hash `b5b289171983c8c6`, 240 grid items.
- Settings: 7 ABBA rounds x 10 calls, 3 warmup calls, L2 flushed and GPU idle before each call.

## Single-row items (cp1)

Op-boundary time per call in ms (lower is better). `gpu` is the GPU busy time of the same call; the
difference is host overhead. TFLOP/s counts useful FLOPs over op-boundary time (higher is better);
`% peak` is against 989.5 dense BF16 TFLOP/s. `ref/TL` is the FlashMLA
forward reference's op-boundary time over tilelang's forward (below 1 means the reference is faster).

| item | fwd | fwd gpu | f+b | f+b gpu | f+b TFLOP/s | % peak | ref/TL fwd |
|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | 1.21 | 0.54 | 3.26 | 2.16 | 109 | 11.1 | 0.32 |
| single-2048-hca-cp1 | 1.09 | 0.36 | 2.56 | 1.17 | 48 | 4.9 | 0.32 |
| single-2048-sliding-cp1 | 0.99 | 0.31 | 2.36 | 1.03 | 49 | 5.0 | 0.31 |
| short-2048-csa-cp1 | 1.12 | 0.44 | 2.84 | 1.64 | 82 | 8.3 | 0.33 |
| short-2048-hca-cp1 | 1.08 | 0.35 | 2.52 | 1.14 | 46 | 4.6 | 0.32 |
| short-2048-sliding-cp1 | 0.99 | 0.31 | 2.32 | 1.01 | 49 | 4.9 | 0.30 |
| heavy-2048-csa-cp1 | 1.17 | 0.48 | 3.02 | 1.82 | 90 | 9.1 | 0.34 |
| heavy-2048-hca-cp1 | 1.07 | 0.34 | 2.50 | 1.11 | 46 | 4.6 | 0.32 |
| heavy-2048-sliding-cp1 | 1.00 | 0.30 | 2.35 | 1.00 | 46 | 4.7 | 0.30 |
| tiny-2048-csa-cp1 | 1.04 | 0.36 | 2.40 | 1.10 | 22 | 2.3 | 0.33 |
| tiny-2048-hca-cp1 | 0.96 | 0.27 | 2.18 | 0.77 | 20 | 2.0 | 0.31 |
| tiny-2048-sliding-cp1 | 0.95 | 0.27 | 2.14 | 0.77 | 20 | 2.0 | 0.31 |
| single-4096-csa-cp1 | 1.89 | 1.22 | 6.02 | 5.15 | 159 | 16.1 | 0.38 |
| single-4096-hca-cp1 | 1.42 | 0.69 | 3.16 | 2.24 | 84 | 8.5 | 0.36 |
| single-4096-sliding-cp1 | 1.28 | 0.59 | 2.89 | 1.95 | 82 | 8.3 | 0.33 |
| short-4096-csa-cp1 | 1.51 | 0.83 | 3.89 | 3.05 | 116 | 11.7 | 0.36 |
| short-4096-hca-cp1 | 1.39 | 0.66 | 3.08 | 2.13 | 74 | 7.5 | 0.36 |
| short-4096-sliding-cp1 | 1.28 | 0.58 | 2.89 | 1.90 | 77 | 7.8 | 0.33 |
| heavy-4096-csa-cp1 | 1.47 | 0.77 | 3.55 | 2.72 | 100 | 10.1 | 0.36 |
| heavy-4096-hca-cp1 | 1.38 | 0.65 | 3.02 | 2.03 | 69 | 6.9 | 0.36 |
| heavy-4096-sliding-cp1 | 1.31 | 0.59 | 2.82 | 1.84 | 72 | 7.3 | 0.33 |
| tiny-4096-csa-cp1 | 1.37 | 0.68 | 2.92 | 2.07 | 36 | 3.6 | 0.36 |
| tiny-4096-hca-cp1 | 1.23 | 0.53 | 2.46 | 1.43 | 34 | 3.5 | 0.34 |
| tiny-4096-sliding-cp1 | 1.23 | 0.53 | 2.47 | 1.44 | 34 | 3.5 | 0.35 |
| single-16384-csa-cp1 | 5.88 | 5.79 | 23.58 | 22.69 | 194 | 19.6 | 0.45 |
| single-16384-hca-cp1 | 3.54 | 2.89 | 10.87 | 9.97 | 132 | 13.4 | 0.42 |
| single-16384-sliding-cp1 | 2.98 | 2.29 | 8.29 | 7.44 | 116 | 11.7 | 0.39 |
| short-16384-csa-cp1 | 4.50 | 3.91 | 15.91 | 14.98 | 164 | 16.6 | 0.44 |
| short-16384-hca-cp1 | 3.33 | 2.61 | 9.23 | 8.24 | 104 | 10.6 | 0.44 |
| short-16384-sliding-cp1 | 2.96 | 2.27 | 8.16 | 7.29 | 112 | 11.3 | 0.40 |
| heavy-16384-csa-cp1 | 3.96 | 3.32 | 12.93 | 12.01 | 139 | 14.0 | 0.44 |
| heavy-16384-hca-cp1 | 3.22 | 2.49 | 8.69 | 7.75 | 97 | 9.9 | 0.44 |
| heavy-16384-sliding-cp1 | 2.94 | 2.25 | 7.89 | 7.03 | 104 | 10.5 | 0.40 |
| tiny-16384-csa-cp1 | 3.32 | 2.64 | 8.71 | 7.89 | 45 | 4.5 | 0.45 |
| tiny-16384-hca-cp1 | 2.74 | 2.02 | 6.19 | 5.35 | 51 | 5.1 | 0.44 |
| tiny-16384-sliding-cp1 | 2.73 | 2.03 | 6.22 | 5.34 | 51 | 5.1 | 0.43 |
| single-49208-csa-cp1 | 16.39 | 16.38 | 70.85 | 70.01 | 200 | 20.3 | 0.53 |
| single-49208-hca-cp1 | 11.50 | 11.07 | 42.56 | 41.79 | 170 | 17.1 | 0.49 |
| single-49208-sliding-cp1 | 7.47 | 6.81 | 22.95 | 22.06 | 126 | 12.7 | 0.43 |
| short-49208-csa-cp1 | 12.99 | 12.98 | 51.11 | 50.33 | 181 | 18.3 | 0.51 |
| short-49208-hca-cp1 | 8.56 | 7.87 | 26.19 | 25.37 | 120 | 12.1 | 0.48 |
| short-49208-sliding-cp1 | 7.46 | 6.82 | 22.66 | 21.77 | 123 | 12.4 | 0.42 |
| heavy-49208-csa-cp1 | 14.86 | 14.87 | 61.74 | 60.93 | 194 | 19.6 | 0.52 |
| heavy-49208-hca-cp1 | 9.10 | 8.58 | 29.68 | 28.87 | 137 | 13.9 | 0.48 |
| heavy-49208-sliding-cp1 | 7.46 | 6.86 | 22.63 | 21.73 | 123 | 12.4 | 0.42 |
| tiny-49208-csa-cp1 | 8.40 | 7.84 | 24.15 | 23.41 | 50 | 5.0 | 0.51 |
| tiny-49208-hca-cp1 | 6.72 | 6.04 | 16.76 | 15.90 | 58 | 5.8 | 0.47 |
| tiny-49208-sliding-cp1 | 6.73 | 6.04 | 16.76 | 15.88 | 58 | 5.8 | 0.47 |
| single-65536-csa-cp1 | 21.60 | 21.73 | 95.48 | 94.99 | 199 | 20.1 | 0.53 |
| single-65536-hca-cp1 | 16.49 | 16.22 | 64.33 | 63.65 | 179 | 18.1 | 0.51 |
| single-65536-sliding-cp1 | 9.69 | 9.05 | 30.17 | 29.34 | 127 | 12.9 | 0.43 |
| short-65536-csa-cp1 | 16.23 | 15.99 | 63.18 | 62.50 | 177 | 17.9 | 0.51 |
| short-65536-hca-cp1 | 11.01 | 10.36 | 33.57 | 32.75 | 117 | 11.8 | 0.49 |
| short-65536-sliding-cp1 | 9.68 | 9.02 | 29.72 | 28.84 | 124 | 12.5 | 0.43 |
| heavy-65536-csa-cp1 | 17.40 | 17.38 | 69.67 | 68.97 | 183 | 18.5 | 0.52 |
| heavy-65536-hca-cp1 | 11.74 | 11.25 | 38.30 | 37.62 | 134 | 13.6 | 0.49 |
| heavy-65536-sliding-cp1 | 9.61 | 9.08 | 29.28 | 28.46 | 121 | 12.2 | 0.43 |
| tiny-65536-csa-cp1 | 10.91 | 10.43 | 31.85 | 31.18 | 51 | 5.1 | 0.52 |
| tiny-65536-hca-cp1 | 8.69 | 8.04 | 22.03 | 21.19 | 59 | 6.0 | 0.48 |
| tiny-65536-sliding-cp1 | 8.69 | 8.03 | 22.03 | 21.19 | 59 | 6.0 | 0.48 |

## Every arm on the single-row items (cp1)

Time per call in ms (lower is better): op-boundary, then GPU busy time. TFLOP/s counts useful FLOPs (valid
slots only) over op-boundary time (higher is better). `exec/useful` is the FLOPs of the slots the arm's
tiles touch over useful FLOPs (1 is no wasted work). `-` marks a mode the arm does not have.

| item | arm | fwd | fwd gpu | fwd TFLOP/s | exec/useful fwd | f+b | f+b gpu | f+b TFLOP/s |
|---|---|---|---|---|---|---|---|---|
| single-2048-csa-cp1 | tilelang | 1.21 | 0.54 | 84 | 1.09 | 3.26 | 2.16 | 109 |
| single-2048-csa-cp1 | cute | 0.61 | 0.51 | 167 | 1.09 | 2.63 | 2.13 | 136 |
| single-2048-csa-cp1 | cute_ws | 0.30 | 0.26 | 336 | 1.18 | 2.47 | 1.88 | 145 |
| single-2048-csa-cp1 | flashmla_fwd_ref | 0.39 | 0.28 | 262 | 1.69 | - | - | - |
| single-2048-hca-cp1 | tilelang | 1.09 | 0.36 | 32 | 1.41 | 2.56 | 1.17 | 48 |
| single-2048-hca-cp1 | cute | 0.48 | 0.34 | 73 | 1.41 | 1.91 | 1.14 | 65 |
| single-2048-hca-cp1 | cute_ws | 0.22 | 0.17 | 158 | 1.89 | 1.72 | 0.98 | 72 |
| single-2048-hca-cp1 | flashmla_fwd_ref | 0.35 | 0.20 | 100 | 1.95 | - | - | - |
| single-2048-sliding-cp1 | tilelang | 0.99 | 0.31 | 34 | 1.02 | 2.36 | 1.03 | 49 |
| single-2048-sliding-cp1 | cute | 0.42 | 0.29 | 80 | 1.02 | 1.77 | 1.01 | 66 |
| single-2048-sliding-cp1 | cute_ws | 0.17 | 0.12 | 193 | 1.03 | 1.58 | 0.84 | 74 |
| single-2048-sliding-cp1 | flashmla_fwd_ref | 0.30 | 0.15 | 110 | 1.03 | - | - | - |
| short-2048-csa-cp1 | tilelang | 1.12 | 0.44 | 59 | 1.16 | 2.84 | 1.64 | 82 |
| short-2048-csa-cp1 | cute | 0.53 | 0.42 | 124 | 1.16 | 2.19 | 1.61 | 106 |
| short-2048-csa-cp1 | cute_ws | 0.26 | 0.21 | 256 | 1.30 | 2.06 | 1.41 | 113 |
| short-2048-csa-cp1 | flashmla_fwd_ref | 0.38 | 0.23 | 177 | 2.58 | - | - | - |
| short-2048-hca-cp1 | tilelang | 1.08 | 0.35 | 31 | 1.46 | 2.52 | 1.14 | 46 |
| short-2048-hca-cp1 | cute | 0.48 | 0.33 | 70 | 1.46 | 1.88 | 1.12 | 62 |
| short-2048-hca-cp1 | cute_ws | 0.22 | 0.17 | 151 | 1.94 | 1.68 | 0.96 | 69 |
| short-2048-hca-cp1 | flashmla_fwd_ref | 0.35 | 0.19 | 96 | 2.07 | - | - | - |
| short-2048-sliding-cp1 | tilelang | 0.99 | 0.31 | 33 | 1.03 | 2.32 | 1.01 | 49 |
| short-2048-sliding-cp1 | cute | 0.41 | 0.28 | 79 | 1.03 | 1.71 | 0.98 | 66 |
| short-2048-sliding-cp1 | cute_ws | 0.17 | 0.12 | 189 | 1.07 | 1.54 | 0.83 | 73 |
| short-2048-sliding-cp1 | flashmla_fwd_ref | 0.30 | 0.15 | 108 | 1.07 | - | - | - |
| heavy-2048-csa-cp1 | tilelang | 1.17 | 0.48 | 67 | 1.16 | 3.02 | 1.82 | 90 |
| heavy-2048-csa-cp1 | cute | 0.57 | 0.45 | 136 | 1.16 | 2.38 | 1.79 | 115 |
| heavy-2048-csa-cp1 | cute_ws | 0.28 | 0.23 | 278 | 1.29 | 2.23 | 1.58 | 122 |
| heavy-2048-csa-cp1 | flashmla_fwd_ref | 0.39 | 0.26 | 197 | 2.21 | - | - | - |
| heavy-2048-hca-cp1 | tilelang | 1.07 | 0.34 | 30 | 1.44 | 2.50 | 1.11 | 46 |
| heavy-2048-hca-cp1 | cute | 0.47 | 0.32 | 69 | 1.44 | 1.86 | 1.09 | 61 |
| heavy-2048-hca-cp1 | cute_ws | 0.21 | 0.16 | 153 | 1.92 | 1.67 | 0.93 | 68 |
| heavy-2048-hca-cp1 | flashmla_fwd_ref | 0.34 | 0.19 | 95 | 2.11 | - | - | - |
| heavy-2048-sliding-cp1 | tilelang | 1.00 | 0.30 | 31 | 1.05 | 2.35 | 1.00 | 46 |
| heavy-2048-sliding-cp1 | cute | 0.41 | 0.28 | 77 | 1.05 | 1.71 | 0.97 | 64 |
| heavy-2048-sliding-cp1 | cute_ws | 0.17 | 0.12 | 184 | 1.10 | 1.56 | 0.82 | 70 |
| heavy-2048-sliding-cp1 | flashmla_fwd_ref | 0.30 | 0.15 | 105 | 1.10 | - | - | - |
| tiny-2048-csa-cp1 | tilelang | 1.04 | 0.36 | 15 | 3.28 | 2.40 | 1.10 | 22 |
| tiny-2048-csa-cp1 | cute | 0.45 | 0.34 | 34 | 3.28 | 1.75 | 1.08 | 31 |
| tiny-2048-csa-cp1 | cute_ws | 0.23 | 0.18 | 67 | 4.40 | 1.61 | 0.92 | 33 |
| tiny-2048-csa-cp1 | flashmla_fwd_ref | 0.34 | 0.20 | 45 | 11.22 | - | - | - |
| tiny-2048-hca-cp1 | tilelang | 0.96 | 0.27 | 13 | 1.79 | 2.18 | 0.77 | 20 |
| tiny-2048-hca-cp1 | cute | 0.37 | 0.25 | 33 | 1.79 | 1.52 | 0.75 | 28 |
| tiny-2048-hca-cp1 | cute_ws | 0.17 | 0.12 | 73 | 2.79 | 1.37 | 0.63 | 32 |
| tiny-2048-hca-cp1 | flashmla_fwd_ref | 0.30 | 0.15 | 41 | 2.79 | - | - | - |
| tiny-2048-sliding-cp1 | tilelang | 0.95 | 0.27 | 13 | 1.79 | 2.14 | 0.77 | 20 |
| tiny-2048-sliding-cp1 | cute | 0.38 | 0.25 | 33 | 1.79 | 1.49 | 0.75 | 29 |
| tiny-2048-sliding-cp1 | cute_ws | 0.17 | 0.12 | 73 | 2.79 | 1.36 | 0.63 | 32 |
| tiny-2048-sliding-cp1 | flashmla_fwd_ref | 0.30 | 0.15 | 41 | 2.79 | - | - | - |
| single-4096-csa-cp1 | tilelang | 1.89 | 1.22 | 145 | 1.03 | 6.02 | 5.15 | 159 |
| single-4096-csa-cp1 | cute | 1.27 | 1.16 | 215 | 1.03 | 5.34 | 5.09 | 180 |
| single-4096-csa-cp1 | cute_ws | 0.61 | 0.55 | 452 | 1.07 | 4.69 | 4.49 | 204 |
| single-4096-csa-cp1 | flashmla_fwd_ref | 0.72 | 0.60 | 378 | 1.26 | - | - | - |
| single-4096-hca-cp1 | tilelang | 1.42 | 0.69 | 54 | 1.34 | 3.16 | 2.24 | 84 |
| single-4096-hca-cp1 | cute | 0.79 | 0.64 | 96 | 1.34 | 2.52 | 2.18 | 105 |
| single-4096-hca-cp1 | cute_ws | 0.37 | 0.32 | 204 | 1.78 | 2.33 | 1.87 | 114 |
| single-4096-hca-cp1 | flashmla_fwd_ref | 0.51 | 0.36 | 148 | 1.81 | - | - | - |
| single-4096-sliding-cp1 | tilelang | 1.28 | 0.59 | 53 | 1.01 | 2.89 | 1.95 | 82 |
| single-4096-sliding-cp1 | cute | 0.68 | 0.55 | 100 | 1.01 | 2.26 | 1.90 | 105 |
| single-4096-sliding-cp1 | cute_ws | 0.27 | 0.22 | 248 | 1.02 | 2.12 | 1.59 | 112 |
| single-4096-sliding-cp1 | flashmla_fwd_ref | 0.42 | 0.28 | 159 | 1.02 | - | - | - |
| short-4096-csa-cp1 | tilelang | 1.51 | 0.83 | 85 | 1.18 | 3.89 | 3.05 | 116 |
| short-4096-csa-cp1 | cute | 0.90 | 0.78 | 142 | 1.18 | 3.23 | 3.00 | 139 |
| short-4096-csa-cp1 | cute_ws | 0.43 | 0.37 | 300 | 1.33 | 2.96 | 2.61 | 152 |
| short-4096-csa-cp1 | flashmla_fwd_ref | 0.55 | 0.43 | 234 | 2.67 | - | - | - |
| short-4096-hca-cp1 | tilelang | 1.39 | 0.66 | 47 | 1.46 | 3.08 | 2.13 | 74 |
| short-4096-hca-cp1 | cute | 0.76 | 0.62 | 85 | 1.46 | 2.46 | 2.09 | 93 |
| short-4096-hca-cp1 | cute_ws | 0.35 | 0.30 | 184 | 1.95 | 2.27 | 1.78 | 101 |
| short-4096-hca-cp1 | flashmla_fwd_ref | 0.50 | 0.36 | 130 | 2.11 | - | - | - |
| short-4096-sliding-cp1 | tilelang | 1.28 | 0.58 | 49 | 1.04 | 2.89 | 1.90 | 77 |
| short-4096-sliding-cp1 | cute | 0.66 | 0.54 | 96 | 1.04 | 2.22 | 1.85 | 100 |
| short-4096-sliding-cp1 | cute_ws | 0.27 | 0.22 | 237 | 1.08 | 2.07 | 1.53 | 107 |
| short-4096-sliding-cp1 | flashmla_fwd_ref | 0.43 | 0.27 | 149 | 1.08 | - | - | - |
| heavy-4096-csa-cp1 | tilelang | 1.47 | 0.77 | 69 | 1.29 | 3.55 | 2.72 | 100 |
| heavy-4096-csa-cp1 | cute | 0.84 | 0.72 | 121 | 1.29 | 2.90 | 2.67 | 123 |
| heavy-4096-csa-cp1 | cute_ws | 0.41 | 0.36 | 250 | 1.52 | 2.69 | 2.30 | 133 |
| heavy-4096-csa-cp1 | flashmla_fwd_ref | 0.53 | 0.41 | 192 | 3.37 | - | - | - |
| heavy-4096-hca-cp1 | tilelang | 1.38 | 0.65 | 43 | 1.47 | 3.02 | 2.03 | 69 |
| heavy-4096-hca-cp1 | cute | 0.75 | 0.60 | 79 | 1.47 | 2.37 | 1.98 | 87 |
| heavy-4096-hca-cp1 | cute_ws | 0.34 | 0.29 | 174 | 1.96 | 2.18 | 1.67 | 95 |
| heavy-4096-hca-cp1 | flashmla_fwd_ref | 0.49 | 0.34 | 120 | 2.32 | - | - | - |
| heavy-4096-sliding-cp1 | tilelang | 1.31 | 0.59 | 44 | 1.09 | 2.82 | 1.84 | 72 |
| heavy-4096-sliding-cp1 | cute | 0.67 | 0.54 | 86 | 1.09 | 2.16 | 1.80 | 94 |
| heavy-4096-sliding-cp1 | cute_ws | 0.27 | 0.22 | 212 | 1.18 | 2.01 | 1.48 | 101 |
| heavy-4096-sliding-cp1 | flashmla_fwd_ref | 0.43 | 0.28 | 135 | 1.18 | - | - | - |
| tiny-4096-csa-cp1 | tilelang | 1.37 | 0.68 | 22 | 3.35 | 2.92 | 2.07 | 36 |
| tiny-4096-csa-cp1 | cute | 0.75 | 0.64 | 40 | 3.35 | 2.30 | 2.03 | 45 |
| tiny-4096-csa-cp1 | cute_ws | 0.38 | 0.33 | 78 | 4.50 | 2.14 | 1.72 | 49 |
| tiny-4096-csa-cp1 | flashmla_fwd_ref | 0.50 | 0.38 | 60 | 11.49 | - | - | - |
| tiny-4096-hca-cp1 | tilelang | 1.23 | 0.53 | 20 | 1.81 | 2.46 | 1.43 | 34 |
| tiny-4096-hca-cp1 | cute | 0.61 | 0.49 | 39 | 1.81 | 1.81 | 1.39 | 46 |
| tiny-4096-hca-cp1 | cute_ws | 0.29 | 0.23 | 83 | 2.85 | 1.68 | 1.15 | 50 |
| tiny-4096-hca-cp1 | flashmla_fwd_ref | 0.42 | 0.27 | 57 | 2.85 | - | - | - |
| tiny-4096-sliding-cp1 | tilelang | 1.23 | 0.53 | 20 | 1.81 | 2.47 | 1.44 | 34 |
| tiny-4096-sliding-cp1 | cute | 0.62 | 0.49 | 39 | 1.81 | 1.83 | 1.39 | 46 |
| tiny-4096-sliding-cp1 | cute_ws | 0.29 | 0.23 | 84 | 2.85 | 1.67 | 1.15 | 50 |
| tiny-4096-sliding-cp1 | flashmla_fwd_ref | 0.43 | 0.28 | 56 | 2.85 | - | - | - |
| single-16384-csa-cp1 | tilelang | 5.88 | 5.79 | 222 | 1.01 | 23.58 | 22.69 | 194 |
| single-16384-csa-cp1 | cute | 5.10 | 5.03 | 256 | 1.01 | 22.72 | 22.43 | 201 |
| single-16384-csa-cp1 | cute_ws | 2.44 | 2.37 | 534 | 1.01 | 20.08 | 19.87 | 227 |
| single-16384-csa-cp1 | flashmla_fwd_ref | 2.62 | 2.53 | 498 | 1.05 | - | - | - |
| single-16384-hca-cp1 | tilelang | 3.54 | 2.89 | 116 | 1.17 | 10.87 | 9.97 | 132 |
| single-16384-hca-cp1 | cute | 2.79 | 2.70 | 147 | 1.17 | 10.05 | 9.76 | 143 |
| single-16384-hca-cp1 | cute_ws | 1.26 | 1.21 | 325 | 1.34 | 8.42 | 8.29 | 170 |
| single-16384-hca-cp1 | flashmla_fwd_ref | 1.49 | 1.40 | 275 | 1.34 | - | - | - |
| single-16384-sliding-cp1 | tilelang | 2.98 | 2.29 | 92 | 1.00 | 8.29 | 7.44 | 116 |
| single-16384-sliding-cp1 | cute | 2.22 | 2.11 | 123 | 1.00 | 7.50 | 7.26 | 128 |
| single-16384-sliding-cp1 | cute_ws | 0.85 | 0.80 | 321 | 1.00 | 6.08 | 5.95 | 158 |
| single-16384-sliding-cp1 | flashmla_fwd_ref | 1.17 | 1.05 | 234 | 1.00 | - | - | - |
| short-16384-csa-cp1 | tilelang | 4.50 | 3.91 | 166 | 1.11 | 15.91 | 14.98 | 164 |
| short-16384-csa-cp1 | cute | 3.71 | 3.68 | 201 | 1.11 | 15.08 | 14.78 | 173 |
| short-16384-csa-cp1 | cute_ws | 1.78 | 1.73 | 418 | 1.20 | 13.12 | 12.93 | 199 |
| short-16384-csa-cp1 | flashmla_fwd_ref | 1.98 | 1.92 | 377 | 1.84 | - | - | - |
| short-16384-hca-cp1 | tilelang | 3.33 | 2.61 | 83 | 1.42 | 9.23 | 8.24 | 104 |
| short-16384-hca-cp1 | cute | 2.56 | 2.43 | 108 | 1.42 | 8.37 | 8.06 | 115 |
| short-16384-hca-cp1 | cute_ws | 1.24 | 1.19 | 222 | 1.90 | 6.99 | 6.82 | 138 |
| short-16384-hca-cp1 | flashmla_fwd_ref | 1.46 | 1.37 | 188 | 2.00 | - | - | - |
| short-16384-sliding-cp1 | tilelang | 2.96 | 2.27 | 88 | 1.03 | 8.16 | 7.29 | 112 |
| short-16384-sliding-cp1 | cute | 2.21 | 2.10 | 118 | 1.03 | 7.36 | 7.13 | 124 |
| short-16384-sliding-cp1 | cute_ws | 0.85 | 0.79 | 306 | 1.05 | 5.96 | 5.83 | 153 |
| short-16384-sliding-cp1 | flashmla_fwd_ref | 1.17 | 1.04 | 223 | 1.05 | - | - | - |
| heavy-16384-csa-cp1 | tilelang | 3.96 | 3.32 | 130 | 1.22 | 12.93 | 12.01 | 139 |
| heavy-16384-csa-cp1 | cute | 3.19 | 3.12 | 161 | 1.22 | 12.11 | 11.83 | 148 |
| heavy-16384-csa-cp1 | cute_ws | 1.56 | 1.50 | 329 | 1.40 | 10.43 | 10.22 | 172 |
| heavy-16384-csa-cp1 | flashmla_fwd_ref | 1.74 | 1.69 | 296 | 2.68 | - | - | - |
| heavy-16384-hca-cp1 | tilelang | 3.22 | 2.49 | 75 | 1.45 | 8.69 | 7.75 | 97 |
| heavy-16384-hca-cp1 | cute | 2.45 | 2.32 | 99 | 1.45 | 7.85 | 7.57 | 108 |
| heavy-16384-hca-cp1 | cute_ws | 1.16 | 1.10 | 208 | 1.94 | 6.50 | 6.37 | 130 |
| heavy-16384-hca-cp1 | flashmla_fwd_ref | 1.41 | 1.30 | 172 | 2.27 | - | - | - |
| heavy-16384-sliding-cp1 | tilelang | 2.94 | 2.25 | 80 | 1.09 | 7.89 | 7.03 | 104 |
| heavy-16384-sliding-cp1 | cute | 2.20 | 2.08 | 107 | 1.09 | 7.10 | 6.84 | 116 |
| heavy-16384-sliding-cp1 | cute_ws | 0.87 | 0.80 | 270 | 1.17 | 5.73 | 5.59 | 143 |
| heavy-16384-sliding-cp1 | flashmla_fwd_ref | 1.18 | 1.05 | 199 | 1.17 | - | - | - |
| tiny-16384-csa-cp1 | tilelang | 3.32 | 2.64 | 34 | 3.56 | 8.71 | 7.89 | 45 |
| tiny-16384-csa-cp1 | cute | 2.54 | 2.47 | 44 | 3.56 | 7.90 | 7.73 | 50 |
| tiny-16384-csa-cp1 | cute_ws | 1.30 | 1.24 | 86 | 4.79 | 6.61 | 6.50 | 59 |
| tiny-16384-csa-cp1 | flashmla_fwd_ref | 1.48 | 1.44 | 76 | 12.29 | - | - | - |
| tiny-16384-hca-cp1 | tilelang | 2.74 | 2.02 | 33 | 1.89 | 6.19 | 5.35 | 51 |
| tiny-16384-hca-cp1 | cute | 1.99 | 1.86 | 45 | 1.89 | 5.41 | 5.19 | 58 |
| tiny-16384-hca-cp1 | cute_ws | 0.98 | 0.93 | 92 | 3.05 | 4.37 | 4.26 | 72 |
| tiny-16384-hca-cp1 | flashmla_fwd_ref | 1.19 | 1.06 | 76 | 3.05 | - | - | - |
| tiny-16384-sliding-cp1 | tilelang | 2.73 | 2.03 | 33 | 1.89 | 6.22 | 5.34 | 51 |
| tiny-16384-sliding-cp1 | cute | 1.98 | 1.86 | 46 | 1.89 | 5.41 | 5.18 | 58 |
| tiny-16384-sliding-cp1 | cute_ws | 0.98 | 0.93 | 92 | 3.05 | 4.37 | 4.23 | 72 |
| tiny-16384-sliding-cp1 | flashmla_fwd_ref | 1.18 | 1.05 | 76 | 3.05 | - | - | - |
| single-49208-csa-cp1 | tilelang | 16.39 | 16.38 | 248 | 1.00 | 70.85 | 70.01 | 200 |
| single-49208-csa-cp1 | cute | 15.15 | 15.11 | 268 | 1.00 | 69.46 | 69.33 | 204 |
| single-49208-csa-cp1 | cute_ws | 7.87 | 7.27 | 516 | 1.00 | 61.76 | 61.54 | 230 |
| single-49208-csa-cp1 | flashmla_fwd_ref | 8.63 | 8.28 | 470 | 1.02 | - | - | - |
| single-49208-hca-cp1 | tilelang | 11.50 | 11.07 | 179 | 1.10 | 42.56 | 41.79 | 170 |
| single-49208-hca-cp1 | cute | 10.31 | 10.25 | 200 | 1.10 | 41.33 | 41.20 | 175 |
| single-49208-hca-cp1 | cute_ws | 5.01 | 4.76 | 411 | 1.20 | 35.89 | 35.70 | 201 |
| single-49208-hca-cp1 | flashmla_fwd_ref | 5.67 | 5.59 | 364 | 1.60 | - | - | - |
| single-49208-sliding-cp1 | tilelang | 7.47 | 6.81 | 110 | 1.00 | 22.95 | 22.06 | 126 |
| single-49208-sliding-cp1 | cute | 6.39 | 6.28 | 129 | 1.00 | 21.82 | 21.55 | 132 |
| single-49208-sliding-cp1 | cute_ws | 2.43 | 2.37 | 340 | 1.00 | 17.82 | 17.64 | 162 |
| single-49208-sliding-cp1 | flashmla_fwd_ref | 3.18 | 3.09 | 259 | 1.00 | - | - | - |
| short-49208-csa-cp1 | tilelang | 12.99 | 12.98 | 204 | 1.07 | 51.11 | 50.33 | 181 |
| short-49208-csa-cp1 | cute | 11.81 | 11.77 | 224 | 1.07 | 49.85 | 49.72 | 186 |
| short-49208-csa-cp1 | cute_ws | 6.01 | 5.60 | 440 | 1.13 | 43.75 | 43.57 | 212 |
| short-49208-csa-cp1 | flashmla_fwd_ref | 6.61 | 6.28 | 400 | 1.56 | - | - | - |
| short-49208-hca-cp1 | tilelang | 8.56 | 7.87 | 104 | 1.36 | 26.19 | 25.37 | 120 |
| short-49208-hca-cp1 | cute | 7.41 | 7.35 | 121 | 1.36 | 25.04 | 24.83 | 125 |
| short-49208-hca-cp1 | cute_ws | 3.57 | 3.50 | 251 | 1.78 | 21.17 | 20.97 | 148 |
| short-49208-hca-cp1 | flashmla_fwd_ref | 4.12 | 4.09 | 217 | 1.85 | - | - | - |
| short-49208-sliding-cp1 | tilelang | 7.46 | 6.82 | 107 | 1.02 | 22.66 | 21.77 | 123 |
| short-49208-sliding-cp1 | cute | 6.35 | 6.27 | 125 | 1.02 | 21.51 | 21.27 | 129 |
| short-49208-sliding-cp1 | cute_ws | 2.43 | 2.35 | 327 | 1.04 | 17.55 | 17.37 | 159 |
| short-49208-sliding-cp1 | flashmla_fwd_ref | 3.16 | 3.09 | 251 | 1.04 | - | - | - |
| heavy-49208-csa-cp1 | tilelang | 14.86 | 14.87 | 230 | 1.03 | 61.74 | 60.93 | 194 |
| heavy-49208-csa-cp1 | cute | 13.64 | 13.62 | 251 | 1.03 | 60.44 | 60.23 | 198 |
| heavy-49208-csa-cp1 | cute_ws | 6.95 | 6.56 | 492 | 1.05 | 53.46 | 53.26 | 224 |
| heavy-49208-csa-cp1 | flashmla_fwd_ref | 7.66 | 7.28 | 446 | 1.21 | - | - | - |
| heavy-49208-hca-cp1 | tilelang | 9.10 | 8.58 | 128 | 1.21 | 29.68 | 28.87 | 137 |
| heavy-49208-hca-cp1 | cute | 7.98 | 8.00 | 146 | 1.21 | 28.52 | 28.33 | 143 |
| heavy-49208-hca-cp1 | cute_ws | 3.68 | 3.64 | 317 | 1.43 | 24.22 | 24.06 | 168 |
| heavy-49208-hca-cp1 | flashmla_fwd_ref | 4.33 | 4.28 | 269 | 2.13 | - | - | - |
| heavy-49208-sliding-cp1 | tilelang | 7.46 | 6.86 | 106 | 1.02 | 22.63 | 21.73 | 123 |
| heavy-49208-sliding-cp1 | cute | 6.36 | 6.26 | 125 | 1.02 | 21.50 | 21.22 | 129 |
| heavy-49208-sliding-cp1 | cute_ws | 2.43 | 2.39 | 327 | 1.04 | 17.53 | 17.34 | 159 |
| heavy-49208-sliding-cp1 | flashmla_fwd_ref | 3.15 | 3.09 | 252 | 1.04 | - | - | - |
| tiny-49208-csa-cp1 | tilelang | 8.40 | 7.84 | 41 | 3.49 | 24.15 | 23.41 | 50 |
| tiny-49208-csa-cp1 | cute | 7.36 | 7.32 | 47 | 3.49 | 23.04 | 22.89 | 52 |
| tiny-49208-csa-cp1 | cute_ws | 3.73 | 3.64 | 92 | 4.69 | 19.41 | 19.23 | 62 |
| tiny-49208-csa-cp1 | flashmla_fwd_ref | 4.30 | 4.28 | 80 | 12.01 | - | - | - |
| tiny-49208-hca-cp1 | tilelang | 6.72 | 6.04 | 41 | 1.86 | 16.76 | 15.90 | 58 |
| tiny-49208-hca-cp1 | cute | 5.63 | 5.55 | 49 | 1.86 | 15.65 | 15.39 | 62 |
| tiny-49208-hca-cp1 | cute_ws | 2.76 | 2.69 | 100 | 2.98 | 12.71 | 12.55 | 76 |
| tiny-49208-hca-cp1 | flashmla_fwd_ref | 3.19 | 3.11 | 87 | 2.98 | - | - | - |
| tiny-49208-sliding-cp1 | tilelang | 6.73 | 6.04 | 41 | 1.86 | 16.76 | 15.88 | 58 |
| tiny-49208-sliding-cp1 | cute | 5.64 | 5.54 | 49 | 1.86 | 15.65 | 15.38 | 62 |
| tiny-49208-sliding-cp1 | cute_ws | 2.75 | 2.69 | 101 | 2.98 | 12.71 | 12.55 | 76 |
| tiny-49208-sliding-cp1 | flashmla_fwd_ref | 3.19 | 3.12 | 87 | 2.98 | - | - | - |
| single-65536-csa-cp1 | tilelang | 21.60 | 21.73 | 251 | 1.00 | 95.48 | 94.99 | 199 |
| single-65536-csa-cp1 | cute | 20.17 | 20.14 | 269 | 1.00 | 93.90 | 93.68 | 202 |
| single-65536-csa-cp1 | cute_ws | 10.95 | 9.77 | 496 | 1.00 | 83.96 | 83.59 | 226 |
| single-65536-csa-cp1 | flashmla_fwd_ref | 11.47 | 11.08 | 473 | 1.01 | - | - | - |
| single-65536-hca-cp1 | tilelang | 16.49 | 16.22 | 200 | 1.08 | 64.33 | 63.65 | 179 |
| single-65536-hca-cp1 | cute | 15.21 | 15.18 | 216 | 1.08 | 62.94 | 62.83 | 183 |
| single-65536-hca-cp1 | cute_ws | 7.66 | 7.15 | 430 | 1.17 | 55.01 | 54.84 | 210 |
| single-65536-hca-cp1 | flashmla_fwd_ref | 8.43 | 8.25 | 391 | 1.67 | - | - | - |
| single-65536-sliding-cp1 | tilelang | 9.69 | 9.05 | 113 | 1.00 | 30.17 | 29.34 | 127 |
| single-65536-sliding-cp1 | cute | 8.43 | 8.36 | 130 | 1.00 | 28.88 | 28.63 | 133 |
| single-65536-sliding-cp1 | cute_ws | 3.22 | 3.15 | 341 | 1.00 | 23.64 | 23.48 | 163 |
| single-65536-sliding-cp1 | flashmla_fwd_ref | 4.16 | 4.09 | 264 | 1.00 | - | - | - |
| short-65536-csa-cp1 | tilelang | 16.23 | 15.99 | 197 | 1.08 | 63.18 | 62.50 | 177 |
| short-65536-csa-cp1 | cute | 14.92 | 14.89 | 215 | 1.08 | 61.76 | 61.67 | 182 |
| short-65536-csa-cp1 | cute_ws | 7.59 | 7.07 | 422 | 1.16 | 54.08 | 53.91 | 207 |
| short-65536-csa-cp1 | flashmla_fwd_ref | 8.33 | 8.17 | 384 | 1.72 | - | - | - |
| short-65536-hca-cp1 | tilelang | 11.01 | 10.36 | 102 | 1.40 | 33.57 | 32.75 | 117 |
| short-65536-hca-cp1 | cute | 9.66 | 9.69 | 116 | 1.40 | 32.25 | 32.07 | 122 |
| short-65536-hca-cp1 | cute_ws | 4.75 | 4.70 | 237 | 1.87 | 27.29 | 27.12 | 144 |
| short-65536-hca-cp1 | flashmla_fwd_ref | 5.41 | 5.42 | 208 | 1.96 | - | - | - |
| short-65536-sliding-cp1 | tilelang | 9.68 | 9.02 | 108 | 1.02 | 29.72 | 28.84 | 124 |
| short-65536-sliding-cp1 | cute | 8.42 | 8.33 | 125 | 1.02 | 28.41 | 28.14 | 129 |
| short-65536-sliding-cp1 | cute_ws | 3.23 | 3.15 | 325 | 1.05 | 23.20 | 23.03 | 158 |
| short-65536-sliding-cp1 | flashmla_fwd_ref | 4.19 | 4.11 | 250 | 1.05 | - | - | - |
| heavy-65536-csa-cp1 | tilelang | 17.40 | 17.38 | 210 | 1.07 | 69.67 | 68.97 | 183 |
| heavy-65536-csa-cp1 | cute | 16.06 | 16.03 | 227 | 1.07 | 68.24 | 68.14 | 187 |
| heavy-65536-csa-cp1 | cute_ws | 8.45 | 7.75 | 432 | 1.12 | 60.06 | 59.90 | 213 |
| heavy-65536-csa-cp1 | flashmla_fwd_ref | 9.01 | 8.81 | 406 | 1.50 | - | - | - |
| heavy-65536-hca-cp1 | tilelang | 11.74 | 11.25 | 125 | 1.25 | 38.30 | 37.62 | 134 |
| heavy-65536-hca-cp1 | cute | 10.58 | 10.54 | 139 | 1.25 | 36.97 | 36.89 | 139 |
| heavy-65536-hca-cp1 | cute_ws | 4.99 | 4.91 | 295 | 1.52 | 31.45 | 31.28 | 163 |
| heavy-65536-hca-cp1 | flashmla_fwd_ref | 5.78 | 5.72 | 254 | 2.25 | - | - | - |
| heavy-65536-sliding-cp1 | tilelang | 9.61 | 9.08 | 105 | 1.04 | 29.28 | 28.46 | 121 |
| heavy-65536-sliding-cp1 | cute | 8.38 | 8.30 | 121 | 1.04 | 27.99 | 27.77 | 127 |
| heavy-65536-sliding-cp1 | cute_ws | 3.21 | 3.17 | 315 | 1.09 | 22.83 | 22.67 | 155 |
| heavy-65536-sliding-cp1 | flashmla_fwd_ref | 4.13 | 4.10 | 245 | 1.09 | - | - | - |
| tiny-65536-csa-cp1 | tilelang | 10.91 | 10.43 | 42 | 3.45 | 31.85 | 31.18 | 51 |
| tiny-65536-csa-cp1 | cute | 9.77 | 9.74 | 47 | 3.45 | 30.61 | 30.50 | 53 |
| tiny-65536-csa-cp1 | cute_ws | 4.92 | 4.85 | 94 | 4.64 | 25.78 | 25.63 | 63 |
| tiny-65536-csa-cp1 | flashmla_fwd_ref | 5.66 | 5.68 | 82 | 11.87 | - | - | - |
| tiny-65536-hca-cp1 | tilelang | 8.69 | 8.04 | 43 | 1.85 | 22.03 | 21.19 | 59 |
| tiny-65536-hca-cp1 | cute | 7.44 | 7.36 | 50 | 1.85 | 20.74 | 20.52 | 63 |
| tiny-65536-hca-cp1 | cute_ws | 3.67 | 3.64 | 102 | 2.95 | 16.93 | 16.75 | 77 |
| tiny-65536-hca-cp1 | flashmla_fwd_ref | 4.16 | 4.13 | 90 | 2.95 | - | - | - |
| tiny-65536-sliding-cp1 | tilelang | 8.69 | 8.03 | 43 | 1.85 | 22.03 | 21.19 | 59 |
| tiny-65536-sliding-cp1 | cute | 7.44 | 7.37 | 50 | 1.85 | 20.74 | 20.52 | 63 |
| tiny-65536-sliding-cp1 | cute_ws | 3.67 | 3.65 | 102 | 2.95 | 16.94 | 16.72 | 77 |
| tiny-65536-sliding-cp1 | flashmla_fwd_ref | 4.16 | 4.14 | 90 | 2.95 | - | - | - |

## Host overhead (op-boundary minus GPU busy time), µs

| arm | mode | min | median | max | items where it exceeds GPU time |
|---|---|---|---|---|---|
| cute | fwd | -27 | 123 | 156 | 70 of 240 |
| cute | fwd_bwd | 81 | 404 | 1560 | 76 of 240 |
| cute_ws | fwd | 24 | 53 | 1184 | 57 of 240 |
| cute_ws | fwd_bwd | 100 | 540 | 1396 | 82 of 240 |
| flashmla_fwd_ref | fwd | -17 | 147 | 390 | 86 of 240 |
| tilelang | fwd | -130 | 697 | 770 | 127 of 240 |
| tilelang | fwd_bwd | 492 | 1011 | 2173 | 106 of 240 |

## Correctness gate

Largest relative error over all items (max deviation over the reference's max magnitude), and its bound.

| arm | basis | tensor | max relative error | bound | failures |
|---|---|---|---|---|---|
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
| tilelang | cold | 20.94 | 0.617 | 4 | 0 | 42.0 | 16.0 |
| tilelang | warm | 0.29 | 0.614 | 0 | 4 | 41.6 | 0.0 |
| cute | cold | 14.33 | 0.583 | 4 | 0 | 41.3 | 9.8 |
| cute | warm | 1.44 | 0.590 | 1 | 3 | 40.7 | 0.0 |
| cute_ws | cold | 16.08 | 0.510 | 4 | 0 | 41.1 | 9.8 |
| cute_ws | warm | 3.18 | 0.511 | 1 | 3 | 41.0 | 0.0 |

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

