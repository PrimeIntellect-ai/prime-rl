# DSv4 cudnn_flashmla: compile counts and end-to-end A/B

All runs on one exclusive 8xH200 node (SLURM job 3935, prime-nebius-puku-h200-gpu-043), 2026-10-13.

| code | SHA |
|---|---|
| `main` | `7c9d4744c` |
| PR commit 2 (backend, no clamp) | `5e8ec7f2b` |
| PR product tip (clamp + backward assert) | `4a269ac42` |
| harness used for the runs | `237f56d2d` |

## Unit tests

At `4a269ac42`: `tests/unit/train/models/test_deepseek_v4_kernels.py` 106 passed (24 `cudnn_flashmla` cases and the
compile test, none skipped), `tests/unit/train/models/test_deepseek_v4.py` 14 passed.

## Kernel-level compile counts

`compile_counts.py`, one process per row, forward and backward per call. cuDNN counts are the sizes of the frontend's
three executor caches (preprocess, main, postprocess). Raw output is in `kernel/`.

| case | code | calls | slot widths | cuDNN compiles | first call that compiled |
|---|---|---|---|---|---|
| corpus (lengths, docs, layers, CP) | tip | 272 | 21 widths, 128-640 | 3 | call 1 only |
| wide | tip | 8 | 128-8320 | 3 | call 1 only |
| two clamps (16384, then 1024) | tip | 8 | 128-640 | 4 (2 main) | first call of each clamp |
| corpus, second process | tip | 272 | 21 widths, 128-640 | 3 | call 1 only |
| wide | commit 2 | 8 | 128-8320 | 10 (8 main) | every new width |

The main kernel compiles in 2.65-3.15 s, preprocess and postprocess in about 0.3 s each. TileLang compiled 1 kernel
(the `preprocess` delta) in each cold `cudnn_flashmla` process and 0 in the second process, which loaded it from
its disk cache. The TileLang backend compiled 4.

## End-to-end A/B

DeepSeek-V4-Flash-0731, first 6 layers (sliding, sliding, CSA, HCA, CSA, HCA), real weights, cp 8, ep 8, torch MoE
dispatch, full activation checkpointing, no offloading. INTELLECT-4-SFT (swerebench-v2 and scaleswe bash subsets),
seq_len 65536, batch 4, micro-batch 1, 20 steps, fresh compile caches per run. Arms ran serially in the order
A, C, B, D, B2, C2. Full output: `e2e_compare.md`.

| arm | code | backend |
|---|---|---|
| A | `main` | tilelang |
| B, B2 | tip | tilelang |
| C, C2 | tip | cudnn_flashmla |
| D | commit 2 | cudnn_flashmla |

Steady state is steps 6-20, median (min-max).

| arm | step time s (lower is better) | tokens/s/GPU (higher is better) | peak memory GiB |
|---|---|---|---|
| A | 2.847 (2.481-3.416) | 8829 (7850-10750) | 100.5 |
| B | 2.862 (2.523-3.254) | 9177 (7963-10400) | 100.5 |
| B2 | 2.857 (2.527-3.255) | 9161 (8014-10420) | 100.5 |
| C | 2.820 (2.481-3.273) | 9306 (8046-10850) | 100.5 |
| C2 | 2.788 (2.459-3.213) | 9413 (8340-10800) | 100.5 |
| D | 2.787 (2.468-3.240) | 9406 (8004-10790) | 100.5 |

Parity, max absolute per-step difference over 20 steps:

| pair | loss | grad norm |
|---|---|---|
| B vs B2 (noise floor, tilelang) | 1.68e-3 | 0.104 |
| C vs C2 (noise floor, cudnn_flashmla) | 1.08e-3 | 0.024 |
| A vs B (refactor) | 5.78e-4 | 0.030 |
| A vs C | 1.04e-3 | 0.082 |
| C vs D | 6.13e-4 | 0.050 |

Compiles per rank, identical on all 8 ranks, all in step 1 except where noted:

| arm | cuDNN | TileLang | dynamo recompiles |
|---|---|---|---|
| A, B, B2 | 0 | 4 | 1 |
| C, C2 | 3 | 1 | 1 |
| D | not logged at commit 2 | 1 | 1 |

The one dynamo recompile appears on `main` too, so it is not from this PR.

Early steps, logged `time/step` in seconds (lower is better). Compile s is the max over ranks of the step's logged
cuDNN plus TileLang compile time.

| arm | step 1 | step 1 compile s | step 2 | step 4 | cumulative to step 10 | cumulative to step 20 |
|---|---|---|---|---|---|---|
| A | 125.9 | 16.0 | 3.18 | 4.30 | 153.0 | 181.6 |
| B | 116.4 | 13.0 | 3.22 | 2.89 | 142.3 | 170.8 |
| B2 | 115.7 | 16.0 | 3.10 | 2.94 | 141.4 | 169.7 |
| C | 105.3 | 6.7 | 2.98 | 2.75 | 130.4 | 158.0 |
| C2 | 105.6 | 6.6 | 3.02 | 2.77 | 130.6 | 158.0 |
| D | 119.8 | 2.0 (TileLang only) | 6.17 | 5.93 | 151.0 | 178.5 |

D pays the per-width recompiles the clamp removes: steps 2 and 4 each take 3.2 s more than C, one main-kernel
compile apiece, and step 1 takes 14 s more than C for its first widths. From step 5 on D matches C, because a
64k-token batch only produces widths 128, 256, 384, 512 and 640, all seen by then.

## Why the steady-state gain is about 2%

Rank 0 torch.profiler traces of the last of 5 steps (`top_kernels_traceB.md`, `top_kernels_traceC.md`):

| kernels | tilelang (B) ms | cudnn_flashmla (C) ms |
|---|---|---|
| sparse attention forward, incl. recompute (48 calls) | 56.3 | 26.5 |
| sparse attention backward main (24 calls) | 71.3 | 36.7 |
| delta preprocess, cuDNN pre/post | 8.3 | 13.6 |
| total sparse attention | 136.0 | 76.7 |
| step window | 2946 | 2736 |

Sparse attention takes 4.6% of a step at this size, so its 0.56x GPU time saves about 59 ms per step. Exposed NCCL
(about 700-800 ms) and the grouped expert GEMMs dominate the step.

## Longer context: 16k tokens per GPU (128k sequence)

Same setup with `sft_128k.toml` (seq_len 131072, 20 steps), SLURM job 4008 (prime-nebius-puku-h200-gpu-004),
product code `4a269ac42`, harness `b66ea26f1`. Arms ran in the order C, B, C2, B2. Full output:
`e2e_compare_128k.md`. Peak memory is the max over all 8 ranks of the per-step `max_memory_reserved`, steps 2-20.

| arm | step 1 s | steady step s (lower is better) | tokens/s/GPU (higher is better) | peak GiB (lower is better) |
|---|---|---|---|---|
| B, tilelang | 118.7 | 4.590 (4.054-5.053) | 10440 | 114.7 |
| B2, tilelang | 118.7 | 4.584 (4.158-5.024) | 10390 | 114.7 |
| C, cudnn_flashmla | 113.0 | 4.328 (4.068-6.826) | 11000 | 114.1 |
| C2, cudnn_flashmla | 107.1 | 4.394 (3.919-4.817) | 10820 | 114.1 |

The cudnn_flashmla steady step is 4-6% faster here, against about 2% at 8k tokens per GPU. Cumulative time to step
20 is 208.5 and 209.0 s for tilelang, 205.3 and 193.3 s for cudnn_flashmla. C had two slow steps (3 and 6, 8.1 and
6.8 s) with no compile in them, and C2 did not repeat them, so they were noise. Compiles per rank match the 64k
runs: cuDNN 3 and TileLang 1 for cudnn_flashmla, TileLang 4 for tilelang, all in step 1, plus one dynamo recompile
in every arm. Parity: max per-step loss difference B vs C 7.3e-4, against 1.4e-3 for B vs B2.

## 128k tokens per GPU (1M sequence) does not fit

`sft_1m.toml` ran out of memory in step 1 for reasons unrelated to the backend: the CSA FP8 indexer
(`fp8_indexer.py:203`) allocates its full float32 score matrix, 99.65 GiB for 131072 queries against the
compressed keys. Both backends share that indexer.
