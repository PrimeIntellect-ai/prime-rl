# FP8 weight-gradient GEMM: compile cost vs steady-state time

H200, node 049 of SLURM job 3470, code at b9123429c. All three arms run from this tree. The weight-gradient
`fp8_gemm_nt` is called with each arm's `compiled_dims` and scale layout, with the casts done outside the timed
region:

- `main`: `compiled_dims="nk"`, scales contiguous along the token-block axis (the old layout, via `.contiguous()`).
- `change1`: `compiled_dims="mn"`, old scale layout (e300b1929).
- `change2`: `compiled_dims="mn"`, scales MN-major and TMA-aligned (b9123429c).

Weight shapes (out, in) are (1024, 4096) and (32768, 1024). The arms ran in parallel on separate GPUs of one node.

## Varying shapes: compile cost included

`wgrad_harness.py sweep`: 64 calls, every T in 128..4096 step 128 at both shapes, shuffled with a fixed seed.
Each arm starts from an empty `DG_JIT_CACHE_DIR`. CUDA events around each `fp8_gemm_nt` call capture the host JIT
stall. The cold pass is the first visit to each shape; the warm pass repeats the same calls. Lower is better.

| arm | GEMM compiled | transpose compiled | cold total ms | cold max ms | warm total ms |
|---|---|---|---|---|---|
| main | 65 | 31 | 8914.7 | 2172.9 | 10.9 |
| change1 | 3 | 31 | 135.5 | 47.9 | 9.6 |
| change2 | 3 | 0 | 63.2 | 49.9 | 8.6 |

On this idle node each extra GEMM variant costs about 140 ms of compile and each `transpose_fp32` variant about
2 ms. The largest single stall in `main` (2.2 s) is the first compile of the process. In a real run the compiles
contend for host CPU with the other ranks. On a DeepSeek V4 Flash FP8 RL run they cost about 29 min of
forward-backward time over 80 steps.

## Fixed shapes: steady state, no compile

`bench_ops.py cases_wgrad_arms.py`: do_bench (warmup 50 ms, rep 500 ms, L2 flushed), every kernel warm.
Median ms of two repetitions (rep 1 / rep 2), lower is better.

| shape | T | main | change1 | change2 |
|---|---|---|---|---|
| (1024, 4096) | 1024 | 0.032 / 0.033 | 0.033 / 0.033 | 0.028 / 0.028 |
| (1024, 4096) | 4096 | 0.056 / 0.055 | 0.059 / 0.059 | 0.053 / 0.053 |
| (1024, 4096) | 14336 | 0.144 / 0.144 | 0.153 / 0.156 | 0.146 / 0.147 |
| (32768, 1024) | 1024 | 0.135 / 0.135 | 0.138 / 0.139 | 0.134 / 0.135 |
| (32768, 1024) | 4096 | 0.330 / 0.339 | 0.358 / 0.359 | 0.343 / 0.354 |
| (32768, 1024) | 14336 | 1.152 / 1.150 | 1.166 / 1.190 | 1.182 / 1.183 |

The dynamic-K GEMM ("mn") is 1 to 8% slower than "nk" at a fixed shape. Skipping the transpose wins most of that
back at small T and none of it at (32768, 1024), T = 14336, where change2 stays about 3% (0.03 ms) slower than main.

Numerics: `grad_weight` is bitwise equal across all arms (`wgrad_harness.py bitwise` and `layout`).
