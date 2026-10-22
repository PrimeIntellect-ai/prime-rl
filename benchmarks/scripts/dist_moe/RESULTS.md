# dist_moe on B300: validation and SFT A/B

All runs used one 8x B300 node (SM103, NV18), torch 2.14.0+cu130, dist_moe `caf454c`, and locally built torch 2.14 wheels for deep-ep, deep-gemm, prime-kernels and flash-mla. The trainer code was `acf7e6354`. Every run used `model.offload = "none"`, `ac_offloading = "None"`, `ep = 8`, `pad_to_seq_len = true` on every arm, no checkpointing, and fresh Triton, Inductor and CuTe DSL caches. Timing is the median `time/step` over steps 3-20 of untraced runs, with min and max in parentheses. Exact launch commands are in `commands.sh` (via `launch.sh` and `queue.sh`), and per-run summaries are in `results/`.

## Configs

- `qwen3-30b-a3b-sft.toml`: Qwen3-30B-A3B (the Thinking-2507 checkpoint), all 48 layers, fake data, `force_balanced_routing`, 32k tokens per GPU, AC full.
- `glm-4.5-air-sft.toml`: GLM-4.5-Air truncated to 8 layers, real data (`PrimeIntellect/INTELLECT-3-SFT-10K` math split, `glm-4.5` renderer), 16k tokens per GPU, AC full.
- Overlays: `dispatch-{torch,deepep,dist-moe}.toml`, `matrix-small.toml`, `ac-{none,selective}.toml`, `compile-{off,fullgraph}.toml`, `cp2.toml`. Compose them as `uv run --no-sync sft @ <base> @ <overlay> ...`.

## Timing A/B

| Workload | Arm | Step (s) | Tok/s/GPU | MFU | Peak (GiB) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-30B-A3B | torch | 8.608 (8.588-8.627) | 7614 | 32.3% | 79.83 |
| Qwen3-30B-A3B | torch, repeat | 8.612 | 7610 | 32.3% | 79.83 |
| Qwen3-30B-A3B | dist_moe | 5.063 (5.041-5.086) | 12940 | 55.0% | 98.11 |
| Qwen3-30B-A3B | dist_moe, repeat | 5.061 | 12950 | 55.0% | 98.11 |
| GLM-4.5-Air, 8 layers | torch | 1.890 (1.844-1.944) | 17330 | 26.7% | 55.65 |
| GLM-4.5-Air, 8 layers | torch, repeat | 1.900 | 17250 | 26.7% | 55.65 |
| GLM-4.5-Air, 8 layers | dist_moe | 1.130 (1.109-1.165) | 28990 | 44.4% | 63.71 |
| GLM-4.5-Air, 8 layers | dist_moe, repeat | 1.129 | 29030 | 44.4% | 63.71 |

With the default `fullgraph = false`, dist_moe is 1.70x faster on Qwen3-30B-A3B and 1.67x on GLM-4.5-Air, and same-code repeats agree within 0.5%. The extra peak memory matches dist_moe's startup log: on Qwen, +18.3 GiB against a 13.75 GiB device buffer plus 4.5 GiB of routing state; on GLM, +8.1 GiB against 8.31 plus 0.33 GiB.

### With `fullgraph = true`

The same workloads with `compile-fullgraph.toml` added to both arms (runs `fg-*`, summaries `results/fg-*.csv`):

| Workload | Arm | Step (s) | Tok/s/GPU | MFU | Peak (GiB) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-30B-A3B | torch | 5.960 (5.941-6.002) | 10996 | 46.7% | 79.83 |
| Qwen3-30B-A3B | torch, repeat | 5.955 | 11005 | 46.7% | 79.83 |
| Qwen3-30B-A3B | dist_moe | 5.057 (5.034-5.069) | 12959 | 55.0% | 98.11 |
| Qwen3-30B-A3B | dist_moe, repeat | 5.044 | 12994 | 55.2% | 98.11 |
| GLM-4.5-Air, 8 layers | torch | 1.458 (1.414-1.522) | 22481 | 34.9% | 55.10 |
| GLM-4.5-Air, 8 layers | torch, repeat | 1.450 | 22596 | 34.6% | 55.10 |
| GLM-4.5-Air, 8 layers | dist_moe | 1.124 (1.108-1.158) | 29145 | 44.8% | 63.71 |
| GLM-4.5-Air, 8 layers | dist_moe, repeat | 1.125 | 29122 | 44.6% | 63.71 |

Fullgraph speeds up `torch` dispatch 1.44x on Qwen and 1.30x on GLM and leaves dist_moe unchanged, so dist_moe's lead shrinks to 1.18x on Qwen and 1.30x on GLM. In the Qwen traces (`profiling/sft-ab/derived/fg-qwen-accounting.txt`), fullgraph cuts `torch` dispatch's elementwise/copy time from 2156 to 415 ms per step and its permute/gather/scatter time from 1795 to 537 ms, while its exposed all-to-all stays at about 1.2 s. dist_moe removes that all-to-all. Recompiles match the fullgraph-off runs: none on Qwen, one per rank in step 1 on GLM. Fullgraph moves `torch` dispatch's Qwen step-1 loss from 1.45349 to 1.45491, reproducibly.

The `deepep` arm fails at this config on torch 2.14. With AC full, step 1 raises `RuntimeError: Autograd not support dtype: Byte` in `combine` (`src/prime_rl/trainer/distributed/deepep.py:389`) during the AC recompute, with or without compile. With AC none and compile on, Dynamo traces the DeepEP `Buffer` constructor and fails on `ProcessGroup._device_types`. With AC none and compile off on GLM, the step times are torch 1.483 s, deepep 1.676 s and dist_moe 1.084 s, with peaks of 97.5, 107.2 and 79.4 GiB. deepep turns off grad clipping when ep > 1, so its loss curve differs by design.

## Correctness matrix

This used Qwen3-30B-A3B truncated to 4 layers, 8192 tokens per rank, balanced routing and 10 steps (`matrix-small.toml`). All 18 cells, {AC none, full, selective} x {compile off, default, fullgraph} x {torch, dist_moe}, ran without hangs, with 0 recompiles, and with the `dist_moe/*_used` overflow metrics logged in every dist_moe cell. Step-1 loss was 20.3217-20.3246 for torch and 20.3217-20.3231 for dist_moe, and step-10 loss was 12.7864-12.7968 and 12.7881-12.7966. The ranges overlap, and the spread is no larger than torch's own spread across compile modes. Step 1 of dist_moe takes about 33 s, against 12-16 s for torch, because of CuTe DSL JIT compilation.

On GLM real data, step 10 is sensitive. Loss and grad norm were 5.31 / 99.8 and 5.36 / 110.0 for the two torch runs, and 5.52 / 136.0 and 5.46 / 126.3 for the two dist_moe runs. From step 11 on, dist_moe's loss stays about 0.01 above torch's, in the eager no-AC arms as well. A longer run is needed to tell whether this offset is real.

## Context parallelism

On GLM-4.5-Air with `cp = 2` and `fullgraph = false`, the median over steps 6-20 is torch 2.251 s and dist_moe 1.379 s. torch recompiles once per rank in step 1. dist_moe recompiles 20 times in step 1, then 4 more times in step 5 on ranks 6 and 7, on a guard on `ULYSSES_PARAMS['cu_seqlens']`, which makes step 5 take 6.7 s. CP users would therefore see occasional recompile stalls with dist_moe.

## Layer benchmark

`benchmarks/dist_moe/bench_dist_moe.py` uses 128 experts, top-k 8, hidden 2048, intermediate 768 and ep 8, with `NCCL_MAX_CTAS=16`. dist_moe is compared with the bf16 EP path (`TorchTokenDispatcher` plus `BF16ExpertCompute`). The GPU time is fwd+bwd, the median over 50 iterations of the max over ranks. The loop time is back to back and includes host launch time.

| Tokens per rank | Routing | bf16 EP GPU (ms) | dist_moe GPU (ms) | GPU speedup | Loop speedup |
| --- | --- | --- | --- | --- | --- |
| 8192 | balanced | 10.215 | 3.671 | 2.78x | 1.52x |
| 8192 | random | 11.654 | 3.684 | 3.16x | 1.77x |
| 16384 | balanced | 19.012 | 6.837 | 2.78x | 2.36x |
| 16384 | random | 21.616 | 6.897 | 3.13x | 3.14x |

dist_moe's forward costs about 2.3 ms of host time per layer, against about 1.06 ms of GPU time at 8192 tokens per rank. At that size the layer is host-bound when run alone. In the SFT runs above, other GPU work hides it: GPU idle time is 1.6% of the Qwen step. The relative differences from the bf16 EP path are 0.0042 for y, 0.0053 for dx and 0.0036 for the score and weight gradients.

## dist_moe's own tests and compile checks

dist_moe's test suite (`tests/test_dist_moe.py` and `tests/test_dist_moe_vmm.py` at `caf454c`) gives 123 passed and 11 failed. Ten failures are MXFP8 cosine-similarity checks that miss a 0.9975 floor by 4e-9 to 4e-5. One is a bf16 weight-gradient check in which 1 of about 1M elements is 0.031 off, against a 0.02 limit. All 11 are deterministic.

`check_compile_ac.py` runs `DistMoEExpertCompute` with real kernels under {AC none, full, selective} x {eager, `torch.compile(fullgraph=True)`} on 4 GPUs. It gives no graph breaks, one forward-op execution per step, and outputs and gradients bitwise identical to eager.
