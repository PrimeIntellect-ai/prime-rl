# FP8 stochastic rounding: H200 validation

On 2026-10-13, the full 43-layer DeepSeek V4 Flash Base model completed five real-data SFT steps with RNE and SR on two H200 nodes using momentum-free SGD. SR also completed five steps with CP2 at a smaller global batch. All successful runs had finite losses and gradient norms and zero NaN losses. These are correctness and capacity checks, not convergence or model-quality experiments.

**B300 runtime is unvalidated.** The scale and SR kernels cross-compiled for SM103, including UE8M0 scale specialization, but no B300 was available for execution or DeepGEMM integration testing.

## Configuration and provenance

- Tested code: `23cec6d15`, based on `7c9d4744c`. The last kernel/backward change is `cf7e395e5`; subsequent changes only set experiment NCCL limits.
- Model: `PrimeIntellect/DeepSeek-V4-Flash-Base-bf16`, cached snapshot `f6844322b0844389ca71de948dadb74dd9bfa4a6`; all 43 layers, pretrained weights.
- Data: `PrimeIntellect/INTELLECT-4-SFT`, revision `22e1ce974e3c5b20ad229674ea1712172f77c1f3`, subset `i3-math--null--low`, shuffle seed 42.
- Hardware: 16 H200 GPUs, eight per node, each reporting 139.80 GiB capacity. Each job requested 32 CPUs and 1000 GiB host RAM per node with a 30-minute final limit.
- Software: PyTorch `2.13.0+cu130`, Triton `3.7.1`, DeepGEMM `2.5.0+891d57b`, NCCL `2.29.7`. Dependency pins were unchanged.
- SGD: learning rate `1e-5`, weight decay `0.01`, maximum gradient norm `1.0`, no momentum/state, `foreach=False`.
- Precision: FP32 optimization and reduction retained; BF16 materialized weights are cast to FP8 for dense and grouped expert GEMMs. SR applies to weights; activation and gradient casts retain RNE.
- Runtime: EP16, torch expert dispatch, full activation checkpointing, activation offload with five inflight activations, 512-token rows, rounding seed `20261013`.
- Memory settings: `NCCL_BUFFSIZE=1048576`, `NCCL_MAX_CTAS=4`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Dense FP8 excludes `o_a_proj` in both RNE and SR to preserve DeepSeek V4's grouped projection semantics.

The experiment [base](sft-base.toml) describes a four-node AdamW setup. The successful runs compose the [two-node SGD overlay](sgd-two-node.toml) on top. [sr.toml](sr.toml) enables dense and expert SR independently of the optimizer; [cp2.toml](cp2.toml) sets context parallelism. RNE remains the trainer default. The FP32 optimization/reduction defaults are unchanged.

Submodule revisions: `prime-envs=89dcc7f`, `prime-kernels=6f48176`, `pydantic-config=65b15df`, `renderers=41f9290`, `verifiers=2cb0a6d`. The installed `prime-kernels` package was `0.1.0+cu130torch2.13.0`.

## Full-depth SFT results

| Job | Mode | Layout | Global batch | Steps | Final loss | Rank-0 peak allocation, GiB | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3967 | RNE | DP16 / CP1 / EP16 | 16 | 5 | 1.01787 | 138.14 | [f7ao8b6x](https://wandb.ai/primeintellect/fp8-train-sr/runs/f7ao8b6x) |
| 3970 | SR | DP16 / CP1 / EP16 | 16 | 5 | 1.01427 | 138.19 | [q3325d37](https://wandb.ai/primeintellect/fp8-train-sr/runs/q3325d37) |
| 3972 | SR | DP8 / CP2 / EP16 | 8 | 5 | 1.07590 | 138.02 | [vq0cthaj](https://wandb.ai/primeintellect/fp8-train-sr/runs/vq0cthaj) |

All three jobs exited successfully and released their allocations. CP1 RNE/SR model, data and optimizer settings match except SR, and their per-step token counts match. CP2 uses a different global batch and is not a matched loss/throughput comparison against CP1. Each successful configuration has one accumulation microstep per optimizer step.

| Step | RNE CP1 loss | SR CP1 loss | SR CP2 loss | RNE CP1 seconds | SR CP1 seconds | SR CP2 seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.93412 | 0.93846 | 0.94249 | 101.919 | 103.097 | 43.932 |
| 2 | 1.04424 | 1.04251 | 0.93113 | 25.474 | 13.632 | 12.605 |
| 3 | 1.03568 | 1.03293 | 1.06666 | 19.700 | 14.463 | 12.579 |
| 4 | 1.06075 | 1.05948 | 0.97670 | 12.383 | 12.771 | 11.776 |
| 5 | 1.01787 | 1.01427 | 1.07590 | 11.827 | 12.134 | 29.746 |

The last-two-step CP1 medians were **12.1046 seconds RNE / 12.4524 seconds SR (+2.87% observed)**. This is not a robust isolated SR overhead estimate: there are only two late samples per mode, the jobs ran on different node pairs, compilation/allocator warmup varies, and memory pressure repeatedly triggers allocator reclamation. The first step also records allocation stacks and snapshots before grouped gradients. Five-step loss differences do not establish a quality benefit.

## Capacity failures and buffer correction

SGD alone did not make the original two-node configuration fit. These attempts failed before completing an optimizer step:

| Job | Configuration | Failing allocation | W&B |
| --- | --- | --- | --- |
| 3963 | RNE, EP8, batch 32, 1024 tokens | 1 GiB gradient conversion; 675 MiB free on rank 0 | [6odycsw2](https://wandb.ai/primeintellect/fp8-train-sr/runs/6odycsw2) |
| 3964 | RNE, EP16, batch 32, 1024 tokens | 512 MiB gradient conversion; 141 MiB free on rank 0 | [86n4v853](https://wandb.ai/primeintellect/fp8-train-sr/runs/86n4v853) |
| 3966 | RNE, EP16, batch 16, 512 tokens, buffer correction | 1 GiB FP32 gradient allocation; 663 MiB free on rank 0 | [qc27hvuv](https://wandb.ai/primeintellect/fp8-train-sr/runs/qc27hvuv) |
| 3971 | SR, CP2/EP16, batch 16, 512 tokens, reduced NCCL buffers | 512 MiB gradient conversion; 207/281 MiB free on the logged ranks | [vmm5j9n8](https://wandb.ai/primeintellect/fp8-train-sr/runs/vmm5j9n8) |

Grouped backward allocated full-size placeholder gradients before allocating the actual results. Moving those placeholders into the unused-gradient branches reduced additional peak allocation from **2,173,184,512 to 1,627,924,992 bytes (520 MiB saved)** for an EP16 weight shape `[16, 4096, 4096]`. Direct old/new comparison produced **bitwise-identical dX and dW** for both RNE and SR.

The successful runs additionally limit NCCL workspace and use one microstep. CP2 with batch 16 requires two microsteps and still OOMed; batch 8 passed. Peak allocation near 138 GiB leaves little headroom, so these results do not establish capacity for longer sequences or larger batches. No optimization/reduction dtype change or full CPU optimizer offload was used.

## Kernel correctness and performance

The final H200 suite passed **50 GPU cases**, including exhaustive finite-BF16/reference rounding, FP32 interval boundaries, unbiased-rounding probabilities, non-finite values and signed zero, odd/noncontiguous shapes, transposed coordinates, EP slice reconstruction, RNG offsets above `2**32`, and address strides above `2**31`. Eager/compiled dense and grouped backward checks use the exact forward-rounded weights for dX and retain the existing dW calculation. Coverage includes partial gradients, empty experts, tied parameters, saved-seed immutability and full/selective activation-checkpoint replay.

An actual two-rank FSDP/checkpoint test passed three steps with identical replica FP8 bytes and exact global expert slice reconstruction. A 43-layer regional-compilation probe produced graph counts `[1, 1, 2]` across two same-shape steps and a shape change, matching RNE. Immutable tensor stream IDs avoid specializing one graph per layer. The config suite passed **245 cases**, and a direct two-update SGD check matched the expected equation with no optimizer state.

Operator benchmarks used BF16 inputs on H200, CUDA-event timing via `triton.testing.do_bench`, 100 ms warmup and 500 ms measurement in RNE → SR → RNE order. Entries below are medians; these are operator timings, not full-model throughput:

| Operation | Shape | RNE, ms | SR, ms | RNE repeat, ms | Change |
| --- | --- | --- | --- | --- | --- |
| Weight cast | `[1, 4096, 4096]` | 0.02109 | 0.08240 | 0.02106 | +291% |
| Weight cast | `[32, 2048, 4096]` | 0.22170 | 1.12606 | 0.22150 | +408% |
| Dense forward + backward | X `[512,4096]`, W `[4096,4096]` | 0.67814 | 0.73091 | 0.67645 | +7.8% |
| Grouped forward + backward | X `[4096,4096]`, W `[8,4096,2048]`, 512 tokens/expert | 1.91251 | 2.22934 | 1.90966 | +16.6% |

SR uses a scale pass and a 16×128 rounding pass without allocating a full random mask. The measured rounding kernel has zero spills; RNE retains its single-pass cast. Earlier one-layer real-data smoke runs also passed for RNE/CP1, SR/CP1, SR/CP2, and SR/SGD; those are supplementary evidence, distinct from the full-depth results above.

GPU checks require an allocated SM90+ GPU with DeepGEMM installed:

```bash
uv run pytest -q tests/unit/train/models/test_fp8_utils.py tests/unit/train/models/test_fp8_rounding.py
uv run pytest -q tests/unit/test_configs.py
```

## Compose the validation configs

The base uses cluster-specific model/tokenizer cache paths and the `all` Slurm partition; adjust these for another environment. These commands generate launch previews without submitting jobs:

```bash
# RNE, CP1, batch 16
uv run sft @ configs/experiments/fp8-sr/sft-base.toml \
  @ configs/experiments/fp8-sr/sgd-two-node.toml \
  --slurm.time 00:30:00 --run.name fp8-rne-sgd-validation --dry-run

# SR, CP1, batch 16
uv run sft @ configs/experiments/fp8-sr/sft-base.toml \
  @ configs/experiments/fp8-sr/sgd-two-node.toml \
  @ configs/experiments/fp8-sr/sr.toml \
  --slurm.time 00:30:00 --run.name fp8-sr-sgd-validation --dry-run

# SR, CP2, batch 8: one accumulation microstep
uv run sft @ configs/experiments/fp8-sr/sft-base.toml \
  @ configs/experiments/fp8-sr/sgd-two-node.toml \
  @ configs/experiments/fp8-sr/sr.toml \
  @ configs/experiments/fp8-sr/cp2.toml --data.batch-size 8 \
  --slurm.time 00:30:00 --run.name fp8-sr-sgd-cp2-validation --dry-run
```

The measured jobs used a private Slurm template with bounded CPU/RAM requests, no whole-node exclusivity, a job-specific rendezvous port, locked environment synchronization and no broad process cleanup. Its private entrypoint records rank-0 memory before the first 86 grouped weight-gradient calls and then disables allocation-stack tracing. Those diagnostics do not change the kernel/trainer calculations. The default generated launch template in the commands above does not reproduce these allocation and profiling details.

## Evidence locations

Original run root: `/home/phi/lib/.worktrees/prime-rl-fp8-sr/outputs/fp8-sr/`.

- RNE CP1: `rne-sgd-dp16-cp1-ep16-nccl`.
- SR CP1: `sr-sgd-dp16-cp1-ep16-nccl`.
- SR CP2: `sr-sgd-dp8-cp2-ep16-b8`.

Each run contains its submitted command, resolved config, generated Slurm script, logs, `monitors/file/metrics.jsonl` and `STATUS.md`. W&B links above provide the corresponding online metrics. No task allocations remain active; prepared three-node fallback jobs were never submitted.

Local artifacts under `/home/phi/lib/.worktrees/prime-rl-fp8-sr/outputs/validation/` include:

- `full-rne-sgd-summary.json`, `full-sr-sgd-cp1-summary.json`, `full-sr-sgd-cp2-summary.json`, `full-sgd-cp1-comparison.json`: per-step metrics, matched-config checks and first-backward memory samples.
- `sr-memory-h200.{log,xml}`, `config-tests-sgd-pr.log`, `distributed-sr-h200.json`, `regional-compile-sr.json`: test evidence.
- `benchmark-sr-h200.json` and `benchmark_sr.py`: operator medians/p20/p80 and the benchmark procedure.
- `grouped-backward-memory.{log,json}` and `grouped_backward_memory.py`: old/new gradient equality and memory measurements.
- `sm103-compile.json`, `sr-scale-sm103.ptx`, `sr-round-sm103.ptx`: cross-compilation evidence only.

These local artifacts are not tracked in Git. The chronological execution log remains at `/home/phi/lib/research-prod/study/workstreams/fp8_train_sr/fp8_sr_note.md`.
