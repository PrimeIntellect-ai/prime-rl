# Experimental Grouped GEMM

This page tracks the current DeepGEMM FP8 grouped GEMM prototype that was developed outside the main trainer path and imported here for iterative integration.

## What is included

- `src/prime_rl/trainer/fp8_linear.py` contains the current single-linear FP8 path used for integration work.
- `src/prime_rl/trainer/experimental/grouped_gemm/fp8_grouped_linear.py` contains the grouped GEMM prototype classes (`FP8GroupedLinearDeepGEMM`, `FP8LinearDeepGEMM`) and supporting quantization kernels.
- `scripts/experimental/bench_deepgemm_grouped_linear.py` runs grouped correctness and performance checks.
- `scripts/experimental/bench_deepgemm_linear.py` runs single-linear correctness and performance checks against BF16 baselines.

## Running on a multi-GPU node

Both benchmark scripts now honor `LOCAL_RANK` and call `torch.cuda.set_device(local_rank)`, so they can run directly with `torchrun`.

Single process sanity check:

```bash
uv run python scripts/experimental/bench_deepgemm_grouped_linear.py --experts 8 --k 7168 --n 4096 --tokens-per-expert 2048
```

Multi-GPU launch (one benchmark process per GPU):

```bash
uv run torchrun --standalone --nproc-per-node 8 scripts/experimental/bench_deepgemm_grouped_linear.py --experts 8 --k 7168 --n 4096 --tokens-per-expert 2048 --skip-correctness
```

Single-linear benchmark:

```bash
uv run python scripts/experimental/bench_deepgemm_linear.py --mkn 16384,4096,4096
```

## Integration plan

1. Keep grouped GEMM work isolated in `trainer/experimental/grouped_gemm` while the API is stabilized.
2. Port grouped kernels and autograd into `trainer/fp8_linear.py` in small PRs, preserving existing trainer behavior.
3. Add dedicated tests under `tests/unit/trainer` once grouped forward and backward API contracts are finalized.
