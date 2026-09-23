# FP8 grouped GEMM on H200: base (`ae37b35a3`)

Produced by `bench.py base` (alignment 8, the current `DeepGemmFP8GroupedGemm` dispatcher setting) on one
H200 (SM90, driver 580.173.02, torch 2.13.0+cu130, triton 3.7.1, deep_gemm 2.5.0+891d57b). Raw output:
`results-base-prime-nebius-puku-h200-gpu-064-NVIDIA_H200-sm90.txt`. Full sweep wall time 203 s.

## Method

- Inputs come from the real dispatcher helper `permute_for_grouped_gemm`, so `x.size(0)` is `max_len` with
  a tail beyond `offs[-1]`, and empty experts are padded up to the alignment. Weights are a
  `.transpose(-2, -1)` view of an `(E, N, K)` parameter, as in `GroupedExperts.forward`.
- 32 local experts. `weight: (G, K, N)`, so gate_up is `K=4096, N=4096` (hidden to fused gate and up) and
  down is `K=2048, N=4096` (intermediate to hidden).
- Ragged routing uses a power-law count distribution with max/mean = 6, matching the real run's imbalance.
- `triton.testing.do_bench(warmup=10, rep=40, return_mode="min")`, fastest of 3 repeats, after one
  discarded warm-up bench per shape to ramp clocks. Spread between repeats is logged per number; nearly all
  are below 3%.
- Correctness gate before any timing: fp8 forward, dgrad and wgrad against `torch._grouped_mm` on grouped
  rows. All 36 configurations pass with relative L2 error 0.036 to 0.037 against a bound of 0.1.
- Ceiling arm: a device copy moving the same number of bytes as each cast, so casts read as "Nx off copy".
- Noise control: the first configuration's bf16 forward re-timed at the end drifted by -3.1%.
- `fwd`, `dgrad` and `wgrad` time the ops directly (`grouped_fp8_gemm`, and `grouped_fp8_gemm_backward` with
  one of the two `needs_grad_*` flags). `ag` is a full autograd forward plus backward through the leaf
  tensors, which is what the model sees.

## Forward plus backward, fp8 over bf16 (autograd)

| rows/expert | 128 | 256 | 512 | 1024 | 1536 | 2048 | 4096 | 6144 | 8192 | break-even |
|---|---|---|---|---|---|---|---|---|---|---|
| gate_up balanced | 7.27 | 6.48 | 4.01 | 2.15 | 1.61 | 1.39 | 1.08 | 0.95 | 0.96 | ~5260 |
| gate_up ragged | 6.72 | 5.28 | 3.42 | 2.09 | 1.62 | 1.33 | 1.06 | 0.95 | 0.93 | ~5140 |
| down balanced | 7.26 | 6.61 | 4.15 | 2.39 | 2.01 | 1.67 | 1.19 | 1.03 | 1.00 | > 8192 |
| down ragged | 6.58 | 5.48 | 3.79 | 2.52 | 1.89 | 1.74 | 1.26 | 1.05 | 1.00 | > 8192 |

PLAN.md quotes break-even at ~3072 (gate_up) and ~6200 (down). Those came from summing op-level times.
Summing this harness's op-level times gives a similar picture (gate_up 1.20 at 1536, 0.92 at 6144), but the
autograd numbers are what the model pays, and they are worse. See the finding on grad layout below.

## Per-op time, balanced routing (ms)

gate_up:

| rows/expert | bf16 fwd | bf16 dgrad | bf16 wgrad | fp8 fwd | fp8 dgrad | fp8 wgrad | bf16 ag | fp8 ag |
|---|---|---|---|---|---|---|---|---|
| 128 | 0.393 | 0.392 | 0.459 | 0.832 | 0.843 | 3.213 | 1.187 | 8.635 |
| 512 | 0.847 | 0.844 | 1.014 | 1.169 | 1.176 | 3.454 | 2.382 | 9.556 |
| 1536 | 2.444 | 2.446 | 2.506 | 2.264 | 2.242 | 4.070 | 7.616 | 12.289 |
| 4096 | 6.524 | 6.543 | 6.721 | 4.937 | 5.115 | 7.481 | 19.534 | 21.012 |
| 6144 | 9.749 | 9.836 | 9.586 | 7.360 | 7.513 | 10.942 | 29.943 | 28.527 |
| 8192 | 13.330 | 13.411 | 12.753 | 9.597 | 9.906 | 14.020 | 40.105 | 38.559 |

down:

| rows/expert | bf16 fwd | bf16 dgrad | bf16 wgrad | fp8 fwd | fp8 dgrad | fp8 wgrad | bf16 ag | fp8 ag |
|---|---|---|---|---|---|---|---|---|
| 128 | 0.213 | 0.197 | 0.240 | 0.526 | 0.518 | 1.747 | 0.640 | 4.645 |
| 512 | 0.444 | 0.414 | 0.510 | 0.722 | 0.721 | 1.882 | 1.246 | 5.177 |
| 1536 | 1.267 | 1.230 | 1.300 | 1.339 | 1.325 | 2.262 | 3.329 | 6.676 |
| 4096 | 3.339 | 3.295 | 3.298 | 3.033 | 2.939 | 4.224 | 10.119 | 12.059 |
| 6144 | 5.102 | 4.924 | 4.827 | 4.283 | 4.176 | 6.019 | 15.704 | 16.200 |
| 8192 | 6.874 | 6.616 | 6.573 | 5.390 | 5.433 | 7.889 | 20.332 | 20.375 |

The fp8 forward and dgrad beat bf16 from 1536 rows/expert up for gate_up and from 4096 for down. The fp8
wgrad never wins: it carries ~1.2 ms (gate_up) of token-independent `grad_weight` zeroing and fp32 downcast,
and the k-grouped kernel is weak at low occupancy.

## Casts and helpers, gate_up balanced (ms)

| helper | runs per fwd+bwd | 512 | 1536 | 6144 | x off copy at 1536 |
|---|---|---|---|---|---|
| layout build (incl. `.item()` sync) | 2 | 0.126 | 0.125 | 0.124 | latency bound |
| per-token act cast (`x` fwd, `dy` dgrad) | 2 | 0.064 | 0.183 | 0.870 | 1.25x |
| per-block weight cast (fwd, dgrad) | 2 | 0.425 | 0.428 | 0.424 | 1.12x |
| per-channel k-major cast (`x`, `dy` wgrad) | 2 | 0.092 | 0.273 | 1.225 | 1.88x |
| unpack rows | 2 | 0.072 | 0.208 | 0.819 | 1.08x |
| `grad_weight` fp32 zeros | 1 | 0.459 | 0.459 | 0.459 | 0.89x |
| `grad_weight` fp32 to bf16 | 1 | 0.735 | 0.735 | 0.735 | 0.97x |

These agree with PLAN.md at 1536 rows/expert: weight cast 0.42 ms each (0.847 per fwd+bwd in the plan),
unpack 0.405 per fwd+bwd, zeros 0.457, downcast 0.734. The weight cast, unpack, zeros and downcast all take
at most 1.12x the time of a copy of the same bytes, so they can only be removed, not sped up. The k-major
per-channel cast is the one cast with headroom (1.9x to 2.2x off copy).

## Host launch and syncs

| shape, rows/expert | bf16 wall / GPU (ms) | fp8 wall / GPU (ms) | fp8 syncs per fwd+bwd |
|---|---|---|---|
| gate_up 1536 | 0.16 / 7.62 | 3.35 / 12.29 | 5 |
| gate_up 6144 | 0.65 / 29.94 | 8.53 / 28.53 | 5 |
| down 1536 | 0.14 / 3.33 | 2.20 / 6.68 | 5 |
| down 6144 | 0.49 / 15.70 | 5.45 / 16.20 | 5 |

bf16 launches a fwd+bwd in 2-4% of its GPU time. fp8 blocks the host for 28-34% of it, because each sync
drains the queue: the host cannot launch past a sync until the GPU has finished everything before it.

The 5 syncs, traced with `torch.cuda.set_sync_debug_mode("warn")`:

- forward: `starts_tensor[0] = 0` (`fp8_utils.py:38`, a host-to-device scalar copy from pageable memory)
  and `int(padded_ends[-1].item())` (`fp8_utils.py:46`)
- backward: the same two in the backward's `build_grouped_layout`, plus `ks_tensor.tolist()`
  (`fp8_grouped_gemm.py:182`)

PLAN.md counts only the `.item()` and `tolist()` syncs. Removing `.item()` alone leaves the forward with one
sync, from `starts_tensor[0] = 0`.

## Finding: AccumulateGrad re-lays out the fp8 weight gradient

A profile of one gate_up fwd+bwd at 1536 rows/expert shows a 3.88 ms strided `elementwise_kernel` copy in the
fp8 arm and nothing like it in the bf16 arm. `grouped_fp8_gemm_backward` returns `grad_weight` contiguous
as `(G, K, N)`. The parameter is `(E, N, K)`, so autograd's transpose-backward hands AccumulateGrad a
non-contiguous gradient, and AccumulateGrad copies it into the parameter's layout. `torch._grouped_mm`'s
backward already emits the gradient in the parameter's layout, so bf16 skips the copy. This one copy is the
gap between fp8's op-level sum (8.58 ms) and its autograd time (12.29 ms), and it is larger than every cast
in the table above. PLAN.md does not mention it.
