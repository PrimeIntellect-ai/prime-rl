# Why the H200 FP8 gains do not show up on B300

This is a two-hour investigation on one B300 node (`primeintellect-tur-r1n8`, 8x B300 SXM6, sm103) that
follows up `E2E_B300.md`. Short answer:

1. **The H200 gain was a recovery to parity, not a win over bf16.** On H200 the branch took FP8 from 18%
   slower than bf16 (8.43 vs 7.16 s) to level with it (7.13 s). On B300 the FP8 baseline never had that
   penalty, so there was nothing to recover. The H200 shape on one B300 node gives fwd+bwd of 2.045 s at
   the H200 base `ae37b35`, 1.983 s at main `33e02a7`, 1.976 s on this branch and 1.979 s in bf16. Most of
   the small B300 gain since `ae37b35` came from main (#3643), not from the branch.
2. **The branch's savings are real on B300, but the branch also adds a cost of the same size.** Traces
   of main and the branch at the 256k one-node proxy show the branch removing 34.6 ms per step of
   AccumulateGrad and unpack copies, as the op benchmark predicts. Its switch to 128-row alignment,
   however, raises the MoE permute's padding slots from 256 to 4096 per layer, and autograd's
   `IndexBackward0` serializes all of their atomic adds onto one shared zero row: +36.0 ms per step. The
   op benchmark never runs the permute, so it could not see this. Against bf16, the FP8 arm's kernel time
   is +46.7 ms per step: the FP8 GEMMs save 56 ms, the casts cost back 38.6 ms, the permute backward costs
   36.0 ms more, and a side-stream NCCL all-gather costs 15 ms more.
3. **The remaining host sync (`ks_tensor.tolist()`) is not the cause.** Measured from the profile, it runs
   12 times per step, blocks the host for 0.06 ms in total, and leaves the GPU idle for 1.46 ms per step.
   The GPU is 98% busy in both arms.
4. **Three commits on this branch** make FP8 0.5% faster than bf16 on the one-node proxy of the 8-node
   run (it was 1.4-2.0% slower). All three are bitwise-identical in outputs and gradients:
   - `05e50e5`: multiply by the reciprocal of UE8M0 scales instead of `div_rn` (SM100 only).
   - `07d9c06`: launch the SM100 row-major per-channel cast with 2 warps.
   - `72b78c1`: make the MoE permute's backward a gather instead of an atomic scatter-add. This one speeds
     up bf16 too.

Everything below was measured on the node above with torch 2.13.0+cu130, triton 3.7.1, deep_gemm
2.5.0+891d57b and driver 580.173.02. Scripts, logs and traces are in `~/tmp/fp8-b300/`.

## End to end on one node

Median `time/forward_backward` over steps 5-15 (`~/tmp/bench_summary.py`); step-to-step spread is about
0.01 s. All runs use fake random tokens and real weights truncated to 6 layers (`model.debug.num_layers`).
FP8 means dense plus MoE (`sft_fp8.toml`); "MoE FP8" uses `sft_moe_fp8.toml`, without
`model.quantization`. Configs are `sft_1n_32k.toml` and `sft_1n_256k.toml` in this directory, launched
inside a one-node allocation as `uv run sft @ <base>.toml @ <overlay>.toml --run.name <name>`. The older
code points ran from this worktree's venv with `PYTHONPATH` pointing at a `git archive` of their `src/`.

### H200 reference shape: 32k tokens, cp 1, ep 8, full AC, learned routing

This matches the H200 commit messages except the unknown batch size (`batch_size = 8`, one micro-step).

| Arm | Code | fwd+bwd | step | tokens/s |
|---|---|---|---|---|
| bf16 | `9ad35e6` | 1.979 s | 2.046 s | 128,110 |
| fp8, H200 base | `ae37b35` | 2.045 s | 2.113 s | 124,055 |
| fp8, main | `33e02a7` | 1.983 s | 2.052 s | 127,764 |
| fp8, branch | `9ad35e6` | 1.976 s | 2.041 s | 128,436 |
| fp8, + cast commits | `07d9c06` | 1.955 s | 2.020 s | 129,776 |

### One-node analog of the 8-node run: 256k, cp 8, ep 8, SAC, balanced routing

This is `sft.toml` at 6 layers and `batch_size = 1`, so it has the same 32768 tokens per GPU and the same
attention context as the 8-node run. Its bf16 arm logs 59% MFU (the 8-node run: 57.5%), and its FP8 gap
to bf16 (+2.0%) is close to the 8-node gap (+2.7%). It is a good proxy.

| Arm | Code | fwd+bwd | step | vs bf16 `9ad35e6` |
|---|---|---|---|---|
| bf16 | `9ad35e6` | 2.219 s | 2.287 s | |
| fp8, main | `33e02a7` | 2.266 s | 2.332 s | +2.0% |
| fp8, branch | `9ad35e6` | 2.261 s | 2.327 s | +1.7% |
| fp8, + cast commits | `07d9c06` | 2.255 s | 2.320 s | +1.4% |
| MoE fp8, branch | `9ad35e6` | 2.256 s | 2.322 s | +1.5% |
| MoE fp8, + cast commits | `07d9c06` | 2.252 s | 2.318 s | +1.4% |

With the permute-backward fix (`72b78c1`), FP8 overtakes bf16 on the same code:

| Arm | Code | fwd+bwd | step | vs bf16 `72b78c1` |
|---|---|---|---|---|
| bf16 | `72b78c1` | 2.207 s | 2.274 s | |
| fp8 | `72b78c1` | 2.195 s | 2.263 s | -0.5% |
| MoE fp8 | `72b78c1` | 2.196 s | 2.263 s | -0.5% |

The fix removes 60 ms per step from the FP8 arm (2.255 to 2.195 s) and 12 ms from bf16 (2.219 to
2.207 s). Losses are unchanged to four decimals in all three arms.

Dense FP8 costs almost nothing here: full FP8 is 2-5 ms per step slower than MoE-only FP8.

## Op level: where FP8 time goes on B300

`bench.py` for the base (`47e24bb`, alignment 8) and the branch (alignment 128), one B300 each. The raw
outputs are `results-{base,branch}-primeintellect-tur-r1n8-NVIDIA_B300_SXM6_PC-sm103.txt`. Autograd
fwd+bwd, balanced routing, 32 local experts (ms):

| rows/expert | gate_up bf16 | fp8 base | fp8 branch | down bf16 | fp8 base | fp8 branch |
|---|---|---|---|---|---|---|
| 1536 | 2.91 | 7.33 | 4.76 | 1.49 | 4.44 | 2.73 |
| 6144 | 14.44 | 15.90 | 12.80 | 6.41 | 9.36 | 7.30 |
| 8192 | 19.81 | 20.11 | 16.45 | 9.68 | 11.75 | 9.53 |

The branch helps B300 about as much as it helped H200 at the op level.

Kernel breakdown of one gate_up fwd+bwd at 6144 rows/expert (torch profiler, branch code before the cast
commits): the bf16 grouped GEMMs take 13.1 ms. The FP8 GEMMs take 6.9 ms (2x faster), but the kernels
around them add 5.4 ms:

| kernel | ms per fwd+bwd | after `05e50e5` |
|---|---|---|
| per-channel cast (x, dy for wgrad) | 2.56 | 1.95 |
| per-token cast (x fwd, dy dgrad) | 1.42 | 1.07 |
| per-block weight cast (fwd, dgrad) | 0.62 | 0.64 |
| fp32 grad_weight to bf16 | 0.46 | 0.46 |
| fp32 grad_weight zero-fill | 0.31 | 0.33 |

Roughly, B300's tensor cores are about 2.2x faster than H200's on these GEMMs, while its bandwidth-bound
kernels are only about 1.6x faster (compare the zero-fill: 0.46 ms on H200, 0.29 ms on B300). The casts
therefore eat a larger share of the FP8 saving on B300.

### Cast kernels (commits `05e50e5`, `07d9c06`)

The per-token and per-channel casts divide with `tl.math.div_rn`, which #3623 introduced to match vLLM bit
for bit on SM90. On SM100 the scales are UE8M0 powers of two, so `x * (1 / scale)` is exact and rounds
identically. A sweep (`~/tmp/fp8-b300/cast_sweep.py`, 32 experts x 6144 rows) shows the per-channel cast
was ALU-bound on the division:

| cast | K | div_rn, best config | multiply, best config | copy of the same bytes |
|---|---|---|---|---|
| per-token | 4096 | 0.722 ms | 0.713 ms | 0.735 ms |
| per-channel | 4096 | 1.125 ms | 0.834 ms | 0.735 ms |
| per-channel | 2048 | 0.567 ms | 0.424 ms | 0.372 ms |

All swept configs were bitwise identical to the `div_rn` path. After both commits the activation casts
run at 0.97x to 1.14x the time of a plain copy, so there is little left to gain from them in isolation.
The remaining cast cost could only shrink through fusion, for example reading `dy` once for both its
per-token and per-channel casts, or emitting FP8 straight from the SwiGLU and the permute.

### Dense FP8 linears at 32768 tokens (fwd+bwd kernel time)

| K x N | bf16 | fp8 |
|---|---|---|
| 4096 x 4096 | 1.82 ms | 1.44 ms |
| 4096 x 2048 | 0.92 ms | 0.85 ms |
| 4096 x 1024 | 0.46 ms | 0.60 ms |
| 1024 x 4096 | 0.47 ms | 0.58 ms |

Four per-token casts per linear dominate the narrow shapes. End to end, dense FP8 is about neutral.

## Profile of the 256k step: what actually differs

`trace_path` runs of 5 steps (bf16 and MoE FP8, code `07d9c06`), analyzed with
`~/tmp/fp8-b300/trace_breakdown.py` and `trace_syncs.py` on rank 0, last step. The largest per-kernel
deltas, FP8 minus bf16, per step:

| kernel | bf16 | MoE fp8 | delta |
|---|---|---|---|
| cutlass bf16 grouped GEMM (48 launches) | 146.5 ms | | -146.5 ms |
| DeepGEMM fp8 fwd/dgrad (36) | | 66.2 ms | +66.2 ms |
| DeepGEMM fp8 k-grouped wgrad (12) | | 24.3 ms | +24.3 ms |
| `indexing_backward_kernel` (24) | 38.3 ms | 74.3 ms | +36.0 ms |
| `_grouped_per_channel_fp8_kernel` | | 16.2 ms | +16.2 ms |
| NCCL all-gather (side stream) | 30.8 ms | 45.7 ms | +15.0 ms |
| `_grouped_per_token_fp8_kernel` | | 14.8 ms | +14.8 ms |
| `_grouped_per_block_fp8_kernel` | | 7.6 ms | +7.6 ms |
| all kernels | | | +46.7 ms |

- The GEMMs themselves save 56 ms per step (146.5 vs 90.5 ms), and the casts give back 38.6 ms.
- `indexing_backward_kernel` is `IndexBackward0` of `x[permuted_indices]` in `permute_for_grouped_gemm`.
  Its input shapes in the trace are `[200704, 4096]` into `[196609, 4096]` for FP8 (26.1 ms) against
  `[196864, 4096]` for bf16 (8.0 ms). `max_len` always reserves `experts_per_rank * alignment` slots
  (4096 at alignment 128, 256 at alignment 8), and every padding slot indexes the single trailing zero
  row, so the accumulating scatter piles thousands of atomic adds onto one row. Commit `72b78c1`
  replaces it with a gather through the inverse permutation: bitwise-identical gradients, 1.1 ms instead
  of 9.2 ms (alignment 128) or 3.1 ms (alignment 8) in isolation.
- The other `IndexBackward0` in the list (`[196608, 4096]` into `[32768, 4096]`, 15 ms per step in both
  arms) is the top-k token expansion. It needs a real sum over the 6 copies of each token, so it was left
  alone. It could become a reshape and sum if the expansion were ordered by token.

Host syncs, from `cudaStreamSynchronize` and synchronous `cudaMemcpy` events matched to the kernel timeline
(GPU idle is measured from the last kernel end before the sync returns to the next kernel start):

| | bf16 | MoE fp8 |
|---|---|---|
| step window | 2351 ms | 2393 ms |
| GPU busy, any stream | 2304 ms | 2338 ms |
| GPU idle | 47 ms | 55 ms |
| `grouped_fp8_gemm_backward` syncs (`ks_tensor.tolist()`) | | 12 per step |
| host time blocked in them | | 0.06 ms |
| GPU idle right after them | | 1.46 ms |

The step is GPU-bound. With 12 `tolist()` syncs costing 1.5 ms of GPU idle in a 2.4 s step, removing the
last sync would not be measurable, and removing the other four syncs (what the branch did) could save at
most a few ms per step.

### Why main to branch gains nothing: the branch's own alignment change cancels its savings

Traces of the MoE FP8 arm at main `33e02a7` (A) and at the branch `9ad35e6` (B), same config, per step:

| kernel | main | branch | delta |
|---|---|---|---|
| `indexing_backward_kernel` (permute backward, 24) | 37.6 ms | 73.7 ms | +36.0 ms |
| strided `direct_copy` elementwise (AccumulateGrad) | 87.8 ms (x345) | 69.7 ms (x333) | -18.1 ms |
| `_unpack_grouped_rows_kernel` (36) | 16.5 ms | | -16.5 ms |
| NCCL send/recv + all-gather (side streams) | 220.3 ms | 230.9 ms | +10.7 ms |
| all kernels | | | +19.6 ms |

The branch does remove what it set out to remove: 12 strided AccumulateGrad copies (one per expert weight
per step) and the unpack copies, 34.6 ms per step together, which matches the op-level prediction. Its
switch from 8-row to 128-row alignment, though, raises the permute's padding slots from 256 to 4096 per
layer, and the scatter-add backward on the shared zero row costs 36.0 ms more. The net is zero, which is
what `E2E_B300.md` measured. Syncs drop from 200 to 128 per step (all ranks' collectives included), and
GPU idle right after syncs drops from 19.6 to 15.2 ms, so the sync removal is worth about 4 ms per step.
With `72b78c1` the permute penalty is gone and the branch's savings show up (FP8 60 ms faster per step).

The H200 end-to-end configs and traces are not available here, so I cannot say why the same changes
gained 1.3 s on H200. The permute penalty would have hit H200 too, which makes the H200 gain even
harder to explain from the op-level numbers alone. As `E2E_B300.md` already notes, the H200 gain is far
larger than its op-level cause, which points to something specific to that run (for example allocator
pressure on the 141 GB H200, which the 268 GB B300 would not see). This is a hypothesis, not measured.

## Other angles checked

- **MXFP8 grouped GEMM** (`model.moe.compute.type = "mxfp8"`, Blackwell-native block scaling): the
  installed `prime-kernels` wheel (v0.1.1) does not ship `mxfp8_moe`. Loading it from the
  `deps/prime-kernels` submodule fails on sm103 with a CuTe DSL compiler error ("tma_partition:
  shared-memory operand stride ... not a multiple of the required 128-bit TMA smem alignment"). The kernel
  manifest targets arch 10.0 only. Not usable on this node without upstream work.
- **DeepGEMM zero-fill and downcast**: `k_grouped_fp8_gemm_tn_contiguous` asserts that `c` is given and
  contiguous, so the fp32 zero-fill cannot be skipped with `c=None`, and `d` cannot be bf16. That leaves
  0.8 ms per gate_up fwd+bwd of fixed cost.
- **Existing test failure on SM100**: 12 cases of `test_op_matches_the_quantized_float32_oracle` in
  `tests/unit/train/models/test_fp8_grouped_gemm.py` fail on B300 at `9ad35e6`, before any change here,
  with identical error values after the cast commits. The oracle likely assumes non-UE8M0 scales. The other
  46 tests in that file and `test_fp8_utils.py` pass, and the MoE tests pass after `72b78c1`.
- **Numerics**: losses match to four decimals between the fp8 arms at every code point, and the cast and
  permute commits are bitwise-identical by construction and by test.

## What would move FP8 further on B300

- The permute fix (`72b78c1`) should be validated on the 8-node 256k config. Scaling the one-node
  numbers from 6 to 43 layers predicts roughly 0.4 s per step off the FP8 arm and 0.1 s off bf16, which
  would close most of the 0.45 s gap in `E2E_B300.md`.
- The 128-row alignment still costs 4096 wasted rows per layer in the permute's forward, the grouped GEMM
  tail and the combine. Reserving `max_len` from the actual padded counts would need a host sync, but
  the dispatcher already has the splits on the host after the all-to-all.
- Beyond that, FP8 MoE on B300 is bounded by the casts. Each gate_up fwd+bwd still spends about 4.4 ms
  on activation and weight casts plus 0.8 ms on the wgrad zero-fill and downcast, against a 6.2 ms GEMM
  saving. Fusing the casts into their producers (SwiGLU, permute, all-to-all) is the next lever.
