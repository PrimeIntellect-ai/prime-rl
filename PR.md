This PR adds `model.dsv4_cudnn_sink_grad`, which takes DeepSeek V4's sparse-attention sink gradient from what the `cudnn_flashmla` backward already computes instead of rerunning the TileLang preprocess pass, saving about 1.55 ms per layer per 32768-token micro batch on H200 with no new nondeterminism.

#4078 folded the sink into the LSE to keep TileLang's op contract, not because cuDNN lacked sinks, so with the flag the forward keeps FlashMLA's sink-free LSE and cuDNN gets the real sinks. cuDNN's own SM90 `d_sink` issues one fp32 atomic per query and head and measures 1.35 ms per call slower than today's path, so on SM90 the change folds the sink into cuDNN's LSE without `d_sink`, takes cuDNN's internal Delta (dpsum), and reduces dsink in torch from `rowsum(out * dout)`, which keeps dsink deterministic. On SM100 the path uses cuDNN's `d_sink` (untested here, no B300).

The numbers come from the DSv4 stack A/B campaign on the H200 perf proxy (a DeepSeek V4 Flash checkpoint at 4 and 6 layers, 262144-token rows, cp 8, ep 8, full activation checkpointing, 20 steps of 8 micro batches, force-balanced routing), recomputed from the raw metrics in `~/tmp/dsv4-stack-attribution.md`. `fb` is `time/forward_backward` per step, the medians are over steps 4 to 20, which is the post-compile window (nothing compiles after step 3), and the all-20-steps median is the second view. A negative delta is faster. MFU is the logged `perf/mfu` fraction of the H200 peak.

| run | layers | fb med s (steps 4-20) | fb med s (all steps) | MFU % | tok/s per GPU | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| main-L4-A | 4 | 10.372 | 10.387 | 29.56 | 23722 | 87.71 |
| l1-L4-A (this change) | 4 | 10.333 | 10.362 | 29.70 | 23832 | 87.71 |
| main-L6-A | 6 | 14.631 | 14.649 | 30.53 | 16830 | 107.74 |
| l1-L6-A (this change) | 6 | 14.540 | 14.591 | 30.70 | 16903 | 107.62 |

This arm is this change alone. The marginal is -0.37% at 4 layers and -0.62% at 6 layers versus the paired main on the same node. The same-code repeat noise floor is +0.026 s (+0.25%, run main-L4-r2-A), so the 4-layer marginal sits at the floor (treat as neutral) and the 6-layer marginal is about 2.5x it. The 43-layer projection (per-layer differencing of the paired 4- and 6-layer runs) puts a step at 92.64 s versus main's 93.38 s, about -0.8%. Peak memory is unchanged at 4 layers and 0.12 GiB lower at 6. Compile bar: the flag adds exactly 3 one-time cuDNN DSA backward kernel compiles (4.07 s total) inside the step-1 warm-up, with Dynamo counts identical to main.

Numerics: the forward is bitwise unchanged, so no logprob A/B was needed. dQ differs from the old path on 0.16% of elements by one bf16 ulp at the same fp64 error, dsink is bitwise after bf16 rounding and deterministic run to run (5 runs at the proxy shape bitwise), and dKV stays inside its existing run-to-run spread (cuDNN fp32 atomics). Workstream op-level medians per layer call at 32768 queries per GPU: CSA backward 32.24 to 30.72 ms, sliding 8.35 to 6.72 ms, HCA unchanged within its own 1 ms drift.

The SM100 path is untested (no B300 here) and the 4-layer end-to-end marginal is at the noise floor, so the case for the flag rests on the 6-layer result and the per-call kernel accounting.

### Verification

- Correctness versus fp64 over 6 cases with fp32 and bf16 sinks, dQ, dKV and dsink checked against the old path and run to run (dsink bitwise over 5 proxy-shape repeats).
- Related unit tests pass on H200, including compile fullgraph with the flag enabled.

### Commits

0f02521d2 feat(deepseek-v4): take the sparse-attention sink gradient from the cuDNN backward
