This PR adds `model.dsv4_cudnn_sink_grad`, which takes DeepSeek V4's sparse-attention sink gradient from what the `cudnn_flashmla` backward already computes instead of rerunning the TileLang preprocess pass, saving about 1.55 ms per layer per 32768-token micro batch on H200 with no new nondeterminism.

#4078 folded the sink into the LSE to keep TileLang's op contract, not because cuDNN lacked sinks, so with the flag the forward keeps FlashMLA's sink-free LSE and cuDNN gets the real sinks. cuDNN's own SM90 `d_sink` issues one fp32 atomic per query and head and measures 1.35 ms per call slower than today's path, so on SM90 the change folds the sink into cuDNN's LSE without `d_sink`, takes cuDNN's internal Delta (dpsum), and reduces dsink in torch from `rowsum(out * dout)`, which keeps dsink deterministic. On SM100 the path uses cuDNN's `d_sink`, exercised by the B300 campaign below.

The numbers come from the DSv4 stack A/B campaign on the 6-layer perf proxy (a DeepSeek V4 Flash checkpoint, 262144-token rows, cp 8, ep 8, full activation checkpointing, 20 steps of 8 micro batches, force-balanced routing), run twice: on H200 with FP8 blockwise, recomputed from the raw metrics in `~/tmp/dsv4-stack-attribution.md`, and on B300 with MXFP8, the production-representative path on that hardware, paired against its own main arm on the same node. The step-time metric is `time/forward_backward`, the table shows medians over steps 4 to 20, the post-compile window, and a negative delta is faster. The B300 MFU column divides FP8 work by the dense BF16 peak 2.25e15 FLOP/s while the H200 campaign used a different peak convention, so cross-hardware deltas, not MFU magnitudes, are the comparable quantity. B300 same-protocol repeat floors were 0.04 to 0.12 percent.

| run | median time/forward_backward s, H200 | median time/forward_backward s, B300 | delta vs main, H200 | delta vs main, B300 | mfu % (B300) | tok/s per GPU (B300) | peak GiB (B300) |
|---|---:|---:|---:|---:|---:|---:|---:|
| main | 14.631 | 8.227 | - | - | 23.89 | 29937 | 109.17 |
| l1 (this change) | 14.540 | 8.212 | -0.62% | -0.18% | 23.97 | 30055 | 108.55 |

This arm is this change alone. The marginal is -0.37% at 4 layers and -0.62% at 6 layers versus the paired main on the same node. The same-code repeat noise floor is +0.026 s (+0.25%, run main-L4-r2-A), so the 4-layer marginal sits at the floor (treat as neutral) and the 6-layer marginal is about 2.5x it. The 43-layer projection (per-layer differencing of the paired 4- and 6-layer runs) puts a step at 92.64 s versus main's 93.38 s, about -0.8%. Peak memory is unchanged at 4 layers and 0.12 GiB lower at 6. Compile bar: the flag adds exactly 3 one-time cuDNN DSA backward kernel compiles (4.07 s total) inside the step-1 warm-up, with Dynamo counts identical to main. On B300 the marginal is -0.18 percent at 6 layers against the same-code floor of 0.04 percent, the native `d_sink` path running on its home hardware, with the TileLang sink-preprocess cache files gone entirely (5 to 0) and the remaining compile bar identical to the B300 main (zero recompiles, zero graph breaks, 3 DeepGEMM variants).

Numerics: the forward is bitwise unchanged, so no logprob A/B was needed. dQ differs from the old path on 0.16% of elements by one bf16 ulp at the same fp64 error, dsink is bitwise after bf16 rounding and deterministic run to run (5 runs at the proxy shape bitwise), and dKV stays inside its existing run-to-run spread (cuDNN fp32 atomics). Workstream op-level medians per layer call at 32768 queries per GPU: CSA backward 32.24 to 30.72 ms, sliding 8.35 to 6.72 ms, HCA unchanged within its own 1 ms drift.

The SM100 path ran on B300 in the same campaign and is step-neutral there (-0.18 percent at 6 layers, within 5x the 0.04 percent floor, the native `d_sink` path exercised on its home hardware), and the H200 4-layer end-to-end marginal is at the noise floor, so the case for the flag rests on the H200 6-layer result and the per-call kernel accounting.

### Verification

- Correctness versus fp64 over 6 cases with fp32 and bf16 sinks, dQ, dKV and dsink checked against the old path and run to run (dsink bitwise over 5 proxy-shape repeats).
- Related unit tests pass on H200, including compile fullgraph with the flag enabled.

### Commits

0f02521d2 feat(deepseek-v4): take the sparse-attention sink gradient from the cuDNN backward
