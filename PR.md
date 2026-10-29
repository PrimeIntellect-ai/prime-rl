This PR runs DeepSeek V4's query RMSNorm + RoPE prefill through vLLM's own fused kernel with one fused Triton backward, and fuses the indexer query RoPE, UE8M0 FP8 quantization and scale fold into a single vLLM kernel. The q forward becomes bit-identical to serving (it was one ulp off on about 6 in a million elements), the whole bf16 `[tokens, 64, 128]` rotated indexer query never materializes, and the q forward plus backward drop from 5.55 ms to 2.52 ms at 31078 tokens per GPU.

The q path calls vLLM's op through `vllm_ops.vllm_op`, which loads `_C_stable_libtorch.abi3.so` with `torch.ops.load_library` so the trainer never imports `vllm` (whose import patches Inductor process-wide), and falls back to today's fp32 norm + rotation outside the contract (bf16, head_dim 512, rope_dim 64, 8 to 128 heads) or when vLLM or the op is missing. The backward rotates the gradient back in fp32 and applies the norm's backward in one pass, recomputing rstd from q, and no longer rounds the rotated gradient to bf16, so it lands at the bf16 rounding floor against fp64. The indexer path calls vLLM's `fused_indexer_q_rope_quant` when its contract holds (amax floor, dtype, layout), rotates, per-head UE8M0-quantizes and folds the scale into the per-head weights in one kernel, with the unfused pair as fallback. Traces from the campaign's tip profiling run (6 layers, 2 micro batches per step, 4 steps) show the fusion: `vllm::deepseek_v4_fused_ops::fusedDeepseekV4QNormRopeKV...` runs 12 calls at median 1.003 ms and `_q_norm_rope_backward_kernel` 6 calls at median 1.458 ms per micro-batch window, `_triton_rope_kernel` drops from 70 calls and 42.34 ms in the main trace to 50 calls and 18.78 ms (the remaining kv-path rope), and `_per_token_fp8_kernel` drops from 210 to 174 calls.

The numbers come from the DSv4 stack A/B campaign on the H200 perf proxy (a DeepSeek V4 Flash checkpoint at 4 and 6 layers, 262144-token rows, cp 8, ep 8, full activation checkpointing, 20 steps of 8 micro batches, force-balanced routing), recomputed from the raw metrics in `~/tmp/dsv4-stack-attribution.md`. This campaign arm runs with the halo flag off. `fb` is `time/forward_backward` per step, the medians are over steps 4 to 20, which is the post-compile window (nothing compiles after step 3), and the all-20-steps median is the second view. A negative delta is faster. MFU is the logged `perf/mfu` fraction of the H200 peak.

| run | layers | fb med s (steps 4-20) | fb med s (all steps) | MFU % | tok/s per GPU | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| main-L4-C2 | 4 | 10.342 | 10.379 | 29.63 | 23779 | 87.71 |
| l4-L4-C2 (this change) | 4 | 10.117 | 10.149 | 30.31 | 24316 | 90.74 |
| main-L6-C2 | 6 | 14.563 | 14.614 | 30.66 | 16890 | 107.74 |
| l4-L6-C2 (this change) | 6 | 14.180 | 14.237 | 31.46 | 17337 | 111.24 |

The campaign arm also carries the earlier optimization levels of this series, so the arm-minus-main delta is cumulative: -2.18% at 4 layers and -2.63% at 6 layers, 9 to 10x the campaign's 0.25% same-code repeat floor, and the chain marginal against the previous arm is -1.41% and -1.78%. The 43-layer projection puts a step at 89.60 s versus main's 93.38 s, about -4%. The cost is peak memory: +3.03 GiB at 4 layers and +3.50 GiB at 6, the largest single memory add of this series that is not the AC policy. Compile bar: identical Dynamo counts (160 at 4 layers, 176 at 6, all step 1), zero graph breaks, and Triton cubins drop from 1020 to 858 because the fusion replaces separate rope, norm and quant kernels.

Numerics: the q forward is bitwise equal to vLLM's kernel (one ulp off today's path on about 6e-6 of elements) and the indexer path is bitwise equal to the unfused pair on both of vLLM's dispatches (CuTeDSL and the Triton fallback), 0 of 254,590,976 quantized values differing over three seeds with scales and folded weights exactly equal. The q forward is not bitwise against the old path, so a full-model forward logprob A/B ran: mean |dlogprob| 0.0451, p50 0.0014, p90 0.1326, mean logprob -0.8538 to -0.8542, exactly at the known floor for non-bitwise forward changes.

The +3.5 GiB peak is a relocated step peak, not a leak. Memory snapshots of both arms show identical allocation sites and counts: the step's peak instant moves to a deeper backward point where two more FSDP all-gather buffers and one more attention-side tensor are live, because the fused path runs fewer, faster kernels in the recomputed block. The change is deterministic within and across runs, and the only long-lived extra is 0.25 GiB from the fused kernel's last outputs.

### Verification

- Unit tests: vLLM bitwise equality (torch.equal), composed and compiled paths, fp64 gradient at the bf16 rounding floor, and the head_dim 32 fallback, plus the indexer test pinning both helper branches bit for bit against the unfused pair.
- Op bench at 31078 tokens per GPU: q forward 2.97 to 1.04 ms, backward 2.58 to 1.48 ms, forward transient 5.8 to 1.9 GiB, indexer rotation plus quantization 1.47 to 0.19 ms, whole `fp8_indexer` call 2.11 to 0.84 ms.
- Full-model forward logprob A/B at the known floor.

### Commits

c1b85e2c9 perf(deepseek-v4): run vLLM's fused q norm + RoPE forward and fuse the backward
89ca11e4c perf(deepseek-v4): fuse the indexer q RoPE and quantization via vLLM's kernel
