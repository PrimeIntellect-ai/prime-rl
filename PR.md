This PR pools the DeepSeek V4 compressors' softmax-gated windows through cuDNN's fused CSA/HCA compressor kernels, one kernel per direction with fp32 softmax weights and products, instead of Inductor's two Triton kernels with a bf16 intermediate. At the trainer's 262k-token micro batches this cuts pooling forward+backward time per layer by about 3x for CSA and 2x for HCA and moves the pooling numerics to one bf16 rounding of an fp32 reference.

The new custom op `prime_rl::dsv4_compressor_pool` (plus backward) covers compress rates 2, 4 and 128 at head_dim 128/512 on SM90+ with bf16 activations, and `model.dsv4_compressor_backend` selects `cudnn` (default) or `torch`. cuDNN's kernels locate each entry's document by scanning every document in the row, so they slow down linearly with the document count while the torch path is flat. The dispatch reads the row's document count from `cu_seqlens` (host metadata, no device sync) and keeps cuDNN up to 256 documents per row, 512 at compress rate 128. In the campaign's main-L6 trace (a 4-step, 2-micro-batch profiling run) the pooling appears as an Inductor family of `triton_*_fused__softmax...` kernels, 14 variants totaling 19.4 ms per micro-batch forward+backward window, and in the l3-halo-L6 trace, whose arm has this backend on, that family is gone and the pooling runs as cuDNN's `kernel_cutlass__compressor_fwd_kernel` and `_bwd_kernel` plus the r128 variants, 14 calls and 0.74 ms summed in the same window (that arm also exchanges CP halos, so it pools one rank's entries rather than the full row).

Docs per row sweep, H200, one 262144-token gathered row, per training layer with full AC in ms (lower is better). The CSA row is pooling fwd + fwd+bwd plus two indexer forwards, the HCA row is fwd + fwd+bwd:

| docs/row | CSA torch | CSA cuDNN | HCA torch | HCA cuDNN |
|---:|---:|---:|---:|---:|
| 2 | 9.46 | 2.58 | 2.55 | 1.24 |
| 64 | 9.46 | 3.55 | 2.55 | 1.30 |
| 256 | 9.46 | 7.11 | 2.55 | 1.50 |
| 512 | 9.46 | 12.04 | 2.55 | 1.83 |
| 1024 | 9.46 | 21.92 | 2.55 | 2.57 |
| 2048 | 9.46 | 41.66 | 2.56 | 4.13 |

The end-to-end numbers come from the DSv4 stack A/B campaign on the 6-layer perf proxy (a DeepSeek V4 Flash checkpoint, 262144-token rows, cp 8, ep 8, full activation checkpointing, 20 steps of 8 micro batches, force-balanced routing), run twice: on H200 with FP8 blockwise, recomputed from the raw metrics in `~/tmp/dsv4-stack-attribution.md`, and on B300 with MXFP8, the production-representative path on that hardware, paired against its own main arm on the same node. `fb` is `time/forward_backward` per step, medians over steps 4 to 20, the post-compile window, and a negative delta is faster. The B300 MFU column divides FP8 work by the dense BF16 peak 2.25e15 FLOP/s while the H200 campaign used a different peak convention, so cross-hardware deltas, not MFU magnitudes, are the comparable quantity. B300 same-protocol repeat floors were 0.04 to 0.12 percent.

| run | fb med s (steps 4-20), H200 | fb med s (steps 4-20), B300 | delta H200 | delta B300 | mfu % (B300) | tok/s per GPU (B300) | peak GiB (B300) |
|---|---:|---:|---:|---:|---:|---:|---:|
| main | 14.557 | 8.227 | - | - | 23.89 | 29937 | 109.17 |
| l2 (this change) | 14.433 | 8.120 | -0.85% | -1.30% | 24.20 | 30336 | 108.02 |

The campaign arm also carries the earlier optimization levels of this series, so the arm-minus-main delta is cumulative: -0.88% at 4 layers and -0.85% at 6 layers. The chain marginal against the previous arm is about -0.7% at both depths, and the campaign's same-code repeat floors are 0.15 to 0.25%, so the delta is 4 to 6x the floor. The 43-layer projection puts a step at 91.53 s versus main's 93.38 s, about -1.5%. Peak memory falls 1.48 GiB at 4 layers and 0.94 GiB at 6 (the fused kernels drop the bf16 intermediate). H200 compile bar: identical Dynamo counts and zero graph breaks in both arms, and the cudnn arm compiles about 140 fewer Triton cubins because Inductor no longer generates the pooling kernels. On B300 the same delta is -1.30 percent at 6 layers with the 43-layer projection at 50.1 s versus main's 51.3 s, and the SM100-native kernels hold their H200-class win at the campaign's one-document-per-row mix without threshold retuning, with the same clean compile bar (zero recompiles, zero graph breaks, cubins at or below the main arm, DeepGEMM variants and TileLang cache files unchanged). In the many-docs regime the doc-count dispatch adds one extra graph variant per compressor layer type, in step 1 only.

Numerics: against an fp64 reference on the same bf16 inputs, the cuDNN entries are one bf16 rounding of vLLM-style fp32 pooling (relative Frobenius 1.66e-3, 99.98% bitwise to `round_bf16` of it) and every gradient at 1.66e-3. Today's Inductor path carries a bf16 intermediate between its two kernels and measures 2.7e-3 to 3.6e-3 on entries (up to 6e-3 on CSA gate gradients), eager torch 3.6e-3 to 9.2e-3. The change is not bitwise against the Inductor path, so a full-model forward logprob A/B ran (32768 tokens per GPU, 8 ranks): median |dlogprob| 0.0013, p90 0.128, mean 0.044, mean logprob -0.8540 vs -0.8538, at the known floor for non-bitwise changes (a torch-vs-torch rerun is bitwise identical), a step toward vLLM's reference numerics rather than away from it.

Scope limit: cuDNN's CSA rate-4 pooling degrades with document count (crossover at about 275 documents per row), so the branch keeps the torch pooling above the dispatch threshold, and the doc-count branch adds the one extra Dynamo graph variant noted above.

### Verification

- `uv run --no-sync pytest tests/unit/train/models/test_deepseek_v4_kernels.py -k compressor` on H200: the fp64-reference test (csa, indexer, hca, eager and compiled, bf16 position bias) plus the packed/unpacked, per-document parity and backward tests.
- Op bench: `bench_ops.py` with the pool case sweep at 2 to 2048 docs per row.
- Proxy A/B: `benchmarks/scripts/dsv4-perf-proxy/run_arm.sh`, stack campaign arms l2-L4-B and l2-L6-B against main-L4-B and main-L6-B.
- Logprob A/B: `prime_rl_logprob_ab.py` dumps and `compare`, forward-only, `--tokens-per-gpu 32768`.

### Commits

2611f3037 feat(deepseek-v4): pool compressor windows with cuDNN's fused CSA/HCA compressor
4bcc9e310 feat(deepseek-v4): keep the torch compressor pooling for rows with many documents
