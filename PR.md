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

The end-to-end numbers come from the DSv4 stack A/B campaign on the 6-layer perf proxy (a DeepSeek V4 Flash checkpoint, 262144-token rows, cp 8, ep 8, full activation checkpointing, 20 steps of 8 micro batches, force-balanced routing), run on H200 with FP8 blockwise and on B300 with MXFP8, each paired against its own main arm on the same node. The step-time metric is `time/forward_backward`, the table shows medians over steps 4 to 20, and a negative delta is faster. B300 repeat floors were 0.04 to 0.12 percent.

| run | median time/forward_backward s, H200 | median time/forward_backward s, B300 | delta vs main, H200 | delta vs main, B300 | tok/s per GPU (B300) | peak GiB (B300) |
|---|---:|---:|---:|---:|---:|---:|
| main | 14.557 | 8.227 | - | - | 29937 | 109.17 |
| l2 (this change) | 14.433 | 8.120 | -0.85% | -1.30% | 30336 | 108.02 |

The arm-minus-main delta is cumulative through the stack: -0.85% on H200 and -1.30% on B300 at 6 layers, with the chain marginal against the previous arm at about -0.7% and the 43-layer projection at about -1.5% (91.5 versus 93.4 s on H200, 50.1 versus 51.3 s on B300). Peak memory falls about 1 GiB and the compile bar is clean on both hardware generations (identical Dynamo counts, zero graph breaks, fewer Triton cubins since Inductor no longer generates the pooling kernels). Numerics: the cuDNN pooling is one bf16 rounding of vLLM-style fp32 pooling, closer to the fp64 reference than today's bf16-intermediate Inductor path, with the full-model logprob A/B at the known floor for non-bitwise changes. One scope limit stands: cuDNN's rate-4 pooling degrades with document count (crossover at about 275 documents per row), so the torch pooling stays above the dispatch threshold, at the cost of one extra step-1 graph variant per compressor layer type.
