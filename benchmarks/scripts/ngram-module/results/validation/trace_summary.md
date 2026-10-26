/home/garrett/tmp/ngram-module/ab-validate/main/cell5-4gpu-trace/r0/trace/trace_0.json.gz
5 steps, each ending at an Optimizer.step annotation. Steady steps: [2, 3, 4, 5].

Host-side calls per step (fewer is better for syncs). Lookups: aten::embedding (forward and AC recompute) and aten::embedding_dense_backward, by weight rows x width. a2a: record_param_comms all-to-alls by dtype and the all-to-all op that issued them. Sync point: one CPU op holding a blocking CUDA call or a D2H copy, named by its outermost aten op. An op nested in an op of the same name (e.g. under AC's dispatch mode) counts once. The n-gram table forward runs from its FSDP pre-forward hook to its post-forward hook.

| event                                                  | step 1 | step 2 | step 3 | step 4 | step 5 |
| ------------------------------------------------------ | -----: | -----: | -----: | -----: | -----: |
| a2a BFloat16 _c10d_functional::all_to_all_single       |     12 |     12 |     12 |     12 |     12 |
| a2a BFloat16 prime_rl_collectives::all_to_all_single   |     96 |     96 |     96 |     96 |     96 |
| a2a Long _c10d_functional::all_to_all_single           |     16 |     16 |     16 |     16 |     16 |
| a2a Long prime_rl_collectives::all_to_all_single_equal |     32 |     32 |     32 |     32 |     32 |
| aten::_local_scalar_dense                              |    185 |    194 |    194 |    194 |    194 |
| aten::bincount                                         |      8 |      8 |      8 |      8 |      8 |
| aten::item                                             |    185 |    194 |    194 |    194 |    194 |
| cudaDeviceSynchronize                                  |    484 |      1 |      1 |      1 |      1 |
| cudaMemcpyAsync DtoH (Device -> Pageable)              |     28 |     28 |     28 |     28 |     28 |
| cudaMemcpyAsync DtoH (Device -> Pinned)                |    153 |    162 |    162 |    162 |    162 |
| cudaStreamSynchronize                                  |    381 |    199 |    199 |    199 |    199 |
| lookup bwd embed_tokens 248320x1024                    |      2 |      2 |      2 |      2 |      2 |
| lookup bwd n-gram 320512x160                           |      2 |      2 |      2 |      2 |      2 |
| lookup bwd n-gram 321152x160                           |      2 |      2 |      2 |      2 |      2 |
| lookup fwd embed_tokens 248320x1024                    |      2 |      2 |      2 |      2 |      2 |
| lookup fwd n-gram 320512x160                           |      4 |      4 |      4 |      4 |      4 |
| lookup fwd n-gram 321152x160                           |      4 |      4 |      4 |      4 |      4 |
| sync point in CausalConv1dFunction                     |      1 |      0 |      0 |      0 |      0 |
| sync point in CausalConv1dFunctionBackward             |      1 |      0 |      0 |      0 |      0 |
| sync point in ChunkGatedDeltaRuleFunction              |      1 |      0 |      0 |      0 |      0 |
| sync point in ChunkGatedDeltaRuleFunctionBackward      |      1 |      0 |      0 |      0 |      0 |
| sync point in LayerNormGatedFunction                   |      1 |      0 |      0 |      0 |      0 |
| sync point in LayerNormGatedFunctionBackward           |      1 |      0 |      0 |      0 |      0 |
| sync point in aten::bincount                           |     16 |     16 |     16 |     16 |     16 |
| sync point in aten::copy_                              |      2 |      2 |      2 |      2 |      2 |
| sync point in aten::is_nonzero                         |      2 |      2 |      2 |      2 |      2 |
| sync point in aten::item                               |     15 |     24 |     24 |     24 |     24 |
| sync point in aten::repeat_interleave                  |    128 |    128 |    128 |    128 |    128 |
| sync point in aten::to                                 |     26 |     27 |     27 |     27 |     27 |
| sync point inside an n-gram table forward              |     20 |     20 |     20 |     20 |     20 |

GPU kernel time launched between each module's FSDP pre and post hooks per step, median over steady steps (ms, lower is better). Spans: hook pairs per step (micro-batches, plus AC recomputes in forward). A layer's backward includes its AC recompute.

| module         | pass     | spans | GPU ms |
| -------------- | -------- | ----: | -----: |
| layer 0        | backward |   2.0 |  32.25 |
| layer 0        | forward  |   2.0 |  13.04 |
| layer 1        | backward |   2.0 |  50.38 |
| layer 1        | forward  |   2.0 |  15.68 |
| layer 2        | backward |   2.0 |  30.21 |
| layer 2        | forward  |   2.0 |   9.54 |
| layer 3        | backward |   2.0 |  28.56 |
| layer 3        | forward  |   2.0 |   7.80 |
| layer 4        | backward |   2.0 |  28.18 |
| layer 4        | forward  |   2.0 |   7.70 |
| layer 5        | backward |   2.0 |  52.41 |
| layer 5        | forward  |   2.0 |  14.88 |
| layer 6        | backward |   2.0 |  26.87 |
| layer 6        | forward  |   2.0 |   9.37 |
| layer 7        | backward |   2.0 |  19.29 |
| layer 7        | forward  |   2.0 |  10.93 |
| n-gram table 1 | backward |   2.0 |  56.85 |
| n-gram table 1 | forward  |   4.0 |   8.63 |
| n-gram table 5 | backward |   2.0 | 215.83 |
| n-gram table 5 | forward  |   4.0 |   8.95 |
